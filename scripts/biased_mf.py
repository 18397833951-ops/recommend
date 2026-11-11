"""联合训练用户/电影向量与偏置，每轮验证并保存最佳模型。"""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import time

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from baselines import check_input
from bias import evaluate as evaluate_saved
from prepare_data import fingerprint, log, start_spark


@dataclass
class Ratings:
    users: np.ndarray
    movies: np.ndarray
    ratings: np.ndarray
    timestamps: np.ndarray | None = None

    def __len__(self):
        return len(self.ratings)


class BiasedMF(nn.Module):
    def __init__(self, users, movies, rank, global_mean, init_std):
        super().__init__()
        self.user_factors = nn.Embedding(users, rank)
        self.movie_factors = nn.Embedding(movies, rank)
        self.user_bias = nn.Embedding(users, 1)
        self.movie_bias = nn.Embedding(movies, 1)
        for embedding in (self.user_factors, self.movie_factors):
            nn.init.normal_(embedding.weight, std=init_std)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        self.register_buffer("global_mean", torch.tensor(global_mean, dtype=torch.float32))
        self.register_buffer("known_users", torch.zeros(users, dtype=torch.bool))
        self.register_buffer("known_movies", torch.zeros(movies, dtype=torch.bool))

    def known(self, users, movies):
        u = users.clamp(0, len(self.known_users) - 1)
        i = movies.clamp(0, len(self.known_movies) - 1)
        return ((users > 0) & (users < len(self.known_users)) & self.known_users[u],
                (movies > 0) & (movies < len(self.known_movies)) & self.known_movies[i])

    def parts(self, users, movies):
        known_u, known_i = self.known(users, movies)
        u = users.clamp(0, len(self.known_users) - 1)
        i = movies.clamp(0, len(self.known_movies) - 1)
        return (self.user_factors(u) * known_u[:, None], self.movie_factors(i) * known_i[:, None],
                self.user_bias(u).squeeze(1) * known_u, self.movie_bias(i).squeeze(1) * known_i)

    def score(self, parts):
        p, q, bu, bi = parts
        return self.global_mean + bu + bi + (p * q).sum(dim=1)

    def forward(self, users, movies):
        return self.score(self.parts(users, movies))

    def loss(self, users, movies, ratings, regularization):
        parts = self.parts(users, movies)
        p, q, bu, bi = parts
        mse = (self.score(parts) - ratings).square().mean()
        penalty = (p.square().sum(1) + q.square().sum(1) + bu.square() + bi.square()).mean()
        return mse + regularization * penalty


def validate_config(cfg):
    for name in ("rank", "max_epochs", "batch_size", "patience", "num_threads"):
        value = cfg.model[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须为正整数。")
    for name in ("learning_rate", "init_std", "regularization"):
        value = cfg.model[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数。")
    if not isinstance(cfg.data.read_partitions, int) or cfg.data.read_partitions < 1:
        raise ValueError("read_partitions 必须为正整数。")
    if (not all(math.isfinite(v) for v in (cfg.model.rating_min, cfg.model.rating_max))
            or cfg.model.rating_min >= cfg.model.rating_max):
        raise ValueError("评分范围无效。")
    if list(cfg.evaluation.splits) != ["validation", "test"]:
        raise ValueError("需要 evaluation.splits=[validation,test]。")
    device = torch.device(cfg.model.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("请选择可用的 cpu 或 cuda 设备。")


def load_ratings(spark, root, phase, expected, partitions):
    """只在内存保留紧凑数值数组；不再落盘复制训练数据。"""
    from pyspark.sql import functions as F

    users = np.empty(expected, dtype=np.int32)
    movies = np.empty(expected, dtype=np.int32)
    values = np.empty(expected, dtype=np.float32)
    timestamps = None if phase == "train" else np.empty(expected, dtype=np.int64)
    columns = [F.col("userId").cast("int"), F.col("movieId").cast("int"), F.col("rating").cast("float")]
    if timestamps is not None:
        columns.append("timestamp")
    frame = spark.read.parquet(str(root)).where(F.col("split") == phase).select(*columns).repartition(partitions)
    log(f"Loading {phase}: {expected:,} ratings")
    count = 0
    for row in frame.toLocalIterator(prefetchPartitions=False):
        if count >= expected:
            raise ValueError(f"{phase} 记录多于划分报告。")
        users[count], movies[count], values[count] = row[0], row[1], row[2]
        if timestamps is not None:
            timestamps[count] = row[3]
        count += 1
    if count != expected:
        raise ValueError(f"{phase} 记录数与划分报告不一致。")
    return Ratings(users, movies, values, timestamps)


def batch_tensors(data, indices, device):
    return (torch.as_tensor(data.users[indices], dtype=torch.long, device=device),
            torch.as_tensor(data.movies[indices], dtype=torch.long, device=device),
            torch.as_tensor(data.ratings[indices], dtype=torch.float32, device=device))


@torch.no_grad()
def evaluate_model(model, data, cfg, prediction_path=None):
    model.eval()
    device = model.global_mean.device
    score = {"rows": 0, "squared_error_sum": 0.0, "user_fallback_rows": 0,
             "movie_fallback_rows": 0, "fallback_rows": 0}
    stream = gzip.open(prediction_path, "wt", newline="", encoding="utf-8", compresslevel=3) if prediction_path else None
    try:
        writer = csv.writer(stream) if stream else None
        if writer:
            if data.timestamps is None:
                raise ValueError("保存评测预测需要原时间戳。")
            writer.writerow(["userId", "movieId", "rating", "timestamp", "raw_prediction", "prediction", "user_fallback", "movie_fallback"])
        for start in range(0, len(data), cfg.batch_size):
            end = min(start + cfg.batch_size, len(data))
            users, movies, ratings = batch_tensors(data, slice(start, end), device)
            raw = model(users, movies)
            if not torch.isfinite(raw).all() or not torch.isfinite(ratings).all():
                raise ValueError("评分或预测存在 NaN、无穷值。")
            prediction = raw.clamp(cfg.rating_min, cfg.rating_max) if cfg.clip_predictions else raw
            known_u, known_i = model.known(users, movies)
            score["rows"] += end - start
            score["squared_error_sum"] += (prediction.double() - ratings.double()).square().sum().item()
            score["user_fallback_rows"] += (~known_u).sum().item()
            score["movie_fallback_rows"] += (~known_i).sum().item()
            score["fallback_rows"] += (~known_u | ~known_i).sum().item()
            if writer:
                writer.writerows(zip(data.users[start:end].tolist(), data.movies[start:end].tolist(),
                                     data.ratings[start:end].tolist(), data.timestamps[start:end].tolist(),
                                     raw.cpu().tolist(), prediction.cpu().tolist(),
                                     (~known_u).int().cpu().tolist(), (~known_i).int().cpu().tolist()))
    finally:
        if stream:
            stream.close()
    if not score["rows"]:
        raise ValueError("评测记录为空。")
    score["rmse"] = math.sqrt(score["squared_error_sum"] / score["rows"])
    return score


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = BiasedMF(checkpoint["num_users"], checkpoint["num_movies"], checkpoint["rank"], checkpoint["global_mean"], 0.05).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def fit_model(train, validation, cfg, path, report, on_epoch=None):
    """一个模型连续训练，每轮验证，按验证 RMSE 保存最佳权重。"""
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.num_threads)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(cfg.seed)
    global_mean = float(np.mean(train.ratings, dtype=np.float64))
    model = BiasedMF(int(train.users.max()) + 1, int(train.movies.max()) + 1, cfg.rank, global_mean, cfg.init_std)
    model.known_users[torch.from_numpy(train.users.astype(np.int64))] = True
    model.known_movies[torch.from_numpy(train.movies.astype(np.int64))] = True
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    report["training"] = {"rows": len(train), "global_mean": global_mean,
                          "users": model.known_users.sum().item(), "movies": model.known_movies.sum().item()}
    report["history"] = []
    report["best_epoch"] = 0
    best, stale = float("inf"), 0
    path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, cfg.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        order = rng.permutation(len(train))
        loss_sum = 0.0
        for start in range(0, len(train), cfg.batch_size):
            indices = order[start:start + cfg.batch_size]
            users, movies, ratings = batch_tensors(train, indices, cfg.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(users, movies, ratings, cfg.regularization)
            if not torch.isfinite(loss):
                raise ValueError("训练损失出现 NaN 或无穷值。")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        del order
        score = evaluate_model(model, validation, cfg)
        row = {"epoch": epoch, "online_training_loss": loss_sum / len(train),
               "validation": score, "elapsed_seconds": round(time.perf_counter() - started, 2)}
        report["history"].append(row)
        if score["rmse"] < best:
            best, stale = score["rmse"], 0
            report["best_epoch"] = epoch
            torch.save({"state_dict": model.state_dict(), "num_users": len(model.known_users),
                        "num_movies": len(model.known_movies), "rank": cfg.rank, "global_mean": global_mean,
                        "epoch": epoch, "validation_rmse": best}, path)
        else:
            stale += 1
        log(f"epoch={epoch}/{cfg.max_epochs}; validation_RMSE={score['rmse']:.8f}; best_epoch={report['best_epoch']}")
        if on_epoch:
            on_epoch()
        if stale >= cfg.patience:
            break
    report["epochs_trained"] = len(report["history"])
    return load_model(path, cfg.device)[0]


def write_report(report, output):
    (output / "biased_mf.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 联合训练的带偏置矩阵分解", "", f"状态：{report['status']}。", "",
             "预测 = 训练全局均值 + 用户偏置 + 电影偏置 + 用户向量与电影向量的点积。",
             "全局均值固定，向量和偏置从头初始化，使用 Adam 联合更新；一次运行只使用配置中的一组超参数。",
             "训练损失 = 每批原始评分预测 MSE + regularization × 每批出现的两组向量及两个偏置的平方和均值。",
             "每轮评估完整验证集并保存最低 RMSE 权重；连续 patience 轮没有改善则提前停止。仅在选定权重后读取本地测试评分。",
             "未知 ID 对应的向量、偏置为零，已知一侧的偏置继续使用；两侧均未知时使用训练全局均值。", ""]
    cfg = report["configuration"]["model"]
    lines += [f"维数：{cfg['rank']}；学习率：{cfg['learning_rate']}；正则系数：{cfg['regularization']}；设备：{cfg['device']}。",
              f"最大轮数：{cfg['max_epochs']}；实际训练：{report.get('epochs_trained', len(report.get('history', [])))}；最佳轮次：{report.get('best_epoch', '待定')}。",
              f"最终预测{'截断到 ' + str(cfg['rating_min']) + '～' + str(cfg['rating_max']) if cfg['clip_predictions'] else '不截断'}，不四舍五入。", "",
              "| 划分 | 记录数 | RMSE | 用户未见 | 电影未见 | 回退记录 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report.get("metrics", []):
        label = {"validation": "验证集", "test": "本地测试集"}[row["split"]]
        lines.append(f"| {label} | {row['rows']:,} | {row['rmse']:.8f} | {row['user_fallback_rows']:,} | {row['movie_fallback_rows']:,} | {row['fallback_rows']:,} |")
    lines += ["", "models/best.pt 保存最佳轮次的权重、训练 ID 掩码和全局均值；models/parameters.json 保存配置与最佳轮次。",
              "predictions/*.csv.gz 保存用户、电影、真实评分、时间戳、原始和最终预测及回退标记；保存后通过 Spark 读回核对。",
              "biased_mf.json 的 history 记录逐轮验证误差，online_training_loss 为本轮各批更新前的损失加权均值。",
              f"总耗时：{report.get('elapsed_seconds', 0):.2f} 秒。"]
    if "error" in report:
        lines.append(f"失败原因：{report['error']}")
    (output / "biased_mf.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_biased_mf(cfg, output):
    from pyspark.sql import functions as F

    validate_config(cfg)
    output = Path(output).resolve()
    for source_path in (cfg.paths.raw_dir, cfg.paths.splits_dir):
        source_path = Path(source_path).expanduser().resolve()
        if output.is_relative_to(source_path) or source_path.is_relative_to(output):
            raise ValueError("输出与输入目录不可重叠。")
    if any((output / name).exists() for name in ("models", "predictions", "biased_mf.json")):
        raise FileExistsError(f"结果已存在，不会覆盖：{output}")
    started = time.perf_counter()
    root, source = check_input(cfg)
    expected = {phase: sum(row["count"] for row in source["counts"] if row["split"] == phase)
                for phase in ("train", "validation", "test")}
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "generated_at": datetime.now(timezone.utc).isoformat(), "source": source,
              "torch_version": str(torch.__version__),
              "configuration": OmegaConf.to_container(cfg, resolve=True)}
    write_report(report, output)
    try:
        spark = start_spark(cfg, "JointMFTrainingInput")
        try:
            report["spark_version"] = spark.version
            train = load_ratings(spark, root, "train", expected["train"], cfg.data.read_partitions)
            validation = load_ratings(spark, root, "validation", expected["validation"], cfg.data.read_partitions)
        finally:
            spark.stop()
        best_path = output / "models/best.pt"
        model = fit_model(train, validation, cfg.model, best_path, report, lambda: write_report(report, output))
        del train
        predictions = output / "predictions"
        predictions.mkdir()
        validation_score = evaluate_model(model, validation, cfg.model, predictions / "validation.csv.gz")
        best_score = report["history"][report["best_epoch"] - 1]["validation"]["rmse"]
        if not math.isclose(validation_score["rmse"], best_score, abs_tol=1e-12, rel_tol=0):
            raise ValueError("最佳权重读回后的验证误差不一致。")
        del validation
        log("Best epoch frozen; reading local test ratings")
        spark = start_spark(cfg, "JointMFFinalEvaluation")
        try:
            test = load_ratings(spark, root, "test", expected["test"], cfg.data.read_partitions)
            test_score = evaluate_model(model, test, cfg.model, predictions / "test.csv.gz")
            del test
            report["metrics"] = []
            schema = "userId int, movieId int, rating double, timestamp long, raw_prediction double, prediction double, user_fallback int, movie_fallback int"
            for phase, original in (("validation", validation_score), ("test", test_score)):
                saved = spark.read.schema(schema).option("header", True).option("mode", "FAILFAST").csv(str(predictions / f"{phase}.csv.gz"))
                for flag in ("user_fallback", "movie_fallback"):
                    saved = saved.withColumn(flag, F.col(flag).cast("boolean"))
                score = evaluate_saved(saved)
                if score["rows"] != expected[phase] or not math.isclose(score["rmse"], original["rmse"], abs_tol=1e-10, rel_tol=0):
                    raise ValueError("保存预测读回后的数量或误差不一致。")
                report["metrics"].append({"split": phase, **score})
        finally:
            spark.stop()
        parameters = {"training": report["training"], "best_epoch": report["best_epoch"],
                      "validation_rmse": best_score, "model": OmegaConf.to_container(cfg.model, resolve=True)}
        (output / "models/parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
        files = [best_path, output / "models/parameters.json", *sorted(predictions.glob("*.csv.gz"))]
        report["output_files"] = [fingerprint(path) for path in files]
        report["status"] = "passed"
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        write_report(report, output)


@hydra.main(version_base="1.3", config_path="../configs", config_name="biased_mf")
def main(cfg):
    output = Path(HydraConfig.get().runtime.output_dir)
    report = run_biased_mf(cfg, output)
    log(f"BIASED_MF_STATUS={report['status']}; report={output / 'biased_mf.md'}")


if __name__ == "__main__":
    main()
