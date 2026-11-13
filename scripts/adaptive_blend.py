"""训练集训练 4→16→1 融合网络，验证选轮次，最后评估本地测试。"""

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
from biased_mf import load_model as load_mf, load_ratings
from blend import check_runs
from prepare_data import fingerprint, log, start_spark


@dataclass
class BlendData:
    features: np.ndarray
    ratings: np.ndarray
    predictions: np.ndarray
    users: np.ndarray | None = None
    movies: np.ndarray | None = None
    timestamps: np.ndarray | None = None

    def __len__(self):
        return len(self.ratings)


class AdaptiveBlend(nn.Module):
    """输入两个预测分数、两种 log1p 评分数量，输出 ALS 权重。"""
    def __init__(self, hidden_size, mean, scale):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32).clamp_min(1e-6))
        self.network = nn.Sequential(nn.Linear(4, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        # 从 50/50 开始，训练数据决定如何偏向两个模型。
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return torch.sigmoid(self.network((features - self.mean) / self.scale).squeeze(1))


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = AdaptiveBlend(checkpoint["hidden_size"], checkpoint["state_dict"]["mean"],
                          checkpoint["state_dict"]["scale"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def validate_config(cfg):
    for name in ("hidden_size", "batch_size", "max_epochs", "patience", "num_threads"):
        v = cfg.model[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} 必须为正整数。")
    if not math.isfinite(cfg.model.learning_rate) or cfg.model.learning_rate <= 0:
        raise ValueError("学习率必须为有限正数。")
    if not math.isfinite(cfg.model.rating_min) or not math.isfinite(cfg.model.rating_max) or cfg.model.rating_min >= cfg.model.rating_max:
        raise ValueError("评分范围无效。")
    if isinstance(cfg.data.read_partitions, bool) or not isinstance(cfg.data.read_partitions, int) or cfg.data.read_partitions < 1:
        raise ValueError("read_partitions 必须为正整数。")
    device = torch.device(cfg.model.device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("请选择可用的 cpu 或 cuda。")


def verify_files(report, files):
    manifest = {Path(f["path"]).resolve(): f for f in report["output_files"]}
    for file in files:
        if file not in manifest or fingerprint(file) != manifest[file]:
            raise ValueError(f"输入文件指纹不一致：{file}")


def prepare_training(spark, ratings_root, expected, runs, cfg, output):
    """基础模型保持不变，用它们对原训练集的预测构造融合训练输入。"""
    data = load_ratings(spark, ratings_root, "train", expected, cfg.data.read_partitions)
    users = np.bincount(data.users)
    movies = np.bincount(data.movies)
    np.savez_compressed(output / "models/counts.npz", users=users, movies=movies)
    als_path = runs["als"][0] / "models/als"
    factors = []
    for name, size in (("userFactors", len(users)), ("itemFactors", len(movies))):
        files = sorted((als_path / name).rglob("*.parquet"))
        verify_files(runs["als"][1], files)
        if not files:
            raise ValueError("缺少 ALS 特征文件。")
        table = None
        known = np.zeros(size, dtype=bool)
        for row in spark.read.parquet(str(als_path / name)).toLocalIterator():
            if table is None:
                table = np.zeros((size, len(row.features)), dtype=np.float32)
            if not 0 <= row.id < size:
                raise ValueError("ALS ID 超出当前训练集范围。")
            table[row.id] = row.features
            known[row.id] = True
        if table is None or not known[np.flatnonzero(users if name == "userFactors" else movies)].all():
            raise ValueError("训练集中存在缺失的 ALS 特征。")
        factors.append(table)
    mf_path = runs["mf"][0] / "models/best.pt"
    verify_files(runs["mf"][1], [mf_path])
    mf, _ = load_mf(mf_path)
    mf.requires_grad_(False)
    features = np.empty((len(data), 4), dtype=np.float32)
    log(f"Computing frozen base predictions for {len(data):,} training ratings")
    with torch.no_grad():
        for start in range(0, len(data), cfg.model.batch_size):
            end = min(start + cfg.model.batch_size, len(data))
            u, i = data.users[start:end], data.movies[start:end]
            tu, ti = torch.from_numpy(u.astype(np.int64)), torch.from_numpy(i.astype(np.int64))
            known_u, known_i = mf.known(tu, ti)
            if not known_u.all() or not known_i.all():
                raise ValueError("训练集中存在 MF 未见 ID。")
            features[start:end, 0] = np.sum(factors[0][u] * factors[1][i], axis=1, dtype=np.float32)
            features[start:end, 1] = mf(tu, ti).numpy()
            features[start:end, 2] = np.log1p(users[u])
            features[start:end, 3] = np.log1p(movies[i])
    np.clip(features[:, :2], cfg.model.rating_min, cfg.model.rating_max, out=features[:, :2])
    if not np.isfinite(features).all():
        raise ValueError("训练特征存在非法数值。")
    return BlendData(features, data.ratings, features[:, :2]), users, movies


def load_phase(spark, root, report, phase, counts, expected, cfg):
    """验证/测试复用固定融合的已对齐预测；数量特征只来自训练集。"""
    path = root / "predictions" / f"split={phase}"
    files = sorted(path.rglob("*.parquet"))
    verify_files(report, files)
    recorded = {Path(f["path"]).resolve() for f in report["output_files"] if Path(f["path"]).resolve().is_relative_to(path)}
    if not files or set(files) != recorded:
        raise ValueError("融合预测文件清单不一致。")
    features = np.empty((expected, 4), np.float32)
    predictions = np.empty((expected, 2), np.float64)
    ratings = np.empty(expected, np.float64)
    ids = np.empty((expected, 2), np.int32)
    timestamps = np.empty(expected, np.int64)
    frame = spark.read.parquet(str(path))
    quality = frame.selectExpr("count(*) as rows", "count(distinct struct(userId,movieId)) as pairs").first()
    if quality.rows != expected or quality.pairs != expected:
        raise ValueError("融合预测存在重复或缺失组合。")
    log(f"Loading adaptive-blend {phase}: {expected:,} rows")
    n = 0
    for row in frame.select("userId", "movieId", "timestamp", "rating", "als_prediction", "mf_prediction").repartition(cfg.data.read_partitions).toLocalIterator():
        if n >= expected or row.userId is None or row.movieId is None or row.timestamp is None:
            raise ValueError("预测行数或 ID/时间戳无效。")
        u, i = row.userId, row.movieId
        cu = counts[0][u] if 0 <= u < len(counts[0]) else 0
        ci = counts[1][i] if 0 <= i < len(counts[1]) else 0
        predictions[n] = row.als_prediction, row.mf_prediction
        features[n] = row.als_prediction, row.mf_prediction, np.log1p(cu), np.log1p(ci)
        ratings[n], ids[n], timestamps[n] = row.rating, (u, i), row.timestamp
        n += 1
    if n != expected or not np.isfinite(features).all() or not np.isfinite(ratings).all():
        raise ValueError("预测数量或数值无效。")
    if not np.isfinite(predictions).all() or np.any(predictions < cfg.model.rating_min) or np.any(predictions > cfg.model.rating_max):
        raise ValueError("基础预测分数超出范围。")
    for model, column in (("als", 0), ("mf", 1)):
        rmse = float(np.sqrt(np.mean((predictions[:, column] - ratings)**2)))
        original = next(r["rmse"] for r in report["metrics"] if r["model"] == model and r["split"] == phase)
        if not math.isclose(rmse, original, rel_tol=0, abs_tol=1e-9):
            raise ValueError("基础预测读回误差不一致。")
    return BlendData(features, ratings, predictions, ids[:, 0], ids[:, 1], timestamps)


@torch.no_grad()
def evaluate(model, data, cfg, fixed_alpha, prediction_path=None):
    model.eval()
    device = model.mean.device
    sse = fixed_sse = alpha_sum = 0.0
    alpha_min, alpha_max = 1.0, 0.0
    stream = gzip.open(prediction_path, "wt", newline="", encoding="utf-8", compresslevel=3) if prediction_path else None
    try:
        writer = csv.writer(stream) if stream else None
        if writer:
            writer.writerow(["userId", "movieId", "timestamp", "rating", "als_prediction", "mf_prediction", "als_weight", "prediction", "fixed_prediction"])
        for start in range(0, len(data), cfg.batch_size):
            end = min(start + cfg.batch_size, len(data))
            alpha = model(torch.as_tensor(data.features[start:end], device=device)).cpu().numpy().astype(np.float64)
            a, b = data.predictions[start:end].T
            prediction = b + alpha * (a - b)
            fixed = b + fixed_alpha * (a - b)
            if not np.isfinite(prediction).all() or not np.isfinite(alpha).all():
                raise ValueError("融合预测出现非法数值。")
            sse += float(np.sum((prediction - data.ratings[start:end])**2))
            fixed_sse += float(np.sum((fixed - data.ratings[start:end])**2))
            alpha_sum += float(alpha.sum())
            alpha_min, alpha_max = min(alpha_min, float(alpha.min())), max(alpha_max, float(alpha.max()))
            if writer:
                writer.writerows(zip(data.users[start:end], data.movies[start:end], data.timestamps[start:end],
                                     data.ratings[start:end], a, b, alpha, prediction, fixed))
    finally:
        if stream:
            stream.close()
    if not len(data):
        raise ValueError("评测数据为空。")
    return {"rows": len(data), "squared_error_sum": sse, "rmse": math.sqrt(sse / len(data)),
            "fixed_rmse": math.sqrt(fixed_sse / len(data)),
            "als_weight": {"min": alpha_min, "mean": alpha_sum / len(data), "max": alpha_max}}


def fit_model(train, validation, cfg, fixed_alpha, path, report, on_epoch=None):
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.num_threads)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(cfg.seed)
    # 标准化统计仅由原训练集计算，并随模型保存。
    model = AdaptiveBlend(cfg.hidden_size, train.features.mean(0, dtype=np.float64),
                          train.features.std(0, dtype=np.float64)).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    report["history"] = []
    report["training_rows"] = len(train)
    report["best_epoch"] = 0
    initial = evaluate(model, validation, cfg, fixed_alpha)
    report["initial_validation"] = initial
    best, stale = initial["rmse"], 0
    path.parent.mkdir(parents=True, exist_ok=True)
    def save(epoch, score):
        torch.save({"state_dict": model.state_dict(), "hidden_size": cfg.hidden_size,
                    "epoch": epoch, "validation_rmse": score, "features": ["als_prediction", "mf_prediction", "log1p_user_count", "log1p_movie_count"]}, path)
    save(0, best)
    for epoch in range(1, cfg.max_epochs + 1):
        started = time.perf_counter()
        model.train()
        order = rng.permutation(len(train))
        loss_sum = 0.0
        for start in range(0, len(train), cfg.batch_size):
            indices = order[start:start + cfg.batch_size]
            x = torch.as_tensor(train.features[indices], device=cfg.device)
            y = torch.as_tensor(train.ratings[indices], dtype=torch.float32, device=cfg.device)
            optimizer.zero_grad(set_to_none=True)
            alpha = model(x)
            prediction = x[:, 1] + alpha * (x[:, 0] - x[:, 1])
            loss = (prediction - y).square().mean()
            if not torch.isfinite(loss):
                raise ValueError("融合训练损失无效。")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        del order
        score = evaluate(model, validation, cfg, fixed_alpha)
        report["history"].append({"epoch": epoch, "training_mse": loss_sum / len(train), "validation": score,
                                  "elapsed_seconds": round(time.perf_counter() - started, 2)})
        if score["rmse"] < best:
            best, stale = score["rmse"], 0
            report["best_epoch"] = epoch
            save(epoch, best)
        else:
            stale += 1
        log(f"adaptive epoch={epoch}; validation_RMSE={score['rmse']:.8f}; best_epoch={report['best_epoch']}")
        if on_epoch:
            on_epoch()
        if stale >= cfg.patience:
            break
    report["epochs_trained"] = len(report["history"])
    return load_model(path, cfg.device)[0]


def write_report(report, output):
    (output / "adaptive_blend.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 自适应权重融合网络", "", f"状态：{report['status']}。", "",
             "输入：ALS 分数、矩阵分解分数、log1p 用户训练评分数、log1p 电影训练评分数。",
             "结构：4 → 隐藏层（ReLU）→ 1（Sigmoid）；输出 alpha，最终预测为 alpha × ALS + (1-alpha) × MF。",
             "仅用原 split=train 训练网络和计算标准化统计，原 validation 选最佳轮次，原 test 在恢复最佳权重后读取。",
             "基础模型保持固定。训练输入来自基础模型对已见训练记录的预测，验证和测试输入来自未见记录的预测。", "",
             f"最佳轮次：{report.get('best_epoch', '待定')}；实际训练轮次：{report.get('epochs_trained', len(report.get('history', [])))}。", ""]
    for name, title in (("fixed", "固定权重融合"), ("adaptive", "自适应权重融合")):
        lines += [f"## {title}", "", "| 划分 | 记录数 | RMSE |", "| --- | ---: | ---: |"]
        for row in report.get("metrics", []):
            score = row["fixed_rmse"] if name == "fixed" else row["rmse"]
            lines.append(f"| {'验证集' if row['split']=='validation' else '本地测试集'} | {row['rows']:,} | {score:.8f} |")
        lines.append("")
    if "selected_method" in report:
        lines += [f"验证集选定方案：{report['selected_method']}（本地测试不参与选择）。", ""]
    if report.get("best_epoch") == 0:
        lines += ["第 0 轮是尚未训练的 50/50 初始状态，不代表网络已学到变化的权重。本次训练轮次均未超过这个初始参照。", ""]
    if report.get("history"):
        lines += ["逐轮验证 RMSE：" + "；".join(f"第 {row['epoch']} 轮 {row['validation']['rmse']:.8f}" for row in report["history"]) + "。", ""]
    if "error" in report:
        lines.append(f"失败原因：{report['error']}")
    lines += [f"总耗时：{report.get('elapsed_seconds', 0):.2f} 秒。",
              "models/best.pt 保存网络、标准化统计和最佳轮次；counts.npz 保存原训练集的数量统计。",
              "predictions/*.csv.gz 保存两种融合评分及每条记录的 ALS 权重。"]
    (output / "adaptive_blend.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_adaptive_blend(cfg, output):
    validate_config(cfg)
    torch.set_num_threads(cfg.model.num_threads)
    output = Path(output).resolve()
    for source_path in (cfg.paths.raw_dir, cfg.paths.splits_dir, cfg.paths.blend_run):
        source_path = Path(source_path).expanduser().resolve()
        if output.is_relative_to(source_path) or source_path.is_relative_to(output):
            raise ValueError("输出与输入目录不可重叠。")
    if any((output / name).exists() for name in ("models", "predictions", "adaptive_blend.json")):
        raise FileExistsError("已有运行结果，不会覆盖。")
    started = time.perf_counter()
    root, source = check_input(cfg)
    runs, inputs = check_runs(cfg, output)
    for _, base_report in runs.values():
        if base_report["source"]["files"] != source["files"]:
            raise ValueError("基础模型与当前训练划分不一致。")
    blend_root = Path(cfg.paths.blend_run).expanduser().resolve()
    blend_path = blend_root / "blend.json"
    blend_report = json.loads(blend_path.read_text(encoding="utf-8"))
    if blend_report["status"] != "passed" or blend_report["input_reports"] != inputs:
        raise ValueError("固定融合报告未通过或基础模型不一致。")
    expected = {phase: sum(r["count"] for r in source["counts"] if r["split"] == phase) for phase in ("train", "validation", "test")}
    fixed_alpha = blend_report["weights"]["als"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir()
    report = {"status": "running", "generated_at": datetime.now(timezone.utc).isoformat(), "source": source,
              "configuration": OmegaConf.to_container(cfg, resolve=True), "input_reports": inputs + [fingerprint(blend_path)],
              "fixed_alpha": fixed_alpha, "metrics": []}
    write_report(report, output)
    try:
        spark = start_spark(cfg, "AdaptiveBlendTrainingInputs")
        try:
            train, users, movies = prepare_training(spark, root, expected["train"], runs, cfg, output)
            validation = load_phase(spark, blend_root, blend_report, "validation", (users, movies), expected["validation"], cfg)
        finally:
            spark.stop()
        best_path = output / "models/best.pt"
        model = fit_model(train, validation, cfg.model, fixed_alpha, best_path, report, lambda: write_report(report, output))
        del train
        validation_score = evaluate(model, validation, cfg.model, fixed_alpha)
        _, checkpoint = load_model(best_path)
        if not math.isclose(validation_score["rmse"], checkpoint["validation_rmse"], abs_tol=1e-12, rel_tol=0):
            raise ValueError("最佳权重读回误差不一致。")
        # 先保存候选方案与最佳轮次，测试集不会影响这个选择。
        report["selected_method"] = "adaptive" if validation_score["rmse"] < validation_score["fixed_rmse"] else "fixed"
        parameters = {"selected_method": report["selected_method"], "best_epoch": report["best_epoch"],
                      "fixed_alpha": fixed_alpha, "model": OmegaConf.to_container(cfg.model, resolve=True),
                      "inputs": {name: str(run[0]) for name, run in runs.items()}}
        (output / "models/parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
        predictions = output / "predictions"
        predictions.mkdir()
        validation_score = evaluate(model, validation, cfg.model, fixed_alpha, predictions / "validation.csv.gz")
        report["metrics"].append({"split": "validation", **validation_score})
        del validation
        write_report(report, output)
        log(f"Best epoch and method frozen: {report['best_epoch']}, {report['selected_method']}; reading local test")
        spark = start_spark(cfg, "AdaptiveBlendFinalEvaluation")
        try:
            test = load_phase(spark, blend_root, blend_report, "test", (users, movies), expected["test"], cfg)
            test_score = evaluate(model, test, cfg.model, fixed_alpha, predictions / "test.csv.gz")
            report["metrics"].append({"split": "test", **test_score})
            del test
            schema = "userId int, movieId int, timestamp long, rating double, als_prediction double, mf_prediction double, als_weight double, prediction double, fixed_prediction double"
            from pyspark.sql import functions as F
            for row in report["metrics"]:
                saved = spark.read.schema(schema).option("header", True).option("mode", "FAILFAST").csv(str(predictions / f"{row['split']}.csv.gz"))
                score = saved.agg(F.count("*").alias("rows"),
                    F.sum(F.pow(F.col("prediction")-F.col("rating"), 2)).alias("sse"),
                    F.sum(F.when(F.col("prediction").isNull() | F.isnan("prediction") | ~F.col("prediction").between(cfg.model.rating_min, cfg.model.rating_max) | ~F.col("als_weight").between(0., 1.), 1).otherwise(0)).alias("invalid")).first()
                if score.rows != row["rows"] or score.invalid or not math.isclose(score.sse, row["squared_error_sum"], abs_tol=1e-9, rel_tol=1e-10):
                    raise ValueError("预测保存后读回不一致。")
        finally:
            spark.stop()
        files = [best_path, output / "models/counts.npz", output / "models/parameters.json", *sorted(predictions.glob("*.csv.gz"))]
        report["output_files"] = [fingerprint(p) for p in files]
        report["status"] = "passed"
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        write_report(report, output)


@hydra.main(version_base="1.3", config_path="../configs", config_name="adaptive_blend")
def main(cfg):
    output = Path(HydraConfig.get().runtime.output_dir)
    report = run_adaptive_blend(cfg, output)
    log(f"ADAPTIVE_BLEND_STATUS={report['status']}; report={output / 'adaptive_blend.md'}")


if __name__ == "__main__":
    main()
