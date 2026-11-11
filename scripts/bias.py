"""训练正则化用户、电影偏置，用验证集选优后评估本地测试集。"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from baselines import check_input
from prepare_data import fingerprint, log, start_spark


def update_biases(train, users, global_mean, regularization):
    """固定用户偏置求电影偏置，再固定电影偏置求用户偏置。"""
    from pyspark.sql import functions as F

    if users is None:
        residual = train.withColumn("residual", F.col("rating") - global_mean)
    else:
        residual = (train.join(F.broadcast(users), "userId")
                    .withColumn("residual", F.col("rating") - global_mean - F.col("user_bias")))
    # 迭代时必须截断逻辑计划，普通 persist 会继续携带之前各轮的计划。
    # 本地检查点由 Spark 缓存系统管理，结束 Spark 会话时释放。
    movies = (residual.groupBy("movieId")
              .agg((F.sum("residual") / (F.count("*") + regularization)).alias("movie_bias"))
              .localCheckpoint(eager=True))
    new_users = (train.join(F.broadcast(movies), "movieId").groupBy("userId")
                 .agg((F.sum(F.col("rating") - global_mean - F.col("movie_bias"))
                       / (F.count("*") + regularization)).alias("user_bias"))
                 .localCheckpoint(eager=True))
    return new_users, movies


def predict_bias(pairs, users, movies, global_mean, model):
    """未知 ID 的偏置取零；预测不读取 rating，支持无标签的待预测表。"""
    from pyspark.sql import functions as F

    result = (pairs.join(F.broadcast(users), "userId", "left")
              .join(F.broadcast(movies), "movieId", "left")
              .withColumn("user_fallback", F.col("user_bias").isNull())
              .withColumn("movie_fallback", F.col("movie_bias").isNull()))
    prediction = F.lit(global_mean) + F.coalesce(F.col("user_bias"), F.lit(0.0)) + F.coalesce(F.col("movie_bias"), F.lit(0.0))
    if model.clip_predictions:
        prediction = F.greatest(F.lit(model.rating_min), F.least(F.lit(model.rating_max), prediction))
    return result.withColumn("prediction", prediction).drop("user_bias", "movie_bias")


def evaluate(predictions):
    from pyspark.sql import functions as F

    invalid = F.lit(False)
    for name in ("rating", "prediction"):
        invalid = invalid | F.col(name).isNull() | F.isnan(name) | (F.abs(F.col(name)) == float("inf"))
    row = predictions.agg(F.count("*").alias("rows"),
                          F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid"),
                          F.sum(F.pow(F.col("prediction") - F.col("rating"), 2)).alias("squared_error_sum"),
                          F.sum(F.col("user_fallback").cast("long")).alias("user_fallback_rows"),
                          F.sum(F.col("movie_fallback").cast("long")).alias("movie_fallback_rows"),
                          F.sum((F.col("user_fallback") | F.col("movie_fallback")).cast("long")).alias("fallback_rows")).first().asDict()
    if not row["rows"] or row.pop("invalid"):
        raise ValueError("评测为空，或评分、预测存在缺失、NaN、无穷值。")
    if not math.isfinite(row["squared_error_sum"]):
        raise ValueError("平方误差不是有限数值。")
    row["rmse"] = math.sqrt(row["squared_error_sum"] / row["rows"])
    return row


def validate_model_config(model):
    values = list(model.regularizations)
    if (not values or len(set(values)) != len(values)
            or any(not math.isfinite(value) or value <= 0 for value in values)):
        raise ValueError("regularizations 必须为不重复的有限正数列表。")
    if isinstance(model.max_epochs, bool) or not isinstance(model.max_epochs, int) or model.max_epochs < 1:
        raise ValueError("max_epochs 必须为正整数。")
    if not all(math.isfinite(value) for value in (model.rating_min, model.rating_max)) or model.rating_min >= model.rating_max:
        raise ValueError("评分范围无效。")


def fit_bias(train, validation, model, output):
    """最小化训练 SSE + λ(用户偏置平方和 + 电影偏置平方和)，μ 固定。"""
    from pyspark.sql import functions as F

    validate_model_config(model)
    training = train.agg(F.avg("rating").alias("global_mean"), F.count("*").alias("rows")).first().asDict()
    if not training["rows"] or not math.isfinite(training["global_mean"]):
        raise ValueError("训练评分为空或均值无效。")
    output.mkdir(parents=True, exist_ok=True)
    trials, best = [], None
    for regularization in model.regularizations:
        users = movies = None
        for epoch in range(1, model.max_epochs + 1):
            started = time.perf_counter()
            users, movies = update_biases(train, users, training["global_mean"], regularization)
            score = evaluate(predict_bias(validation, users, movies, training["global_mean"], model))
            trial = {"regularization": float(regularization), "epoch": epoch,
                     **score, "elapsed_seconds": round(time.perf_counter() - started, 2)}
            trials.append(trial)
            log(f"regularization={regularization:g}; epoch={epoch}; validation_RMSE={score['rmse']:.8f}")
            # 只根据验证误差选优。相同误差时保留先出现的方案。
            if best is None or trial["rmse"] < best["rmse"]:
                best = dict(trial)
                users.write.mode("overwrite").parquet(str(output / "users"))
                movies.write.mode("overwrite").parquet(str(output / "movies"))
    parameters = {"training": training, "selected": best, "model": OmegaConf.to_container(model, resolve=True)}
    (output / "parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
    return training, best, trials


def write_report(report, output):
    (output / "bias.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 正则化偏置基线", "", f"状态：{report['status']}。", "",
             "预测 = 训练全局均值 + 用户偏置 + 电影偏置。未知 ID 的对应偏置为零，另一项已知偏置继续使用。",
             "每轮先更新电影偏置，再更新用户偏置。偏置仅从训练集拟合，验证集整体 RMSE 选择正则强度和轮次。",
             "本地测试集在模型选择完成后评估。回退记录仍全部计入误差，预测保留连续精度。", ""]
    if report.get("selected"):
        selected = report["selected"]
        lines += [f"选定正则强度：{selected['regularization']:g}；轮次：{selected['epoch']}。",
                  f"训练评分：{report['training']['rows']:,}；全局均值：{report['training']['global_mean']:.10f}。", ""]
    model = report["configuration"]["model"]
    lines += [f"预测范围处理：{'截断至 ' + str(model['rating_min']) + '～' + str(model['rating_max']) if model['clip_predictions'] else '不截断'}。", "",
              "| 划分 | 记录数 | RMSE | 用户偏置缺失 | 电影偏置缺失 | 回退记录 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report.get("metrics", []):
        label = {"validation": "验证集", "test": "本地测试集"}[row["split"]]
        lines.append(f"| {label} | {row['rows']:,} | {row['rmse']:.8f} | {row['user_fallback_rows']:,} | {row['movie_fallback_rows']:,} | {row['fallback_rows']:,} |")
    lines += ["", "models/ 保存选定模型的用户、电影偏置与 parameters.json；predictions/ 保存验证和本地测试的评分、预测、回退标记。",
              "bias.json 的 trials 记录所有方案的验证误差和耗时，metrics 记录选定模型读回后的整体误差，output_files 记录输出 SHA256。",
              f"总耗时：{report.get('elapsed_seconds', 0):.2f} 秒。"]
    if "error" in report:
        lines.append(f"失败原因：{report['error']}")
    (output / "bias.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bias(cfg, output):
    from pyspark import StorageLevel
    from pyspark.sql import functions as F

    validate_model_config(cfg.model)
    if list(cfg.evaluation.splits) != ["validation", "test"]:
        raise ValueError("偏置选优需要 evaluation.splits=[validation,test]。")
    output = Path(output).resolve()
    for source in (cfg.paths.raw_dir, cfg.paths.splits_dir):
        source = Path(source).expanduser().resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("运行结果与输入目录不可重叠。")
    if any((output / name).exists() for name in ("models", "predictions", "bias.json")):
        raise FileExistsError(f"结果已存在，不会覆盖：{output}")
    started = time.perf_counter()
    log("Checking split file fingerprints")
    ratings_path, source = check_input(cfg)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "generated_at": datetime.now(timezone.utc).isoformat(), "source": source,
              "configuration": {"model": OmegaConf.to_container(cfg.model, resolve=True), "spark": dict(cfg.spark)}}
    spark = start_spark(cfg, "RatingBiasBaseline")
    report["spark_version"] = spark.version
    train = validation = None
    try:
        write_report(report, output)
        ratings = spark.read.parquet(str(ratings_path))
        # 大表在磁盘缓存，避免挤占 1 GB Java 堆；偏置小表由 Spark 管理缓存。
        train = ratings.where("split='train'").select("userId", "movieId", "rating").persist(StorageLevel.DISK_ONLY)
        validation = ratings.where("split='validation'").persist(StorageLevel.DISK_ONLY)
        report["training"], report["selected"], report["trials"] = fit_bias(train, validation, cfg.model, output / "models")
        expected = {phase: sum(row["count"] for row in source["counts"] if row["split"] == phase)
                    for phase in ("train", "validation", "test")}
        if report["training"]["rows"] != expected["train"]:
            raise ValueError("训练记录数与划分报告不一致。")
        train.unpersist()
        validation.unpersist()
        users = spark.read.parquet(str(output / "models/users"))
        movies = spark.read.parquet(str(output / "models/movies"))
        log("Model selected; saving validation and local-test predictions")
        predictions = predict_bias(ratings.where(F.col("split").isin("validation", "test")), users, movies,
                                   report["training"]["global_mean"], cfg.model)
        (predictions.withColumn("model_version", F.lit(cfg.model.version))
         .repartition(cfg.spark.shuffle_partitions, "split", "userId")
         .write.partitionBy("split").parquet(str(output / "predictions")))
        saved = spark.read.parquet(str(output / "predictions"))
        report["metrics"] = []
        for phase in ("validation", "test"):
            score = evaluate(saved.where(F.col("split") == phase))
            if score["rows"] != expected[phase]:
                raise ValueError("评测数量与固定划分不一致。")
            report["metrics"].append({"split": phase, **score})
        if not math.isclose(report["metrics"][0]["rmse"], report["selected"]["rmse"], abs_tol=1e-10, rel_tol=0):
            raise ValueError("选定模型读回后验证 RMSE 不一致。")
        files = sorted(output.rglob("*.parquet")) + [output / "models/parameters.json"]
        report["output_files"] = [fingerprint(path) for path in files]
        report["status"] = "passed"
        return report
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        try:
            write_report(report, output)
        finally:
            spark.stop()


@hydra.main(version_base="1.3", config_path="../configs", config_name="bias")
def main(cfg):
    output = Path(HydraConfig.get().runtime.output_dir)
    report = run_bias(cfg, output)
    log(f"BIAS_STATUS={report['status']}; report={output / 'bias.md'}")


if __name__ == "__main__":
    main()
