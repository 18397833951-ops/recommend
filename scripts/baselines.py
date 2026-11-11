"""仅从训练评分拟合三种均值，计算验证与本地测试的整体 RMSE。"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import hydra
from hydra.core.hydra_config import HydraConfig

MODELS = ("global_mean", "user_mean", "movie_mean")


def check_input(cfg):
    from prepare_data import fingerprint

    root = Path(cfg.paths.splits_dir).expanduser().resolve() / "ratings"
    reports = list(Path(cfg.paths.preparation_reports_dir).expanduser().resolve().rglob("data_preparation.json"))
    if not reports:
        raise ValueError("请先运行 prepare_data.py。")
    path = max(reports, key=lambda item: item.stat().st_mtime_ns)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "passed" or Path(report["paths"]["splits"]).resolve() / "ratings" != root:
        raise ValueError("最新数据报告未通过，或划分目录与当前配置不一致。")
    expected = {Path(item["path"]): item["sha256"] for item in report["output_files"]
                if Path(item["path"]).is_relative_to(root)}
    if not expected or set(root.rglob("*.parquet")) != set(expected):
        raise ValueError("划分文件清单与准备报告不一致。")
    for file, checksum in expected.items():
        if fingerprint(file)["sha256"] != checksum:
            raise ValueError(f"划分文件已变化：{file}")
    counts = report["summary"]["by_source_and_split"]
    for phase in ("train", *cfg.evaluation.splits):
        if not sum(row["count"] for row in counts if row["split"] == phase):
            raise ValueError(f"缺少非空的 {phase} 划分。")
    return root, {"preparation_report": str(path), "counts": counts,
                  "files": [{"path": str(file), "sha256": checksum} for file, checksum in sorted(expected.items())]}


def fit_means(train, output):
    from pyspark.sql import functions as F

    statistics = train.agg(F.avg("rating").alias("global_mean"), F.count("*").alias("rows")).first().asDict()
    if not statistics["rows"] or not math.isfinite(statistics["global_mean"]):
        raise ValueError("训练评分为空或均值无效。")
    output.mkdir(parents=True, exist_ok=True)
    (output / "global_mean.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    for key, folder in (("userId", "users"), ("movieId", "movies")):
        (train.groupBy(key).agg(F.avg("rating").alias("mean_rating"), F.count("*").alias("rating_count"))
         .write.parquet(str(output / folder)))
    return statistics


def predict_means(evaluation, users, movies, global_mean):
    from pyspark.sql import functions as F

    users = users.select("userId", F.col("mean_rating").alias("user_value"))
    movies = movies.select("movieId", F.col("mean_rating").alias("movie_value"))
    return (evaluation.select("split", "source", "userId", "movieId", "rating", "timestamp")
            .join(users, "userId", "left").join(movies, "movieId", "left")
            .withColumn("global_mean_prediction", F.lit(global_mean))
            .withColumn("user_mean_fallback", F.col("user_value").isNull())
            .withColumn("movie_mean_fallback", F.col("movie_value").isNull())
            .withColumn("user_mean_prediction", F.coalesce(F.col("user_value"), F.lit(global_mean)))
            .withColumn("movie_mean_prediction", F.coalesce(F.col("movie_value"), F.lit(global_mean)))
            .drop("user_value", "movie_value"))


def evaluate_predictions(predictions):
    """按划分汇总全部记录的平方误差和行数，计算整体 RMSE。"""
    from prepare_data import assert_empty
    from pyspark.sql import functions as F

    invalid = F.lit(False)
    for name in ("rating", *(f"{model}_prediction" for model in MODELS)):
        invalid = invalid | F.col(name).isNull() | F.isnan(name) | (F.abs(F.col(name)) == float("inf"))
    assert_empty(predictions.where(invalid), "评分或预测包含缺失、NaN 或无穷值。")
    aggregations = [F.count("*").alias("rows")]
    for model in MODELS:
        aggregations.append(F.sum(F.pow(F.col(f"{model}_prediction") - F.col("rating"), 2)).alias(f"{model}_sse"))
        if model != "global_mean":
            aggregations.append(F.sum(F.col(f"{model}_fallback").cast("long")).alias(f"{model}_fallback_rows"))
    grouped = [row.asDict() for row in predictions.groupBy("split").agg(*aggregations).collect()]
    if not grouped:
        raise ValueError("评测记录为空。")
    metrics = []
    for row in sorted(grouped, key=lambda row: row["split"] == "test"):
        for model in MODELS:
            error = row[f"{model}_sse"]
            metrics.append({"split": row["split"], "model": model, "rows": row["rows"],
                            "squared_error_sum": error, "rmse": math.sqrt(error / row["rows"]),
                            "fallback_rows": row.get(f"{model}_fallback_rows", 0)})
    return metrics


def write_report(report, output):
    (output / "baselines.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 评分均值基线", "", f"状态：{report['status']}。生成时间：{report['generated_at']}。", "",
             "全局、用户、电影均值全部从 split=train 拟合。用户或电影缺少训练统计时，回退到训练全局均值。",
             "预测保留连续值，真实评分仅在计算误差时读取。整体 RMSE 按全部记录的平方误差求和除以记录数后开平方。",
             "这是三个固定规则基线的验证和本地测试成绩，后续模型仍依据验证集选择轮数与参数，再到同一本地测试集比较。", ""]
    if "training" in report:
        lines += [f"训练记录 {report['training']['rows']:,} 条，全局平均评分 {report['training']['global_mean']:.8f}。", ""]
    model_labels = {"global_mean": "全局均值", "user_mean": "用户均值", "movie_mean": "电影均值"}
    split_labels = {"validation": "验证集", "test": "本地测试集"}
    if report.get("metrics"):
        for model in MODELS:
            lines += [f"## {model_labels[model]}", "",
                      "| 划分 | 记录数 | RMSE | 回退记录 |", "| --- | ---: | ---: | ---: |"]
            for row in report["metrics"]:
                if row["model"] == model and row.get("source", "all") == "all":
                    lines.append(f"| {split_labels[row['split']]} | {row['rows']:,} | {row['rmse']:.8f} | {row['fallback_rows']:,} |")
            lines.append("")
    lines += ["", "models/global_mean.json 保存全局均值和训练记录数，models/users、models/movies 保存用户与电影均值及评分数。",
              "predictions 按 split 分目录保存原评分、三种预测和回退标记。配置在 .hydra，输入输出 SHA256 和平方误差在 JSON。",
              f"总耗时 {report.get('elapsed_seconds', 0):.2f} 秒。"]
    if "error" in report:
        lines.append(f"失败原因：{report['error']}")
    (output / "baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_baselines(cfg, output):
    from prepare_data import fingerprint, log, start_spark
    from pyspark.sql import functions as F

    phases = list(cfg.evaluation.splits)
    if not phases or len(set(phases)) != len(phases) or any(phase not in ("validation", "test") for phase in phases):
        raise ValueError("evaluation.splits 应为 validation、test 中不重复的非空列表。")
    output = Path(output).resolve()
    splits = Path(cfg.paths.splits_dir).expanduser().resolve()
    raw = Path(cfg.paths.raw_dir).expanduser().resolve()
    for source in (splits, raw):
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("运行结果与输入目录不可重叠。")
    if any((output / name).exists() for name in ("models", "predictions", "baselines.json")):
        raise FileExistsError(f"结果已存在，不会覆盖：{output}")
    started = time.perf_counter()
    log("Checking split file fingerprints")
    ratings_path, source = check_input(cfg)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "generated_at": datetime.now(timezone.utc).isoformat(),
              "model_version": cfg.model.version, "evaluation": {"splits": phases}, "source": source,
              "configuration": {"spark": dict(cfg.spark)}, "stage_seconds": {}}
    spark = start_spark(cfg, app_name="RatingMeanBaselines")
    report["spark_version"] = spark.version
    try:
        write_report(report, output)
        ratings = spark.read.parquet(str(ratings_path))
        stage = time.perf_counter()
        log("Fitting global, user and movie means from train only")
        report["training"] = fit_means(ratings.where("split='train'"), output / "models")
        expected_train = sum(row["count"] for row in source["counts"] if row["split"] == "train")
        if report["training"]["rows"] != expected_train:
            raise ValueError("训练记录数与划分报告不一致。")
        report["stage_seconds"]["fit"] = time.perf_counter() - stage
        stage = time.perf_counter()
        log("Saving validation and local-test predictions")
        predictions = predict_means(ratings.where(F.col("split").isin(*phases)),
                                    spark.read.parquet(str(output / "models" / "users")),
                                    spark.read.parquet(str(output / "models" / "movies")), report["training"]["global_mean"])
        (predictions.withColumn("model_version", F.lit(cfg.model.version))
         .withColumn("generated_at", F.lit(report["generated_at"]))
         .repartition(cfg.spark.shuffle_partitions, "split", "userId")
         .write.partitionBy("split").parquet(str(output / "predictions")))
        report["stage_seconds"]["predict"] = time.perf_counter() - stage
        stage = time.perf_counter()
        log("Computing overall RMSE for each split")
        report["metrics"] = evaluate_predictions(spark.read.parquet(str(output / "predictions")))
        expected = {phase: sum(row["count"] for row in source["counts"] if row["split"] == phase) for phase in phases}
        for model in MODELS:
            observed = {row["split"]: row["rows"] for row in report["metrics"] if row["model"] == model}
            if observed != expected:
                raise ValueError("评测记录数与固定划分不一致。")
        report["stage_seconds"]["evaluate"] = time.perf_counter() - stage
        files = sorted(output.rglob("*.parquet")) + [output / "models" / "global_mean.json"]
        report["output_files"] = [fingerprint(path) for path in files]
        report["status"] = "passed"
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        try:
            write_report(report, output)
        finally:
            spark.stop()


@hydra.main(version_base="1.3", config_path="../configs", config_name="baselines")
def main(cfg):
    from prepare_data import log
    output = Path(HydraConfig.get().runtime.output_dir)
    report = run_baselines(cfg, output)
    log(f"BASELINES_STATUS={report['status']}; report={output / 'baselines.md'}")


if __name__ == "__main__":
    main()
