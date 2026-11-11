"""按 YAML 中的一组超参数训练显式 ALS，评估验证与本地测试评分。"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from baselines import check_input
from bias import evaluate
from prepare_data import fingerprint, log, start_spark


def validate_config(cfg):
    model = cfg.model
    for name in ("rank", "max_iter", "num_user_blocks", "num_item_blocks", "block_size"):
        if isinstance(model[name], bool) or not isinstance(model[name], int) or model[name] < 1:
            raise ValueError(f"{name} 必须为正整数。")
    if (isinstance(model.reg_param, bool) or not isinstance(model.reg_param, (int, float))
            or not math.isfinite(model.reg_param) or model.reg_param <= 0):
        raise ValueError("reg_param 必须为有限正数。")
    if not all(math.isfinite(v) for v in (model.rating_min, model.rating_max)) or model.rating_min >= model.rating_max:
        raise ValueError("评分范围无效。")
    if list(cfg.evaluation.splits) != ["validation", "test"]:
        raise ValueError("ALS 评测需要 evaluation.splits=[validation,test]。")


def make_estimator(model):
    from pyspark.ml.recommendation import ALS

    return ALS(userCol="userId", itemCol="movieId", ratingCol="rating",
               implicitPrefs=False, nonnegative=False, coldStartStrategy="nan", seed=model.seed,
               rank=model.rank, regParam=model.reg_param, maxIter=model.max_iter,
               numUserBlocks=model.num_user_blocks, numItemBlocks=model.num_item_blocks,
               blockSize=model.block_size, intermediateStorageLevel="MEMORY_AND_DISK",
               finalStorageLevel="MEMORY_AND_DISK").setPredictionCol("als_prediction")


def predict_als(pairs, fitted, user_means, global_mean, model):
    """已知组合采用向量点积；缺少特征时使用用户均值，再回退全局均值。"""
    from pyspark.sql import functions as F

    known_movies = fitted.itemFactors.select(F.col("id").alias("movieId"), F.lit(True).alias("known_movie"))
    result = (fitted.transform(pairs).join(F.broadcast(user_means), "userId", "left")
              .join(F.broadcast(known_movies), "movieId", "left")
              .withColumn("user_fallback", F.col("mean_rating").isNull())
              .withColumn("movie_fallback", F.col("known_movie").isNull()))
    cold = F.col("user_fallback") | F.col("movie_fallback")
    prediction = F.when(cold, F.coalesce(F.col("mean_rating"), F.lit(global_mean))).otherwise(F.col("als_prediction").cast("double"))
    # NaN/无穷值必须保留给评测报错，不能在截断时被掩盖成合法边界。
    invalid = prediction.isNull() | F.isnan(prediction) | (F.abs(prediction) == float("inf"))
    if model.clip_predictions:
        prediction = F.when(invalid, prediction).otherwise(F.greatest(F.lit(model.rating_min), F.least(F.lit(model.rating_max), prediction)))
    return result.withColumn("prediction", prediction).drop("mean_rating", "known_movie")


def fit_als(train, validation, user_means, global_mean, model, output, report):
    """一次运行只拟合配置中的一个模型，不进行参数搜索。"""
    started = time.perf_counter()
    log(f"Fitting ALS: rank={model.rank}; regParam={model.reg_param:g}; maxIter={model.max_iter}")
    fitted = make_estimator(model).fit(train)
    try:
        score = evaluate(predict_als(validation, fitted, user_means, global_mean, model))
        report["fit"] = {"rank": model.rank, "reg_param": float(model.reg_param), "max_iter": model.max_iter,
                         **score, "elapsed_seconds": round(time.perf_counter() - started, 2)}
        log(f"validation_RMSE={score['rmse']:.8f}; seconds={report['fit']['elapsed_seconds']}")
        fitted.write().save(str(output / "als"))
    finally:
        fitted.userFactors.unpersist()
        fitted.itemFactors.unpersist()
    parameters = {"training": report["training"], "fit": report["fit"],
                  "model": OmegaConf.to_container(model, resolve=True), "fallback": "user_mean_then_global_mean"}
    (output / "parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(report, output):
    (output / "als.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 显式 ALS 评分预测", "", f"状态：{report['status']}。", "",
             "训练仅使用 split=train 的真实评分，implicitPrefs=false；预测为用户与电影特征向量的点积。",
             "一次运行只按配置的维数、正则强度和轮次训练一个模型，不进行参数搜索。验证成绩用于比较不同配置。",
             "训练未见的用户或电影保留：优先使用训练用户均值，用户统计缺失时使用训练全局均值。",
             "回退记录按用户或电影任一特征缺失去重计数。本地测试成绩用于评估固定配置，不用于调整超参数。", ""]
    if report.get("fit"):
        fitted = report["fit"]
        lines += [f"训练维数：{fitted['rank']}；正则强度：{fitted['reg_param']:g}；轮次：{fitted['max_iter']}。",
                  f"训练记录：{report['training']['rows']:,}；全局均值：{report['training']['global_mean']:.10f}。", ""]
    model = report["configuration"]["model"]
    lines += [f"预测范围处理：{'截断到 ' + str(model['rating_min']) + '～' + str(model['rating_max']) if model['clip_predictions'] else '不截断'}，保留连续精度。", "",
              "| 划分 | 记录数 | RMSE | 用户特征缺失 | 电影特征缺失 | 回退记录 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in report.get("metrics", []):
        label = {"validation": "验证集", "test": "本地测试集"}[row["split"]]
        lines.append(f"| {label} | {row['rows']:,} | {row['rmse']:.8f} | {row['user_fallback_rows']:,} | {row['movie_fallback_rows']:,} | {row['fallback_rows']:,} |")
    lines += ["", "models/als 保存本次 Spark ALSModel，userFactors、itemFactors 可读取每个 ID 的特征向量。",
              "models/user_means 保存回退统计，models/parameters.json 保存全局均值与本次配置。",
              "predictions 按 split 保存原评分、ALS 原始预测、最终预测及回退标记。冷启动的 als_prediction 为 NaN，最终 prediction 必须有限。",
              "als.json 的 fit 保存本次配置的验证成绩和训练耗时，metrics 保存模型读回后的评测，另记录输入输出校验值及配置。",
              f"总耗时：{report.get('elapsed_seconds', 0):.2f} 秒。"]
    if "error" in report:
        lines.append(f"失败原因：{report['error']}")
    (output / "als.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_als(cfg, output):
    from pyspark import StorageLevel
    from pyspark.ml.recommendation import ALSModel
    from pyspark.sql import functions as F

    validate_config(cfg)
    output = Path(output).resolve()
    for source in (cfg.paths.raw_dir, cfg.paths.splits_dir):
        source = Path(source).expanduser().resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("运行结果与输入目录不可重叠。")
    if any((output / name).exists() for name in ("models", "predictions", "als.json")):
        raise FileExistsError(f"结果已存在，不会覆盖：{output}")
    started = time.perf_counter()
    log("Checking split file fingerprints")
    ratings_path, source = check_input(cfg)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "generated_at": datetime.now(timezone.utc).isoformat(), "source": source,
              "configuration": {"model": OmegaConf.to_container(cfg.model, resolve=True), "spark": dict(cfg.spark)}}
    spark = start_spark(cfg, "ExplicitRatingALS")
    report["spark_version"] = spark.version
    try:
        write_report(report, output)
        ratings = spark.read.parquet(str(ratings_path))
        train = ratings.where("split='train'").select("userId", "movieId", "rating").persist(StorageLevel.DISK_ONLY)
        validation = ratings.where("split='validation'").persist(StorageLevel.DISK_ONLY)
        report["training"] = train.agg(F.avg("rating").alias("global_mean"), F.count("*").alias("rows")).first().asDict()
        expected = {phase: sum(row["count"] for row in source["counts"] if row["split"] == phase)
                    for phase in ("train", "validation", "test")}
        if report["training"]["rows"] != expected["train"] or not math.isfinite(report["training"]["global_mean"]):
            raise ValueError("训练数量或均值无效。")
        models = output / "models"
        (train.groupBy("userId").agg(F.avg("rating").alias("mean_rating"))
         .write.parquet(str(models / "user_means")))
        user_means = spark.read.parquet(str(models / "user_means")).cache()
        fit_als(train, validation, user_means, report["training"]["global_mean"], cfg.model, models, report)
        train.unpersist()
        validation.unpersist()
        fitted = ALSModel.load(str(models / "als"))
        if fitted.rank != report["fit"]["rank"]:
            raise ValueError("保存模型的维数不一致。")
        log("Model trained; saving validation and local-test predictions")
        predictions = predict_als(ratings.where(F.col("split").isin("validation", "test")), fitted, user_means,
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
        if not math.isclose(report["metrics"][0]["rmse"], report["fit"]["rmse"], abs_tol=1e-9, rel_tol=0):
            raise ValueError("模型读回后的验证 RMSE 不一致。")
        files = [path for path in models.rglob("*") if path.is_file() and not path.name.startswith((".", "_"))]
        files += list((output / "predictions").rglob("*.parquet"))
        report["output_files"] = [fingerprint(path) for path in sorted(files)]
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="als")
def main(cfg):
    output = Path(HydraConfig.get().runtime.output_dir)
    report = run_als(cfg, output)
    log(f"ALS_STATUS={report['status']}; report={output / 'als.md'}")


if __name__ == "__main__":
    main()
