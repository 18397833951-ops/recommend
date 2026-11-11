"""评分版 ItemCF：训练期相似电影，按评分差值加权预测验证和本地测试评分。"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import OmegaConf

from baselines import check_input, fit_means
from prepare_data import fingerprint, log, start_spark


def validate_config(cfg):
    for name in ("neighbors", "min_neighbors", "min_item_ratings"):
        value = cfg.model[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须为正整数。")
    if cfg.model.min_neighbors > cfg.model.neighbors:
        raise ValueError("min_neighbors 不能大于 neighbors。")
    if not isinstance(cfg.data.prediction_partitions, int) or cfg.data.prediction_partitions < 1:
        raise ValueError("prediction_partitions 必须为正整数。")
    if not math.isfinite(cfg.model.sampling_threshold) or not 0 <= cfg.model.sampling_threshold <= 1:
        raise ValueError("sampling_threshold 必须在 0～1 之间。")
    if not math.isfinite(cfg.model.min_similarity) or not 0 <= cfg.model.min_similarity < 1:
        raise ValueError("min_similarity 必须在 0～1 之间，不含 1。")
    if not all(math.isfinite(v) for v in (cfg.model.rating_min, cfg.model.rating_max)) or cfg.model.rating_min >= cfg.model.rating_max:
        raise ValueError("评分范围无效。")
    if list(cfg.evaluation.splits) != ["validation", "test"]:
        raise ValueError("需要 evaluation.splits=[validation,test]。")


def fit_neighbors(spark, train, users, movies, cfg, output):
    from pyspark import StorageLevel
    from pyspark.mllib.linalg import Vectors, VectorUDT
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.sql import Window, functions as F
    from pyspark.sql.types import StructType, StructField

    eligible = movies.where(F.col("rating_count") >= cfg.model.min_item_ratings).select("movieId")
    centered = (train.join(eligible, "movieId")
                .join(users.select("userId", F.col("mean_rating").alias("user_mean")), "userId")
                .withColumn("value", F.col("rating") - F.col("user_mean")).where("value != 0"))
    groups = (centered.groupBy("userId")
              .agg(F.sort_array(F.collect_list(F.struct("movieId", "value"))).alias("entries"))
              .repartition(cfg.spark.shuffle_partitions, "userId").sortWithinPartitions("userId"))
    dimension = int(movies.agg(F.max("movieId")).first()[0]) + 1

    def vector(row):
        return (Vectors.sparse(dimension, [(int(entry.movieId), float(entry.value)) for entry in row.entries]),)

    # 在 JVM 内缓存原生向量，RowMatrix 的多次扫描不必反复转换 Python 对象。
    schema = StructType([StructField("vector", VectorUDT(), False)])
    vectors = spark.createDataFrame(groups.rdd.map(vector), schema).persist(StorageLevel.DISK_ONLY)
    edges = None
    try:
        vector_rows = vectors.count()
        summary = {"vector_rows": vector_rows, "vector_dimension": dimension,
                   "eligible_movies": eligible.count(), "sampling_threshold": cfg.model.sampling_threshold}
        if not vector_rows:
            empty = spark.createDataFrame([], "movieId int, neighbor_ids array<int>, similarities array<double>")
            empty.write.parquet(str(output))
            return {**summary, "candidate_edges": 0, "directed_neighbors": 0, "movies_with_neighbors": 0}
        log("Computing movie cosine similarities")
        matrix = RowMatrix(vectors, numRows=vector_rows, numCols=dimension)
        entries = matrix.columnSimilarities(float(cfg.model.sampling_threshold)).entries
        pairs = entries.map(lambda entry: (int(entry.i), int(entry.j), float(entry.value)))
        edges = (spark.createDataFrame(pairs, "i int, j int, similarity double")
                 .where(F.col("similarity").isNotNull() & ~F.isnan("similarity")
                        & (F.col("similarity") > cfg.model.min_similarity) & (F.col("similarity") != float("inf")))
                 .withColumn("similarity", F.least(F.lit(1.0), F.col("similarity")))
                 .persist(StorageLevel.DISK_ONLY))
        summary["candidate_edges"] = edges.count()
        log(f"Positive candidate movie pairs: {summary['candidate_edges']:,}; keeping top {cfg.model.neighbors}")
        forward = edges.select(F.col("i").alias("movieId"), F.col("j").alias("neighbor"), "similarity")
        reverse = edges.select(F.col("j").alias("movieId"), F.col("i").alias("neighbor"), "similarity")
        order = Window.partitionBy("movieId").orderBy(F.desc("similarity"), F.asc("neighbor"))
        selected = (forward.unionByName(reverse).withColumn("position", F.row_number().over(order))
                    .where(F.col("position") <= cfg.model.neighbors))
        graph = (selected.groupBy("movieId")
                 .agg(F.sort_array(F.collect_list(F.struct("position", "neighbor", "similarity"))).alias("entries"))
                 .select("movieId", F.transform("entries", lambda entry: entry["neighbor"]).alias("neighbor_ids"),
                         F.transform("entries", lambda entry: entry["similarity"]).alias("similarities")))
        graph.write.parquet(str(output))
        saved = spark.read.parquet(str(output))
        counts = saved.agg(F.count("*").alias("movies_with_neighbors"),
                           F.coalesce(F.sum(F.size("neighbor_ids")), F.lit(0)).alias("directed_neighbors")).first().asDict()
        return {**summary, **counts}
    finally:
        vectors.unpersist()
        if edges is not None:
            edges.unpersist()


def load_catalog(movies, neighbors):
    from pyspark.sql import functions as F

    catalog = {}
    joined = movies.join(neighbors, "movieId", "left")
    for row in joined.select("movieId", "mean_rating", "neighbor_ids", "similarities").toLocalIterator():
        catalog[row.movieId] = (float(row.mean_rating), np.asarray(row.neighbor_ids or [], dtype=np.int32),
                                np.asarray(row.similarities or [], dtype=np.float64))
    return catalog


def predict_pairs(spark, pairs, history, catalog, global_mean, cfg):
    from pyspark.sql import functions as F

    targets = pairs.groupBy("userId").agg(F.collect_list(F.struct("movieId", "rating", "timestamp", "source")).alias("targets"))
    joined = targets.join(history, "userId", "left")
    broadcast = spark.sparkContext.broadcast(catalog)
    model = OmegaConf.to_container(cfg.model, resolve=True)
    spark.sparkContext.addPyFile(str(Path(__file__).with_name("itemcf_scoring.py").resolve()))

    def score_partition(rows):
        from itemcf_scoring import predict_user
        graph = broadcast.value
        for row in rows:
            values = None if row.history is None else [(entry.movieId, entry.residual) for entry in row.history]
            targets = [(entry.movieId, entry.rating, entry.timestamp, entry.source) for entry in row.targets]
            yield from predict_user(row.userId, values, targets, row.user_mean, graph, global_mean, model)

    schema = "userId int, movieId int, rating double, timestamp long, source string, raw_prediction double, prediction double, actual_neighbors int, user_fallback boolean, movie_fallback boolean, neighbor_fallback boolean"
    rows = joined.rdd.coalesce(cfg.data.prediction_partitions)
    return spark.createDataFrame(rows.mapPartitions(score_partition), schema), broadcast


def evaluate(predictions):
    from pyspark.sql import functions as F

    invalid = F.lit(False)
    for name in ("rating", "prediction", "raw_prediction"):
        invalid = invalid | F.col(name).isNull() | F.isnan(name) | (F.abs(F.col(name)) == float("inf"))
    result = predictions.agg(F.count("*").alias("rows"), F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid"),
                             F.sum(F.pow(F.col("prediction") - F.col("rating"), 2)).alias("squared_error_sum"),
                             F.sum(F.col("user_fallback").cast("long")).alias("user_fallback_rows"),
                             F.sum(F.col("movie_fallback").cast("long")).alias("movie_fallback_rows"),
                             F.sum(F.col("neighbor_fallback").cast("long")).alias("fallback_rows"),
                             F.avg("actual_neighbors").alias("mean_actual_neighbors")).first().asDict()
    if not result["rows"] or result.pop("invalid"):
        raise ValueError("评测为空，或存在缺失、NaN、无穷值。")
    result["rmse"] = math.sqrt(result["squared_error_sum"] / result["rows"])
    return result


def write_report(report, output):
    (output / "itemcf.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    model = report["configuration"]["model"]
    lines = ["# 评分版 ItemCF", "", f"状态：{report['status']}。", "",
             "相似度只使用训练评分：先减去用户训练均值，构造电影的用户评分向量，计算余弦相似度。",
             f"sampling_threshold={model['sampling_threshold']}：{'精确计算' if model['sampling_threshold'] == 0 else 'Spark DIMSUM 采样近似计算，相似度不是精确值'}。",
             f"电影至少有 {model['min_item_ratings']} 条训练评分才参与相似度计算；每部电影保留最多 {model['neighbors']} 个正相似邻居。",
             "预测 = 目标电影训练均值 + sum(相似度 × (用户对邻居的训练评分 − 邻居电影训练均值)) / sum(相似度)。",
             f"匹配邻居少于 {model['min_neighbors']} 时回退目标电影训练均值。目标电影未见时使用用户训练均值；两者均未见时使用全局均值。",
             "回退记录包含邻居不足的记录，全部评测评分保留。相似度、均值、用户历史均只来自训练集，一次只运行一组配置。", ""]
    if "training" in report:
        lines += [f"训练记录：{report['training']['rows']:,}；全局均值：{report['training']['global_mean']:.10f}。", ""]
    if "graph" in report:
        lines += [f"保留有向邻居：{report['graph']['directed_neighbors']:,}；有邻居的电影：{report['graph']['movies_with_neighbors']:,}。", ""]
    lines += ["| 划分 | 记录数 | RMSE | 用户未见 | 电影未见 | 邻居不足回退 | 平均匹配邻居数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report.get("metrics", []):
        label = {"validation": "验证集", "test": "本地测试集"}[row["split"]]
        lines.append(f"| {label} | {row['rows']:,} | {row['rmse']:.8f} | {row['user_fallback_rows']:,} | {row['movie_fallback_rows']:,} | {row['fallback_rows']:,} | {row['mean_actual_neighbors']:.2f} |")
    lines += ["", "models/users、models/movies 保存训练均值与评分数，models/neighbors 保存邻居 ID 和相似度数组。",
              "models/parameters.json 和 global_mean.json 保存模型配置及全局统计。预测时复用 data/splits 中的训练历史，不复制评分数据。",
              "predictions 按 split 保存原评分、时间戳、原始/最终预测、实际邻居数与回退标记。",
              f"总耗时：{report.get('elapsed_seconds', 0):.2f} 秒。"]
    if "error" in report:
        lines.append(f"失败原因：{report['error']}")
    (output / "itemcf.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_itemcf(cfg, output):
    from pyspark import StorageLevel
    from pyspark.sql import functions as F

    validate_config(cfg)
    output = Path(output).resolve()
    for source in (cfg.paths.raw_dir, cfg.paths.splits_dir):
        source = Path(source).expanduser().resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("输出与输入目录不可重叠。")
    if any((output / name).exists() for name in ("models", "predictions", "itemcf.json")):
        raise FileExistsError(f"结果已存在，不会覆盖：{output}")
    started = time.perf_counter()
    root, source = check_input(cfg)
    expected = {phase: sum(row["count"] for row in source["counts"] if row["split"] == phase)
                for phase in ("train", "validation", "test")}
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "generated_at": datetime.now(timezone.utc).isoformat(), "source": source,
              "configuration": OmegaConf.to_container(cfg, resolve=True)}
    spark = start_spark(cfg, "RatingItemCF")
    report["spark_version"] = spark.version
    try:
        write_report(report, output)
        ratings = spark.read.parquet(str(root))
        train = ratings.where("split='train'").select("userId", "movieId", "rating").persist(StorageLevel.DISK_ONLY)
        models = output / "models"
        report["training"] = fit_means(train, models)
        if report["training"]["rows"] != expected["train"]:
            raise ValueError("训练数量不一致。")
        users = spark.read.parquet(str(models / "users"))
        movies = spark.read.parquet(str(models / "movies"))
        report["graph"] = fit_neighbors(spark, train, users, movies, cfg, models / "neighbors")
        write_report(report, output)
        neighbors = spark.read.parquet(str(models / "neighbors"))
        catalog = load_catalog(movies, neighbors)
        history = (train.join(movies.select("movieId", F.col("mean_rating").alias("movie_mean")), "movieId")
                   .groupBy("userId").agg(F.avg("rating").alias("user_mean"),
                         F.collect_list(F.struct("movieId", (F.col("rating") - F.col("movie_mean")).alias("residual"))).alias("history"))
                   .persist(StorageLevel.DISK_ONLY))
        report["metrics"] = []
        for phase in ("validation", "test"):
            log(f"Predicting {phase} ratings from training histories")
            predictions, broadcast = predict_pairs(spark, ratings.where(F.col("split") == phase), history, catalog,
                                                   report["training"]["global_mean"], cfg)
            try:
                path = output / "predictions" / f"split={phase}"
                predictions.withColumn("model_version", F.lit(cfg.model.version)).write.parquet(str(path))
            finally:
                broadcast.destroy()
            saved = spark.read.parquet(str(path))
            metric = evaluate(saved)
            if metric["rows"] != expected[phase]:
                raise ValueError("评测数量与固定划分不一致。")
            report["metrics"].append({"split": phase, **metric})
            log(f"{phase}_RMSE={metric['rmse']:.8f}; fallback_rows={metric['fallback_rows']:,}")
            write_report(report, output)
        history.unpersist()
        train.unpersist()
        parameters = {"training": report["training"], "model": OmegaConf.to_container(cfg.model, resolve=True),
                      "graph": report["graph"], "train_history_path": str(root)}
        (models / "parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
        files = sorted(output.rglob("*.parquet")) + [models / "global_mean.json", models / "parameters.json"]
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="itemcf")
def main(cfg):
    output = Path(HydraConfig.get().runtime.output_dir)
    report = run_itemcf(cfg, output)
    log(f"ITEMCF_STATUS={report['status']}; report={output / 'itemcf.md'}")


if __name__ == "__main__":
    main()
