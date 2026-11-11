"""检查教师评分数据，整理 Parquet，按用户时间划分训练、验证和本地测试。"""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import hydra
from hydra.core.hydra_config import HydraConfig

HEADERS = {
    "ratings.csv": ["userId", "movieId", "rating", "timestamp"],
    "test_pairs.csv": ["userId", "movieId", "timestamp"],
    "movies.csv": ["movieId", "title", "genres"],
    "movie_mapping.csv": ["combinedMovieId", "source", "sourceMovieId"],
    "links.csv": ["movieId", "imdbId", "tmdbId"],
    "tags.csv": ["userId", "movieId", "tag", "timestamp"],
}


def log(message):
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def fingerprint(path):
    with path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": checksum}


def start_spark(cfg, app_name="TeacherDataPreparation"):
    from pyspark.sql import SparkSession
    temp = Path(cfg.paths.temp_dir).expanduser().resolve()
    temp.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = os.environ["TEMP"] = str(temp)
    os.environ["PYSPARK_PYTHON"] = os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    if os.name == "nt":
        home = os.environ.get("HADOOP_HOME")
        if not home or not all((Path(home) / "bin" / name).is_file() for name in ("hadoop.dll", "winutils.exe")):
            raise RuntimeError("请配置 HADOOP_HOME，bin 目录需含 hadoop.dll 和 winutils.exe。")
        os.environ["PATH"] = str(Path(home) / "bin") + os.pathsep + os.environ["PATH"]
    spark = (SparkSession.builder.master(cfg.spark.master).appName(app_name)
             .config("spark.driver.memory", cfg.spark.driver_memory).config("spark.ui.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false").config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.shuffle.partitions", cfg.spark.shuffle_partitions)
             .config("spark.io.compression.codec", cfg.spark.io_compression_codec)
             .config("spark.sql.parquet.compression.codec", cfg.spark.parquet_compression_codec).getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def split_ratings(ratings, train_percent, validation_percent=None):
    from pyspark.sql import Window, functions as F
    groups = ratings.groupBy("userId", "timestamp").count()
    users = Window.partitionBy("userId")
    timeline = users.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    groups = (groups.withColumn("total", F.sum("count").over(users))
              .withColumn("cumulative", F.sum("count").over(timeline)))
    validation_end = train_percent + validation_percent if validation_percent is not None else 100
    boundaries = groups.groupBy("userId").agg(
        F.min(F.struct(F.abs(F.col("cumulative") * 100 - F.col("total") * train_percent).alias("distance"),
                       F.col("timestamp"))).alias("train_boundary"),
        F.min(F.struct(F.abs(F.col("cumulative") * 100 - F.col("total") * validation_end).alias("distance"),
                       F.col("timestamp"))).alias("validation_boundary"))
    boundaries = boundaries.select("userId", F.col("train_boundary.timestamp").alias("train_cutoff"),
                                   F.col("validation_boundary.timestamp").alias("validation_cutoff"))
    return (ratings.join(boundaries, "userId")
            .withColumn("split", F.when(F.col("timestamp") <= F.col("train_cutoff"), "train")
                        .when(F.col("timestamp") <= F.col("validation_cutoff"), "validation").otherwise("test"))
            .drop("train_cutoff", "validation_cutoff"))


def assert_empty(frame, message):
    if frame.limit(1).count():
        raise ValueError(message)


def inspect_inputs(raw):
    for name, header in HEADERS.items():
        with (raw / name).open(encoding="utf-8-sig", newline="") as source:
            if next(csv.reader(source), None) != header:
                raise ValueError(f"{name} 的字段与教师数据格式不符。")
    summary = json.loads((raw / "summary.json").read_text(encoding="utf-8-sig"))
    ranges = json.loads((raw / "id_ranges.json").read_text(encoding="utf-8-sig"))
    for kind in ("users", "movies"):
        public, synthetic = (ranges[f"{name}_{kind}"] for name in ("movielens32m", "synthetic"))
        if not (1 <= public[0] <= public[1] < synthetic[0] <= synthetic[1]):
            raise ValueError(f"{kind} 的来源范围无效或重叠。")
    return summary, ranges


def source_column(ranges):
    from pyspark.sql import functions as F
    return F.when(F.col("userId").between(*ranges["movielens32m_users"]), "movielens32m").otherwise("synthetic")


def invalid_ids(ranges):
    from pyspark.sql import functions as F
    public = F.col("userId").between(*ranges["movielens32m_users"]) & F.col("movieId").between(*ranges["movielens32m_movies"])
    synthetic = F.col("userId").between(*ranges["synthetic_users"]) & F.col("movieId").between(*ranges["synthetic_movies"])
    return F.col("userId").isNull() | F.col("movieId").isNull() | ~(public | synthetic)


def prepare_and_audit(spark, raw, processed, summary, ranges):
    from pyspark.sql import functions as F
    ratings = (spark.read.option("header", True).option("mode", "FAILFAST").option("enforceSchema", False)
               .schema("userId long, movieId long, rating double, timestamp long").csv(str(raw / "ratings.csv"))
               .withColumn("source", source_column(ranges)))
    ratings.write.parquet(str(processed / "ratings"))
    ratings = spark.read.parquet(str(processed / "ratings"))
    bad = (invalid_ids(ranges) | F.col("timestamp").isNull() | (F.col("timestamp") < 0)
           | F.col("rating").isNull() | F.isnan("rating") | ~F.col("rating").between(0.5, 5.0)
           | (F.col("rating") * 2 != F.floor(F.col("rating") * 2)))
    assert_empty(ratings.where(bad), "训练评分有缺失、非法评分、时间戳或来源 ID。")
    assert_empty(ratings.groupBy("userId", "movieId").count().where("count > 1"), "训练中存在重复用户电影组合。")
    counts = {r.source: r["count"] for r in ratings.groupBy("source").count().collect()}
    if sum(counts.values()) != summary["training_ratings"]:
        raise ValueError("训练记录数与 summary.json 不符。")

    movies = (spark.read.option("header", True).option("mode", "FAILFAST").option("enforceSchema", False)
              .option("escape", '"').option("multiLine", True)
              .schema("movieId long, title string, genres string").csv(str(raw / "movies.csv")))
    movies.write.parquet(str(processed / "movies"))
    movies = spark.read.parquet(str(processed / "movies"))
    assert_empty(movies.where("movieId IS NULL OR title IS NULL"), "电影 ID 或名称缺失。")
    assert_empty(movies.groupBy("movieId").count().where("count > 1"), "电影 ID 重复。")
    if movies.count() != summary["combined_movies"]:
        raise ValueError("电影数量与 summary.json 不符。")
    known_movies = movies.select("movieId")
    assert_empty(ratings.select("movieId").distinct().join(known_movies, "movieId", "left_anti"), "评分引用不存在的电影。")

    mapping = (spark.read.option("header", True).option("mode", "FAILFAST").option("enforceSchema", False)
               .schema("combinedMovieId long, source string, sourceMovieId long").csv(str(raw / "movie_mapping.csv"))
               .withColumnRenamed("combinedMovieId", "movieId"))
    assert_empty(mapping.where(F.col("movieId").isNull() | F.col("sourceMovieId").isNull()
                 | ~F.col("source").isin("movielens32m", "synthetic") | F.col("source").isNull()), "电影映射存在非法值。")
    assert_empty(mapping.groupBy("movieId").count().where("count > 1"), "电影映射 ID 重复。")
    assert_empty(known_movies.join(mapping.select("movieId"), "movieId", "left_anti"), "电影映射缺少 ID。")
    assert_empty(mapping.select("movieId").join(known_movies, "movieId", "left_anti"), "电影映射包含未知 ID。")
    for origin in ("movielens32m", "synthetic"):
        assert_empty(mapping.where((F.col("source") == origin) & ~F.col("movieId").between(*ranges[f"{origin}_movies"])), "电影来源映射与 ID 范围不一致。")
    mapping.write.parquet(str(processed / "movie_mapping"))

    with (raw / "test_pairs.csv").open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        next(reader)
        pairs = [(index, *(int(value) for value in row)) for index, row in enumerate(reader)]
    if len(pairs) != summary["hidden_test_ratings"]:
        raise ValueError("待预测行数与 summary.json 不符。")
    test = spark.createDataFrame(pairs, "row_id long, userId long, movieId long, timestamp long")
    test = test.withColumn("source", source_column(ranges))
    assert_empty(test.where(invalid_ids(ranges) | (F.col("timestamp") < 0)), "待预测数据 ID 或时间戳非法。")
    assert_empty(test.groupBy("userId", "movieId").count().where("count > 1"), "待预测组合重复。")
    assert_empty(ratings.join(F.broadcast(test.select("userId", "movieId")), ["userId", "movieId"]), "待预测组合已存在于训练评分。")
    test.write.parquet(str(processed / "test_pairs"))
    return ratings, spark.read.parquet(str(processed / "test_pairs")), counts


def validate_split(spark, ratings, test, splits, expected_count):
    from pyspark.sql import functions as F
    saved = spark.read.parquet(str(splits / "ratings"))
    users = saved.groupBy("userId").agg(
        F.count("*").alias("total_rows"),
        F.sum(F.when(F.col("split") == "train", 1).otherwise(0)).alias("train_rows"),
        F.sum(F.when(F.col("split") == "validation", 1).otherwise(0)).alias("validation_rows"),
        F.sum(F.when(F.col("split") == "test", 1).otherwise(0)).alias("test_rows"),
        F.max(F.when(F.col("split") == "train", F.col("timestamp"))).alias("train_max_timestamp"),
        F.min(F.when(F.col("split") == "validation", F.col("timestamp"))).alias("validation_min_timestamp"),
        F.max(F.when(F.col("split") == "validation", F.col("timestamp"))).alias("validation_max_timestamp"),
        F.min(F.when(F.col("split") == "test", F.col("timestamp"))).alias("test_min_timestamp"))
    users.write.parquet(str(splits / "user_summary"))
    users = spark.read.parquet(str(splits / "user_summary"))
    assert_empty(users.where((F.col("train_rows") == 0)
                 | (F.col("train_max_timestamp") >= F.col("validation_min_timestamp"))
                 | (F.col("train_max_timestamp") >= F.col("test_min_timestamp"))
                 | (F.col("validation_max_timestamp") >= F.col("test_min_timestamp"))
                 | (F.col("train_rows") + F.col("validation_rows") + F.col("test_rows") != F.col("total_rows"))),
                 "划分缺少训练历史、包含未知划分或时间边界交叉。")
    original_users = ratings.groupBy("userId").count()
    assert_empty(users.join(original_users, "userId", "full").where(
        F.col("total_rows").isNull() | F.col("count").isNull() | (F.col("total_rows") != F.col("count"))), "划分前后每用户条数不同。")
    phases = [r.asDict() for r in saved.groupBy("source", "split").count().orderBy("source", "split").collect()]
    if sum(r["count"] for r in phases) != expected_count:
        raise ValueError("划分前后总条数不同。")
    train, validation = saved.where("split='train'"), saved.where("split='validation'")
    cold_movies = validation.join(train.select("movieId").distinct(), "movieId", "left_anti")
    local_test_cold = saved.where("split='test'").join(train.select("movieId").distinct(), "movieId", "left_anti")
    warm = test.join(users.select("userId", "total_rows", "train_max_timestamp"), "userId", "left")
    assert_empty(warm.where("total_rows IS NULL"), "正式测试中存在完整训练集未见用户。")
    assert_empty(test.join(ratings.select("movieId").distinct(), "movieId", "left_anti"), "正式测试中存在完整训练集未见电影。")
    # 正式测试只做数据边界检查，不用于本地验证划分和参数选择。
    latest = ratings.groupBy("userId").agg(F.max("timestamp").alias("latest"))
    chronology = test.join(latest, "userId")
    assert_empty(chronology.where("timestamp < latest"), "正式待预测时间早于该用户训练历史。")
    return {"by_source_and_split": phases, "users": users.count(),
            "users_without_validation": users.where("validation_rows=0").count(),
            "users_without_local_test": users.where("test_rows=0").count(),
            "validation_cold_movie_rows": cold_movies.count(),
            "validation_cold_movies": cold_movies.select("movieId").distinct().count(),
            "local_test_cold_movie_rows": local_test_cold.count(),
            "local_test_cold_movies": local_test_cold.select("movieId").distinct().count(),
            "test_rows": test.count(), "test_same_second_as_last_training": chronology.where("timestamp=latest").count()}


def write_report(report, output):
    (output / "data_preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 教师评分数据检查与划分", "", f"状态：{report['status']}。", "",
             f"目标训练比例为每用户 {report['train_percent']}%，验证比例 {report.get('validation_percent', 100-report['train_percent'])}%，剩余为本地测试。按完整时间组选择最接近累计目标的两条边界，距离相同时取较早边界。",
             "至少保留首组训练记录，单时间组用户全部进入训练。验证与本地测试的新电影保留并统计，真实评分原值保留。",
             "验证集用于选择算法和参数，本地测试集供方案冻结后评估。", "",
             "检查覆盖评分、待预测清单、电影和映射的内容与引用，以及来源范围、重复组合和划分数量、时间边界。",
             "tags.csv、links.csv 仅核对字段并记录指纹，内容整理按后续模型需要实施。", ""]
    if "summary" in report:
        s = report["summary"]
        total_rows = sum(row["count"] for row in s["by_source_and_split"])
        for phase, label in (("train", "训练"), ("validation", "验证"), ("test", "本地测试")):
            count = sum(row["count"] for row in s["by_source_and_split"] if row["split"] == phase)
            lines.append(f"{label} {count:,} 条（{count / total_rows:.4%}）。")
        lines.append("")
        lines += ["| 来源 | 划分 | 评分条数 |", "| --- | --- | ---: |"]
        for row in s["by_source_and_split"]:
            lines.append(f"| {row['source']} | {row['split']} | {row['count']:,} |")
        lines += ["", f"用户数 {s['users']:,}，其中无验证记录 {s['users_without_validation']:,}。",
                  f"验证冷电影 {s['validation_cold_movies']:,} 部，对应 {s['validation_cold_movie_rows']:,} 条评分。",
                  f"无本地测试记录用户 {s.get('users_without_local_test', 0):,} 位，本地测试冷电影 {s.get('local_test_cold_movies', 0):,} 部、{s.get('local_test_cold_movie_rows', 0):,} 条评分。",
                  f"正式待预测共 {s['test_rows']:,} 条，其中 {s['test_same_second_as_last_training']:,} 条与该用户最后训练评分同秒。"]
    lines += ["", f"处理结果在 `{report['paths']['processed']}`，本地划分在 `{report['paths']['splits']}`。",
              "processed/ratings 是完整教师评分。默认使用选定的本地训练模型提交，完整数据重训作为可选实验。processed/test_pairs 保留 row_id 恢复正式提交行顺序。",
              "splits/ratings 按 split=train、split=validation、split=test 分目录存储，user_summary 记录逐用户数量与边界。",
              f"耗时 {report.get('elapsed_seconds', 0):.2f} 秒。配置与文件 SHA256 见 JSON 和 .hydra。"]
    if "error" in report:
        lines.append(f"失败原因：{report['error']}")
    (output / "data_preparation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def checked_processed(processed, inputs, reports_dir):
    """重划时核对已通过检查的完整 Parquet 与原始输入指纹。"""
    reports = sorted(Path(reports_dir).expanduser().resolve().rglob("data_preparation.json"),
                     key=lambda path: path.stat().st_mtime_ns, reverse=True)
    actual_inputs = {item["path"]: item["sha256"] for item in inputs}
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report["status"] != "passed" or Path(report["paths"]["processed"]).resolve() != processed
                or {item["path"]: item["sha256"] for item in report["input_files"]} != actual_inputs):
            continue
        expected = {Path(item["path"]): item["sha256"] for item in report["output_files"]
                    if Path(item["path"]).is_relative_to(processed)}
        if not expected or set(processed.rglob("*.parquet")) != set(expected):
            raise ValueError("已整理文件清单与检查报告不符。")
        for file, checksum in expected.items():
            if fingerprint(file)["sha256"] != checksum:
                raise ValueError(f"已整理文件发生变化：{file}")
        return {"path": str(path), "training_by_source": report["training_by_source"]}
    raise ValueError("未找到匹配原始输入和整理目录的通过报告，请先完成数据检查。")


def run_preparation(cfg, output):
    if type(cfg.data.train_percent) is not int or not 1 <= cfg.data.train_percent <= 99:
        raise ValueError("train_percent 必须是 1～99 的整数。")
    if (type(cfg.data.validation_percent) is not int or cfg.data.validation_percent <= 0
            or cfg.data.train_percent + cfg.data.validation_percent > 100):
        raise ValueError("验证比例必须为正整数，训练与验证比例之和不能超过 100。")
    raw, processed, splits = (Path(cfg.paths[name]).expanduser().resolve() for name in ("raw_dir", "processed_dir", "splits_dir"))
    output = Path(output).resolve()
    paths = [raw, processed, splits, output]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for index, a in enumerate(paths) for b in paths[index+1:]):
        raise ValueError("输入、处理结果、划分和报告目录不可重叠。")
    if (processed.exists() and not cfg.data.reuse_processed) or splits.exists() or (output / "data_preparation.json").exists():
        raise FileExistsError("输出已存在，请使用新的输出目录。")
    started = time.perf_counter()
    summary, ranges = inspect_inputs(raw)
    log("Recording teacher input fingerprints")
    inputs = [fingerprint(raw / name) for name in (*HEADERS, "summary.json", "id_ranges.json")]
    reused = checked_processed(processed, inputs, cfg.paths.preparation_reports_dir) if cfg.data.reuse_processed else None
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "generated_at": datetime.now(timezone.utc).isoformat(),
              "train_percent": cfg.data.train_percent, "validation_percent": cfg.data.validation_percent,
              "paths": {"raw": str(raw), "processed": str(processed), "splits": str(splits)},
              "input_files": inputs, "configuration": {"data": dict(cfg.data), "spark": dict(cfg.spark)}, "dataset": summary["dataset"]}
    spark = start_spark(cfg)
    report["spark_version"] = spark.version
    try:
        write_report(report, output)
        if reused:
            log("Reusing verified teacher Parquet files")
            ratings = spark.read.parquet(str(processed / "ratings"))
            test = spark.read.parquet(str(processed / "test_pairs"))
            counts = reused["training_by_source"]
            report["prepared_source"] = reused["path"]
        else:
            log("Preparing typed teacher data and auditing contents")
            ratings, test, counts = prepare_and_audit(spark, raw, processed, summary, ranges)
        report["training_by_source"] = counts
        log("Splitting complete per-user timestamp groups")
        split = split_ratings(ratings, cfg.data.train_percent, cfg.data.validation_percent)
        (split.repartition(cfg.spark.shuffle_partitions, "split", "userId").write
         .partitionBy("split").parquet(str(splits / "ratings")))
        log("Reading back and verifying saved train/validation/local-test data")
        report["summary"] = validate_split(spark, ratings, test, splits, summary["training_ratings"])
        report["output_files"] = [fingerprint(path) for root in (processed, splits) for path in sorted(root.rglob("*.parquet"))]
        report["status"] = "passed"
        return report
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        write_report(report, output)
        spark.stop()


@hydra.main(version_base="1.3", config_path="../configs", config_name="data_prepare")
def main(cfg):
    report = run_preparation(cfg, Path(HydraConfig.get().runtime.output_dir))
    log(f"DATA_PREPARATION_STATUS={report['status']}")


if __name__ == "__main__":
    main()
