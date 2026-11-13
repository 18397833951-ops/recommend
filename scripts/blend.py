"""验证集选择 ALS 与带偏置矩阵分解的凸组合权重，再评估本地测试。"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from prepare_data import fingerprint, log, start_spark


def choose_weight(moments):
    """最小化 sum((mf-y) + alpha*(als-mf))²，alpha 限制在 [0,1]。"""
    a, b = moments["difference_squared_sum"], moments["error_difference_sum"]
    if not math.isfinite(a) or not math.isfinite(b) or a < 0:
        raise ValueError("融合统计无效。")
    # 两份预测完全相同时保留 MF。
    return min(1.0, max(0.0, -b / a)) if a else 0.0


def check_runs(cfg, output):
    reports, inputs = {}, []
    if not math.isfinite(cfg.model.rating_min) or not math.isfinite(cfg.model.rating_max) or cfg.model.rating_min >= cfg.model.rating_max:
        raise ValueError("评分范围无效。")
    for name, filename in (("als", "als.json"), ("mf", "biased_mf.json")):
        root = Path(cfg.paths[f"{name}_run"]).expanduser().resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("融合输出目录与模型输入目录不可重叠。")
        path = root / filename
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise ValueError(f"{name} 运行报告尚未通过。")
        reports[name] = (root, report)
        inputs.append(fingerprint(path))
    # 比较完整划分的指纹和条数，拒绝混用不同划分训练的模型。
    left, right = (reports[name][1]["source"] for name in ("als", "mf"))
    manifest = lambda source: {str(Path(f["path"]).resolve()): f["sha256"] for f in source["files"]}
    if not manifest(left) or manifest(left) != manifest(right) or left["counts"] != right["counts"]:
        raise ValueError("两个模型的数据划分不一致。")
    return reports, inputs


def load_predictions(spark, runs, phase, model):
    """按阶段读取并校验文件；测试文件只在权重保存后读取。"""
    from pyspark.sql import functions as F

    frames = []
    for name in ("als", "mf"):
        root, report = runs[name]
        path = root / "predictions" / (f"split={phase}" if name == "als" else f"{phase}.csv.gz")
        files = sorted(path.rglob("*.parquet")) if name == "als" else [path]
        expected = {Path(item["path"]).resolve(): item for item in report["output_files"]}
        recorded = {p for p in expected if p == path or p.is_relative_to(path)}
        if not files or set(files) != recorded:
            raise ValueError(f"{name}/{phase} 预测清单不一致。")
        for file in files:
            if fingerprint(file) != expected[file]:
                raise ValueError(f"预测文件已变化：{file}")
        if name == "als":
            frame = spark.read.parquet(str(path))
        else:
            schema = "userId int, movieId int, rating double, timestamp long, raw_prediction double, prediction double, user_fallback int, movie_fallback int"
            frame = spark.read.schema(schema).option("header", True).option("mode", "FAILFAST").csv(str(path))
        frames.append(frame.select(F.col("userId").cast("long"), F.col("movieId").cast("long"),
                                  F.col("timestamp").cast("long").alias(f"{name}_timestamp"),
                                  F.col("rating").cast("double").alias(f"{name}_rating"),
                                  F.col("prediction").cast("double").alias(f"{name}_prediction")))
    return frames[0].join(frames[1], ["userId", "movieId"], "full")


def summarize(frame, expected_rows, model):
    from pyspark.sql import functions as F

    invalid = F.col("userId").isNull() | F.col("movieId").isNull()
    for name in ("als_timestamp", "mf_timestamp", "als_rating", "mf_rating", "als_prediction", "mf_prediction"):
        invalid = invalid | F.col(name).isNull() | F.isnan(name) | (F.abs(F.col(name)) == float("inf"))
    invalid = invalid | (F.col("als_timestamp") != F.col("mf_timestamp")) | (F.col("als_rating") != F.col("mf_rating"))
    for name in ("als_prediction", "mf_prediction"):
        invalid = invalid | ~F.col(name).between(model.rating_min, model.rating_max)
    difference = F.col("als_prediction") - F.col("mf_prediction")
    error = F.col("mf_prediction") - F.col("mf_rating")
    result = frame.agg(F.count("*").alias("rows"),
                       F.countDistinct(F.struct("userId", "movieId")).alias("distinct_pairs"),
                       F.sum(F.when(invalid, 1).otherwise(0)).alias("invalid_rows"),
                       F.sum(difference * difference).alias("difference_squared_sum"),
                       F.sum(error * difference).alias("error_difference_sum"),
                       F.sum(error * error).alias("mf_sse"),
                       F.sum(F.pow(F.col("als_prediction") - F.col("als_rating"), 2)).alias("als_sse")).first().asDict()
    if result["rows"] != expected_rows or result["distinct_pairs"] != expected_rows or result["invalid_rows"]:
        raise ValueError("预测缺失、重复、数量错误、标签不匹配或出现非法数值。")
    return result


def blended_predictions(frame, alpha):
    from pyspark.sql import functions as F

    return frame.select("userId", "movieId", F.col("mf_timestamp").alias("timestamp"),
                        F.col("mf_rating").alias("rating"), "als_prediction", "mf_prediction",
                        (F.col("mf_prediction") + F.lit(alpha) * (F.col("als_prediction") - F.col("mf_prediction"))).alias("prediction"))


def evaluate_blend(frame):
    from pyspark.sql import functions as F

    result = frame.agg(F.count("*").alias("rows"),
                       F.sum(F.pow(F.col("prediction") - F.col("rating"), 2)).alias("squared_error_sum")).first().asDict()
    if not result["rows"] or result["squared_error_sum"] is None or not math.isfinite(result["squared_error_sum"]):
        raise ValueError("融合评测为空或包含非法数值。")
    result["rmse"] = math.sqrt(result["squared_error_sum"] / result["rows"])
    return result


def write_report(report, output):
    (output / "blend.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# ALS 与带偏置矩阵分解的加权融合", "", f"状态：{report['status']}。", "",
             "复用两个模型的最终预测（含回退和评分截断），不重新训练。",
             "预测 = alpha × ALS + (1-alpha) × 带偏置矩阵分解；权重范围为 [0,1]。",
             "仅用验证集求平方误差最小的权重，保存权重后才读取本地测试；本地测试不参与选权重。", ""]
    if "weights" in report:
        lines += [f"ALS 权重：{report['weights']['als']:.10f}；矩阵分解权重：{report['weights']['mf']:.10f}。", ""]
    for model, title in (("als", "显式 ALS"), ("mf", "带偏置矩阵分解"), ("blend", "加权融合")):
        lines += [f"## {title}", "", "| 划分 | 记录数 | RMSE |", "| --- | ---: | ---: |"]
        for row in report.get("metrics", []):
            if row["model"] == model:
                label = "验证集" if row["split"] == "validation" else "本地测试集"
                lines.append(f"| {label} | {row['rows']:,} | {row['rmse']:.8f} |")
        lines.append("")
    lines += ["models/parameters.json 保存冻结的权重和输入运行目录，predictions 按 split 保存逐行分数。",
              f"总耗时：{report.get('elapsed_seconds', 0):.2f} 秒。"]
    if "error" in report:
        lines.append(f"失败原因：{report['error']}")
    (output / "blend.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_blend(cfg, output):
    from pyspark import StorageLevel

    if isinstance(cfg.data.output_partitions, bool) or not isinstance(cfg.data.output_partitions, int) or cfg.data.output_partitions < 1:
        raise ValueError("output_partitions 必须为正整数。")
    output = Path(output).resolve()
    if any((output / name).exists() for name in ("models", "predictions", "blend.json")):
        raise FileExistsError(f"结果已存在，不会覆盖：{output}")
    runs, inputs = check_runs(cfg, output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"status": "running", "generated_at": datetime.now(timezone.utc).isoformat(),
              "configuration": OmegaConf.to_container(cfg, resolve=True), "input_reports": inputs, "metrics": []}
    spark = start_spark(cfg, "ValidationWeightedBlend")
    try:
        for phase in ("validation", "test"):
            log(f"Checking and aligning {phase} predictions")
            frame = load_predictions(spark, runs, phase, cfg.model).persist(StorageLevel.DISK_ONLY)
            try:
                expected = [next(row for row in runs[name][1]["metrics"] if row["split"] == phase) for name in ("als", "mf")]
                if expected[0]["rows"] != expected[1]["rows"]:
                    raise ValueError("两个模型的评测条数不一致。")
                moments = summarize(frame, expected[0]["rows"], cfg.model)
                for name, original in zip(("als", "mf"), expected):
                    rmse = math.sqrt(moments[f"{name}_sse"] / moments["rows"])
                    if not math.isclose(rmse, original["rmse"], abs_tol=1e-9, rel_tol=0):
                        raise ValueError(f"{name}/{phase} 读回误差与原报告不一致。")
                    report["metrics"].append({"model": name, "split": phase, "rows": moments["rows"], "rmse": rmse})
                if phase == "validation":
                    alpha = choose_weight(moments)
                    report["weights"] = {"als": alpha, "mf": 1 - alpha}
                    report["validation_moments"] = moments
                    parameters = {"weights": report["weights"], "selection_split": "validation",
                                  "inputs": {name: str(run[0]) for name, run in runs.items()},
                                  "model": OmegaConf.to_container(cfg.model, resolve=True)}
                    (output / "models").mkdir()
                    (output / "models/parameters.json").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
                    log(f"Weights frozen: ALS={alpha:.10f}, MF={1-alpha:.10f}")
                predictions = blended_predictions(frame, alpha)
                path = output / "predictions" / f"split={phase}"
                predictions.coalesce(cfg.data.output_partitions).write.parquet(str(path))
                score = evaluate_blend(spark.read.parquet(str(path)))
                if score["rows"] != moments["rows"]:
                    raise ValueError("融合预测保存后数量不一致。")
                expected_sse = moments["mf_sse"] + 2 * alpha * moments["error_difference_sum"] + alpha * alpha * moments["difference_squared_sum"]
                if not math.isclose(score["squared_error_sum"], expected_sse, rel_tol=1e-10, abs_tol=1e-9):
                    raise ValueError("融合预测读回误差不一致。")
                report["metrics"].append({"model": "blend", "split": phase, **score})
                log(f"{phase} blend RMSE={score['rmse']:.8f}")
                write_report(report, output)
            finally:
                frame.unpersist()
        report["output_files"] = [fingerprint(path) for path in sorted((output / "predictions").rglob("*.parquet"))]
        report["output_files"].append(fingerprint(output / "models/parameters.json"))
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="blend")
def main(cfg):
    output = Path(HydraConfig.get().runtime.output_dir)
    report = run_blend(cfg, output)
    log(f"BLEND_STATUS={report['status']}; report={output / 'blend.md'}")


if __name__ == "__main__":
    main()
