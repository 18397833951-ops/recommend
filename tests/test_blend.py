"""融合权重手算、记录对齐，以及验证选权重与本地测试的隔离。"""

import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import blend
from prepare_data import fingerprint, start_spark


def config(overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="blend", overrides=["spark.driver_memory=1g", "spark.shuffle_partitions=2",
                                                       "data.output_partitions=1", *overrides])


class WeightTest(unittest.TestCase):
    def test_exact_solution_endpoints_and_equal_predictions(self):
        for a, b, expected in ((8., -4., .5), (8., 2., 0.), (8., -12., 1.), (0., 0., 0.)):
            self.assertEqual(blend.choose_weight({"difference_squared_sum": a, "error_difference_sum": b}), expected)
        with self.assertRaises(ValueError):
            blend.choose_weight({"difference_squared_sum": float("nan"), "error_difference_sum": 0.})


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()
        cls.spark = start_spark(cls.cfg, "BlendFixture")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_rejects_missing_duplicate_mismatched_and_invalid_predictions(self):
        schema = "userId long, movieId long, als_timestamp long, als_rating double, als_prediction double, mf_timestamp long, mf_rating double, mf_prediction double"
        valid = (1, 10, 3, 2., 1., 3, 2., 3.)
        cases = [[valid, valid], [(1, 10, 3, 2., 1., None, None, None)],
                 [(1, 10, 3, 2., 1., 4, 2., 3.)], [(1, 10, 3, 2., 1., 3, 4., 3.)],
                 [(1, 10, 3, 2., float("nan"), 3, 2., 3.)]]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                blend.summarize(self.spark.createDataFrame(rows, schema), 1, self.cfg.model)

    def test_full_pipeline_freezes_validation_weight_before_test(self):
        with tempfile.TemporaryDirectory(prefix="blend-test-", dir=ROOT / ".runtime") as folder:
            case = Path(folder)
            cfg = config([f"paths.als_run={(case/'als').as_posix()}", f"paths.mf_run={(case/'mf').as_posix()}"])
            source = {"files": [{"path": str(case / "fixed_split.parquet"), "sha256": "fixture"}],
                      "counts": [{"split": "validation", "count": 2}, {"split": "test", "count": 2}]}
            scores = {"als": [], "mf": []}
            files = {"als": [], "mf": []}
            for phase, targets, als, mf in (("validation", [2., 4.], [1., 5.], [3., 3.]),
                                             ("test", [5., 1.], [1., 5.], [5., 1.])):
                rows = [(1, 10+i, targets[i], 3, als[i]) for i in range(2)]
                path = case / "als/predictions" / f"split={phase}"
                self.spark.createDataFrame(rows, "userId long, movieId long, rating double, timestamp long, prediction double").write.parquet(str(path))
                files["als"] += [fingerprint(p) for p in path.rglob("*.parquet")]
                path = case / "mf/predictions" / f"{phase}.csv.gz"
                path.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["userId", "movieId", "rating", "timestamp", "raw_prediction", "prediction", "user_fallback", "movie_fallback"])
                    for i in reversed(range(2)):
                        writer.writerow([1, 10+i, targets[i], 3, mf[i], mf[i], 0, 0])
                files["mf"].append(fingerprint(path))
                for name, predictions in (("als", als), ("mf", mf)):
                    scores[name].append({"split": phase, "rows": 2,
                                         "rmse": (sum((a-b)**2 for a,b in zip(predictions, targets))/2)**.5})
            for name, filename in (("als", "als.json"), ("mf", "biased_mf.json")):
                (case/name/filename).write_text(json.dumps({"status": "passed", "source": source,
                    "output_files": files[name], "metrics": scores[name]}), encoding="utf-8")
            events = []
            original_load, original_choose = blend.load_predictions, blend.choose_weight
            def load(spark, runs, phase, model):
                events.append(phase)
                if phase == "test":
                    weights = json.loads((case/"result/models/parameters.json").read_text(encoding="utf-8"))["weights"]
                    self.assertEqual(weights, {"als": .5, "mf": .5})
                return original_load(spark, runs, phase, model)
            def choose(stats):
                events.append("choose")
                return original_choose(stats)
            # 当前会话由测试类统一关闭；生产代码仍正常关闭自己的 Spark。
            with patch.object(blend, "load_predictions", side_effect=load), patch.object(blend, "choose_weight", side_effect=choose), patch.object(self.spark, "stop"):
                report = blend.run_blend(cfg, case/"result")
            self.assertEqual(events, ["validation", "choose", "test"])
            self.assertEqual(report["status"], "passed")
            fused = [r for r in report["metrics"] if r["model"] == "blend"]
            self.assertEqual([r["rmse"] for r in fused], [0., 2.])
            self.assertTrue(all(fingerprint(Path(f["path"])) == f for f in report["output_files"]))
            with self.assertRaises(FileExistsError):
                blend.run_blend(cfg, case/"result")
            # 更换划分的模型不能组合。
            path = case/"mf/biased_mf.json"
            altered = json.loads(path.read_text(encoding="utf-8"))
            altered["source"]["files"][0]["sha256"] = "other-split"
            path.write_text(json.dumps(altered), encoding="utf-8")
            with self.assertRaises(ValueError):
                blend.check_runs(cfg, case/"another-result")


if __name__ == "__main__":
    unittest.main()
