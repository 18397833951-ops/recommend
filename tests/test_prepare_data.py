"""核对整时间组划分、比例边界、冷电影保留与正式清单行顺序。"""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_data


class PreparationTest(unittest.TestCase):
    def test_saved_split_ties_short_histories_ratings_and_test_order(self):
        with tempfile.TemporaryDirectory(prefix="prepare-test-", dir=ROOT / ".runtime") as folder:
            case = Path(folder)
            raw = case / "raw"
            raw.mkdir()
            rows = [(1, m, ((m % 10) + 1) / 2, m) for m in range(1, 21)]
            rows += [(2, m, 3.5, 1 if m <= 8 else 2) for m in range(1, 11)]
            rows += [(3, m, 4.0, 10) for m in range(21, 25)]
            rows += [(4, 28, 1.5, 1), (4, 29, 2.5, 2), (5, 30, 4.5, 1)]
            tables = {
                "ratings.csv": rows,
                "movies.csv": [(m, '电影,"带引号"', "Drama") for m in range(1, 31)],
                "movie_mapping.csv": [(m, "movielens32m" if m <= 27 else "synthetic", 1000+m) for m in range(1, 31)],
                "test_pairs.csv": [(4, 30, 1000), (1, 21, 1000)],
                "links.csv": [], "tags.csv": [],
            }
            for name, data in tables.items():
                with (raw / name).open("w", encoding="utf-8", newline="") as target:
                    writer = csv.writer(target)
                    writer.writerow(prepare_data.HEADERS[name])
                    writer.writerows(data)
            (raw / "summary.json").write_text(json.dumps({"dataset": "test", "training_ratings": 37,
                                                         "hidden_test_ratings": 2, "combined_movies": 30}), encoding="utf-8")
            ranges = {"movielens32m_users": [1,3], "synthetic_users": [4,5],
                      "movielens32m_movies": [1,27], "synthetic_movies": [28,30]}
            (raw / "id_ranges.json").write_text(json.dumps(ranges), encoding="utf-8")
            with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
                cfg = compose(config_name="data_prepare", overrides=[
                    f"paths.raw_dir={raw.as_posix()}", f"paths.processed_dir={(case/'processed').as_posix()}",
                    f"paths.splits_dir={(case/'splits').as_posix()}", "spark.shuffle_partitions=3"])
            result = prepare_data.run_preparation(cfg, case / "report")
            self.assertEqual(result["status"], "passed")
            counts = result["summary"]["by_source_and_split"]
            self.assertEqual(sum(row["count"] for row in counts if row["split"] == "train"), 31)
            self.assertEqual(sum(row["count"] for row in counts if row["split"] == "validation"), 2)
            self.assertEqual(sum(row["count"] for row in counts if row["split"] == "test"), 4)
            self.assertEqual(result["summary"]["users_without_validation"], 4)
            self.assertEqual(result["summary"]["users_without_local_test"], 3)
            self.assertEqual(result["summary"]["validation_cold_movie_rows"], 2)
            spark = prepare_data.start_spark(cfg)
            try:
                saved = spark.read.parquet(str(case / "splits" / "ratings"))
                users = spark.read.parquet(str(case / "splits" / "user_summary"))
                self.assertEqual({r.userId: (r.train_rows,r.validation_rows,r.test_rows) for r in users.collect()},
                                 {1:(16,2,2),2:(8,0,2),3:(4,0,0),4:(2,0,0),5:(1,0,0)})
                self.assertEqual({(r.userId,r.movieId,r.rating,r.timestamp) for r in saved.collect()}, set(rows))
                # 分区数变化不会改变边界与记录归属。
                original = spark.read.parquet(str(case / "processed" / "ratings"))
                again = prepare_data.split_ratings(original.repartition(1), 80, 10)
                self.assertEqual(set(map(tuple, again.select(*saved.columns).collect())), set(map(tuple, saved.collect())))
                legacy = prepare_data.split_ratings(original, 90)
                self.assertEqual({r.split: r['count'] for r in legacy.groupBy('split').count().collect()},
                                 {'train':33,'validation':4})
                test = spark.read.parquet(str(case / "processed" / "test_pairs")).orderBy("row_id").collect()
                self.assertEqual([(r.row_id,r.userId,r.movieId) for r in test], [(0,4,30),(1,1,21)])
                movie = spark.read.parquet(str(case / "processed" / "movies")).first()
                self.assertEqual(movie.title, '电影,"带引号"')
            finally:
                spark.stop()
            self.assertTrue(result["output_files"])
            reuse = prepare_data.checked_processed(case / 'processed', result['input_files'], case / 'report')
            self.assertEqual(reuse['training_by_source'], result['training_by_source'])
            changed_inputs = [dict(item) for item in result['input_files']]
            changed_inputs[0]['sha256'] = 'changed'
            with self.assertRaisesRegex(ValueError, '未找到'):
                prepare_data.checked_processed(case / 'processed', changed_inputs, case / 'report')
            self.assertTrue((case / "report" / "data_preparation.md").is_file())
            with self.assertRaises(FileExistsError):
                prepare_data.run_preparation(cfg, case / "other-report")
            (raw / "test_pairs.csv").write_text("wrong,header\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "字段"):
                prepare_data.inspect_inputs(raw)


if __name__ == "__main__":
    unittest.main()
