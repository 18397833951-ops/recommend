"""手算均值与逐记录 RMSE，核对回退、训练隔离及保存读回。"""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import baselines
from prepare_data import fingerprint, start_spark


class BaselineTest(unittest.TestCase):
    def test_training_means_weighted_rmse_cold_starts_and_saved_pipeline(self):
        from pyspark.sql import functions as F

        with tempfile.TemporaryDirectory(prefix="baseline-test-", dir=ROOT / ".runtime") as folder:
            case = Path(folder)
            with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
                cfg = compose(config_name="baselines", overrides=["spark.shuffle_partitions=2",
                    f"paths.splits_dir={(case/'splits').as_posix()}",
                    f"paths.preparation_reports_dir={(case/'preparation').as_posix()}"])
            spark = start_spark(cfg, "BaselineFixture")
            try:
                rows = [(1,30,1.0,1,'movielens32m','train'), (1,40,3.0,2,'movielens32m','train'),
                        (2,10,3.0,1,'movielens32m','train'), (2,20,5.0,2,'movielens32m','train'),
                        (2,30,5.0,3,'movielens32m','validation'), (99,199,1.0,3,'synthetic','validation'),
                        (3,40,3.0,4,'movielens32m','test'), (1,50,5.0,4,'movielens32m','test')]
                ratings = spark.createDataFrame(rows, 'userId long, movieId long, rating double, timestamp long, source string, split string')
                ratings.write.partitionBy('split').parquet(str(case/'splits/ratings'))
                counts = [row.asDict() for row in ratings.groupBy('split','source').count().collect()]
                source = {'status':'passed','paths':{'splits':str(case/'splits')},
                          'summary':{'by_source_and_split':counts},
                          'output_files':[fingerprint(path) for path in (case/'splits').rglob('*.parquet')]}
                (case/'preparation').mkdir()
                report_path = case/'preparation/data_preparation.json'
                report_path.write_text(json.dumps(source),encoding='utf-8')
                report = baselines.run_baselines(cfg, case/'result')
                self.assertEqual(report['status'],'passed')
                self.assertEqual(report['training'],{'global_mean':3.0,'rows':4})
                self.assertEqual(len(report['metrics']),6)
                self.assertTrue(all('source' not in row for row in report['metrics']))
                scores = {(r['split'],r['model']):r for r in report['metrics']}
                self.assertAlmostEqual(scores['validation','global_mean']['rmse'],2.0)
                self.assertAlmostEqual(scores['validation','user_mean']['rmse'],math.sqrt(2.5))
                self.assertAlmostEqual(scores['validation','movie_mean']['rmse'],math.sqrt(10))
                self.assertAlmostEqual(scores['test','user_mean']['rmse'],math.sqrt(4.5))
                self.assertAlmostEqual(scores['test','movie_mean']['rmse'],math.sqrt(2))
                for phase in ('validation','test'):
                    for model in ('user_mean','movie_mean'):
                        self.assertEqual(scores[phase,model]['fallback_rows'],1)
                spark = start_spark(cfg)
                saved = spark.read.parquet(str(case/'result/predictions'))
                self.assertEqual(saved.count(),4)
                users = spark.read.parquet(str(case/'result/models/users'))
                movies = spark.read.parquet(str(case/'result/models/movies'))
                self.assertEqual({r.userId:r.mean_rating for r in users.collect()},{1:2.0,2:4.0})
                evaluation = saved.select('split','source','userId','movieId','rating','timestamp')
                altered = evaluation.withColumn('rating',6-F.col('rating'))
                changed = baselines.predict_means(altered,users,movies,3.0)
                columns=['userId','movieId',*[f'{model}_prediction' for model in baselines.MODELS]]
                self.assertEqual(set(map(tuple,saved.select(*columns).collect())),set(map(tuple,changed.select(*columns).collect())))
                with self.assertRaisesRegex(ValueError,'NaN'):
                    baselines.evaluate_predictions(saved.withColumn('movie_mean_prediction',F.lit(float('nan'))))
                self.assertTrue((case/'result/baselines.md').is_file())
                self.assertTrue(report['output_files'])
                with self.assertRaises(FileExistsError):
                    baselines.run_baselines(cfg,case/'result')
                source['output_files'][0]['sha256']='changed'
                report_path.write_text(json.dumps(source),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'变化'):
                    baselines.check_input(cfg)
            finally:
                spark.stop()


if __name__ == '__main__':
    unittest.main()
