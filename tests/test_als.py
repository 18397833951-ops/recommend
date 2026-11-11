"""核对配置驱动的单次 ALS 训练、点积、冷启动和保存读回。"""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import als
from prepare_data import fingerprint, start_spark


class ALSTest(unittest.TestCase):
    def test_single_fit_saved_model_dot_products_and_cold_starts(self):
        from pyspark.ml.recommendation import ALS, ALSModel
        from pyspark.sql import functions as F

        with tempfile.TemporaryDirectory(prefix="als-test-", dir=ROOT / ".runtime") as folder:
            case = Path(folder)
            with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
                cfg = compose(config_name="als", overrides=["spark.driver_memory=1g", "spark.shuffle_partitions=2",
                    "model.rank=2", "model.reg_param=0.01", "model.max_iter=5",
                    "model.num_user_blocks=2", "model.num_item_blocks=2",
                    f"paths.splits_dir={(case/'splits').as_posix()}",
                    f"paths.preparation_reports_dir={(case/'preparation').as_posix()}"])
            spark = start_spark(cfg, "ALSFixture")
            try:
                rows = []
                factors = {1:1.0,2:1.2,3:1.4,4:1.6}
                movie_values = {10:0.8,20:1.2,30:1.6,40:2.0}
                for user, value in factors.items():
                    for movie, movie_value in movie_values.items():
                        phase = 'train' if movie != user*10 else ('validation' if user<=2 else 'test')
                        rows.append((user,movie,value*movie_value,1 if phase=='train' else 2,'fixture',phase))
                rows += [(9,10,2.0,2,'fixture','validation'),(1,99,4.0,2,'fixture','validation'),
                         (9,99,3.0,3,'fixture','test')]
                ratings = spark.createDataFrame(rows, 'userId long, movieId long, rating double, timestamp long, source string, split string')
                ratings.write.partitionBy('split').parquet(str(case/'splits/ratings'))
                counts = [row.asDict() for row in ratings.groupBy('split','source').count().collect()]
                manifest = {'status':'passed','paths':{'splits':str(case/'splits')},
                            'summary':{'by_source_and_split':counts},
                            'output_files':[fingerprint(p) for p in (case/'splits').rglob('*.parquet')]}
                (case/'preparation').mkdir()
                (case/'preparation/data_preparation.json').write_text(json.dumps(manifest),encoding='utf-8')
                with patch.object(ALS, 'fit', autospec=True, side_effect=ALS.fit) as fit:
                    report = als.run_als(cfg,case/'result')
                    self.assertEqual(fit.call_count,1)
                    estimator = fit.call_args.args[0]
                    self.assertEqual(estimator.getRank(),2)
                    self.assertEqual(estimator.getRegParam(),0.01)
                    self.assertEqual(estimator.getMaxIter(),5)
                self.assertEqual(report['status'],'passed')
                self.assertEqual(report['training']['rows'],12)
                global_mean = sum(r[2] for r in rows if r[-1]=='train')/12
                self.assertAlmostEqual(report['training']['global_mean'],global_mean)
                self.assertNotIn('trials',report)
                self.assertNotIn('selected',report)
                self.assertAlmostEqual(report['fit']['rmse'],report['metrics'][0]['rmse'])
                self.assertEqual([(r['split'],r['rows'],r['fallback_rows']) for r in report['metrics']],
                                 [('validation',4,2),('test',3,1)])
                spark = start_spark(cfg)
                fitted = ALSModel.load(str(case/'result/models/als'))
                users = {r.id:r.features for r in fitted.userFactors.collect()}
                movies = {r.id:r.features for r in fitted.itemFactors.collect()}
                self.assertTrue(all(len(vector)==report['fit']['rank'] for vector in [*users.values(),*movies.values()]))
                means = spark.read.parquet(str(case/'result/models/user_means'))
                saved = spark.read.parquet(str(case/'result/predictions'))
                saved_rows = saved.collect()
                self.assertEqual(len(saved_rows),7)
                manual_sse = {'validation':0.0,'test':0.0}
                warm_errors = []
                for row in saved_rows:
                    if row.userId in users and row.movieId in movies:
                        dot = sum(u*i for u,i in zip(users[row.userId],movies[row.movieId]))
                        self.assertAlmostEqual(row.als_prediction,dot,places=5)
                        expected = min(5.0,max(0.5,dot))
                        warm_errors.append((row.prediction-row.rating)**2)
                    elif row.userId==1:
                        expected = 1.6
                    else:
                        expected = global_mean
                    self.assertAlmostEqual(row.prediction,expected,places=5)
                    manual_sse[row.split] += (row.prediction-row.rating)**2
                # 已知低秩评分矩阵的留出值应能恢复，防止误用隐式反馈训练。
                self.assertLess(math.sqrt(sum(warm_errors)/len(warm_errors)),0.1)
                for score in report['metrics']:
                    self.assertAlmostEqual(score['squared_error_sum'],manual_sse[score['split']])
                # 正式待预测表没有 rating，仍然可以预测，并且保持原始行号。
                pairs = saved.select('userId','movieId').withColumn('row_id',F.col('userId')*100+F.col('movieId'))
                predicted = als.predict_als(pairs,fitted,means,global_mean,cfg.model)
                self.assertEqual(predicted.count(),7)
                self.assertTrue('row_id' in predicted.columns and 'rating' not in predicted.columns)
                changed = als.predict_als(pairs.withColumn('rating',F.lit(100.0)),fitted,means,global_mean,cfg.model)
                columns = ['userId','movieId','prediction']
                self.assertEqual(set(map(tuple,predicted.select(*columns).collect())),set(map(tuple,changed.select(*columns).collect())))
                with self.assertRaisesRegex(ValueError,'NaN'):
                    als.evaluate(saved.withColumn('prediction',F.lit(float('nan'))))
                self.assertTrue(all(fingerprint(Path(item['path']))==item for item in report['output_files']))
                with self.assertRaises(FileExistsError):
                    als.run_als(cfg,case/'result')
            finally:
                spark.stop()


if __name__ == '__main__':
    unittest.main()
