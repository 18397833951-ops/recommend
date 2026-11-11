"""手算正则偏置，检查无标签预测、冷启动、验证选优与独立测试评估。"""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bias
from prepare_data import fingerprint, start_spark


class BiasTest(unittest.TestCase):
    def test_coordinate_updates_selection_and_saved_predictions(self):
        from pyspark.sql import functions as F

        with tempfile.TemporaryDirectory(prefix="bias-test-", dir=ROOT / ".runtime") as folder:
            case = Path(folder)
            with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
                cfg = compose(config_name="bias", overrides=["spark.shuffle_partitions=2",
                    "model.regularizations=[1.0,10.0]", "model.max_epochs=6",
                    f"paths.splits_dir={(case/'splits').as_posix()}",
                    f"paths.preparation_reports_dir={(case/'preparation').as_posix()}"])
            spark = start_spark(cfg, "BiasFixture")
            try:
                rows = [(1,10,1.0,1,'a','train'), (1,20,3.0,2,'a','train'),
                        (2,10,3.0,1,'b','train'), (2,20,5.0,2,'b','train'), (1,30,2.0,2,'a','train'),
                        (1,10,2.0,3,'a','validation'), (9,99,1.0,3,'b','validation'),
                        (2,99,4.0,4,'a','test'), (9,20,5.0,4,'b','test')]
                ratings = spark.createDataFrame(rows, 'userId long, movieId long, rating double, timestamp long, source string, split string')
                train = ratings.where("split='train' AND movieId<>30").select('userId','movieId','rating')
                users, movies = bias.update_biases(train, None, 3.0, 1.0)
                try:
                    self.assertEqual({r.movieId: round(r.movie_bias,12) for r in movies.collect()}, {10:round(-2/3,12),20:round(2/3,12)})
                    self.assertEqual({r.userId: round(r.user_bias,12) for r in users.collect()}, {1:round(-2/3,12),2:round(2/3,12)})
                    # 手算一轮后的正则目标；正则不是按条数重复惩罚每个偏置。
                    fitted = bias.predict_bias(train, users, movies, 3.0, cfg.model)
                    sse = fitted.agg(F.sum(F.pow(F.col('rating')-F.col('prediction'),2))).first()[0]
                    self.assertAlmostEqual(sse + 4*(2/3)**2, 8/3)
                    unlabeled = spark.createDataFrame([(1,10),(2,99),(9,20),(9,99)],'userId long, movieId long')
                    predictions = bias.predict_bias(unlabeled,users,movies,3.0,cfg.model)
                    values = {(r.userId,r.movieId): r.prediction for r in predictions.collect()}
                    for pair, expected in { (1,10):5/3, (2,99):11/3, (9,20):11/3, (9,99):3.0 }.items():
                        self.assertAlmostEqual(values[pair], expected)
                    labeled = predictions.withColumn('rating', F.lit(3.0))
                    score = bias.evaluate(labeled)
                    self.assertAlmostEqual(score['rmse'], math.sqrt(2/3))
                    self.assertEqual(score['fallback_rows'],3)
                    self.assertEqual(score['user_fallback_rows'],2)
                    self.assertEqual(score['movie_fallback_rows'],2)
                    extremes = spark.createDataFrame([(1,100.0)], 'userId long, user_bias double')
                    clipped = bias.predict_bias(unlabeled,extremes,movies,3.0,cfg.model)
                    self.assertEqual(clipped.where('userId=1').first().prediction,5.0)
                    with self.assertRaisesRegex(ValueError,'NaN'):
                        bias.evaluate(labeled.withColumn('prediction',F.lit(float('nan'))))
                finally:
                    users.unpersist()
                    movies.unpersist()
                ratings.write.partitionBy('split').parquet(str(case/'splits/ratings'))
                counts = [r.asDict() for r in ratings.groupBy('split','source').count().collect()]
                manifest = {'status':'passed','paths':{'splits':str(case/'splits')},
                            'summary':{'by_source_and_split':counts},
                            'output_files':[fingerprint(p) for p in (case/'splits').rglob('*.parquet')]}
                (case/'preparation').mkdir()
                (case/'preparation/data_preparation.json').write_text(json.dumps(manifest),encoding='utf-8')
                report = bias.run_bias(cfg,case/'result')
                self.assertEqual(report['status'],'passed')
                self.assertEqual(report['training'],{'global_mean':2.8,'rows':5})
                self.assertEqual(len(report['trials']),12)
                best = min(report['trials'],key=lambda trial:trial['rmse'])
                self.assertEqual(report['selected'],best)
                self.assertAlmostEqual(report['metrics'][0]['rmse'],best['rmse'])
                self.assertEqual([(r['split'],r['rows']) for r in report['metrics']],[('validation',2),('test',2)])
                # 用单独的标量计算重现所有训练轮次与验证误差，完全不读取测试标签。
                expected_trials = []
                training_rows = [(1,10,1.0),(1,20,3.0),(2,10,3.0),(2,20,5.0),(1,30,2.0)]
                for reg in (1.0,10.0):
                    user_values = {1:0.0,2:0.0}
                    for epoch in range(1,7):
                        movie_values = {movie: sum(rating-2.8-user_values[user] for user,item,rating in training_rows if item==movie)
                                        / (sum(item==movie for _,item,_ in training_rows)+reg)
                                        for movie in (10,20,30)}
                        user_values = {user: sum(rating-2.8-movie_values[movie] for person,movie,rating in training_rows if person==user)
                                       / (sum(person==user for person,_,_ in training_rows)+reg)
                                       for user in (1,2)}
                        prediction = min(5.0,max(0.5,2.8+user_values[1]+movie_values[10]))
                        expected_trials.append(math.sqrt(((prediction-2.0)**2+1.8**2)/2))
                for trial, expected in zip(report['trials'],expected_trials):
                    self.assertAlmostEqual(trial['rmse'],expected)
                spark = start_spark(cfg)
                saved = spark.read.parquet(str(case/'result/predictions'))
                self.assertEqual(saved.count(),4)
                self.assertTrue(all(fingerprint(Path(f['path']))==f for f in report['output_files']))
                with self.assertRaises(FileExistsError):
                    bias.run_bias(cfg,case/'result')
            finally:
                spark.stop()


if __name__ == '__main__':
    unittest.main()
