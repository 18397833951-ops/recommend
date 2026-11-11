"""手算评分版 ItemCF，核对精确余弦、小样本近似路径和完整结果读回。"""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

from hydra import compose, initialize_config_dir
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import itemcf
from itemcf_scoring import predict_user
from prepare_data import fingerprint, start_spark


class ItemCFTest(unittest.TestCase):
    def test_weighted_residuals_fallbacks_clipping_and_label_isolation(self):
        model={'min_neighbors':2,'clip_predictions':True,'rating_min':0.5,'rating_max':5.0}
        catalog={10:(3.0,np.array([20,30]),np.array([0.8,0.4]))}
        target=[(10,4.0,3,'fixture')]
        row=list(predict_user(1,[(20,1.0),(30,-0.5)],target,4.0,catalog,3.0,model))[0]
        self.assertAlmostEqual(row[5],3.5)
        self.assertAlmostEqual(row[6],3.5)
        self.assertEqual(row[7:],(2,False,False,False))
        changed=list(predict_user(1,[(20,1.0),(30,-0.5)],[(10,1.0,3,'fixture')],4.0,catalog,3.0,model))[0]
        self.assertEqual(row[5:],changed[5:])
        insufficient=list(predict_user(1,[(20,1.0)],target,4.0,catalog,3.0,model))[0]
        self.assertEqual((insufficient[6],insufficient[-1]),(3.0,True))
        unknown_movie=list(predict_user(1,[(20,1.0)],[(99,2.0,3,'fixture')],4.0,catalog,3.0,model))[0]
        self.assertEqual(unknown_movie[6],4.0)
        unknown_user=list(predict_user(99,None,target,None,catalog,3.0,model))[0]
        self.assertEqual(unknown_user[6],3.0)
        both=list(predict_user(99,None,[(99,1.0,3,'fixture')],None,catalog,3.0,model))[0]
        self.assertEqual(both[6],3.0)
        clipped=list(predict_user(1,[(20,10.0),(30,10.0)],target,4.0,catalog,3.0,model))[0]
        self.assertEqual(clipped[6],5.0)
        zero=list(predict_user(1,[(20,0.0),(30,0.0)],target,4.0,catalog,3.0,model))[0]
        self.assertEqual((zero[7],zero[-1]),(2,False))

    def test_exact_cosine_sparse_high_ids_and_saved_pipeline(self):
        from pyspark.sql import functions as F

        with tempfile.TemporaryDirectory(prefix='itemcf-test-',dir=ROOT/'.runtime') as folder:
            case=Path(folder)
            with initialize_config_dir(version_base='1.3',config_dir=str(ROOT/'configs')):
                cfg=compose(config_name='itemcf',overrides=['spark.driver_memory=1g','spark.shuffle_partitions=2',
                    'model.sampling_threshold=0','model.min_item_ratings=1','model.min_neighbors=1','model.neighbors=2',
                    f"paths.splits_dir={(case/'splits').as_posix()}",f"paths.preparation_reports_dir={(case/'preparation').as_posix()}"])
            spark=start_spark(cfg,'ItemCFFixture')
            try:
                values=[[5.,4.,1.,2.],[4.,5.,2.,1.],[1.,2.,5.,4.],[2.,1.,4.,5.]]
                ids=[10,20,30,70000]
                rows=[(user,movie,rating,1,'fixture','train') for user,ratings in enumerate(values,1) for movie,rating in zip(ids,ratings)]
                rows += [(1,10,5.,3,'fixture','validation'),(99,10,4.,3,'fixture','validation'),
                         (1,99,2.,3,'fixture','validation'),(99,99,1.,3,'fixture','validation'),
                         (3,30,5.,4,'fixture','test'),(4,20,1.,4,'fixture','test')]
                ratings=spark.createDataFrame(rows,'userId long, movieId long, rating double, timestamp long, source string, split string')
                ratings.write.partitionBy('split').parquet(str(case/'splits/ratings'))
                counts=[row.asDict() for row in ratings.groupBy('split','source').count().collect()]
                manifest={'status':'passed','paths':{'splits':str(case/'splits')},'summary':{'by_source_and_split':counts},
                          'output_files':[fingerprint(path) for path in (case/'splits').rglob('*.parquet')]}
                (case/'preparation').mkdir()
                (case/'preparation/data_preparation.json').write_text(json.dumps(manifest),encoding='utf-8')
                report=itemcf.run_itemcf(cfg,case/'result')
                self.assertEqual(report['status'],'passed')
                self.assertEqual(report['training'],{'global_mean':3.0,'rows':16})
                self.assertEqual(report['graph']['directed_neighbors'],4)
                self.assertEqual(report['graph']['vector_dimension'],70001)
                self.assertAlmostEqual(report['metrics'][0]['rmse'],math.sqrt(7/4))
                self.assertAlmostEqual(report['metrics'][1]['rmse'],1.0)
                self.assertEqual(report['metrics'][0]['fallback_rows'],3)
                self.assertEqual(report['metrics'][0]['user_fallback_rows'],2)
                self.assertEqual(report['metrics'][0]['movie_fallback_rows'],2)
                spark=start_spark(cfg)
                ratings=spark.read.parquet(str(case/'splits/ratings'))
                graph=spark.read.parquet(str(case/'result/models/neighbors'))
                found={r.movieId:(list(r.neighbor_ids),list(r.similarities)) for r in graph.collect()}
                for movie, neighbor in {10:20,20:10,30:70000,70000:30}.items():
                    self.assertEqual(found[movie][0],[neighbor])
                    self.assertAlmostEqual(found[movie][1][0],0.8)
                users=spark.read.parquet(str(case/'result/models/users'))
                movies=spark.read.parquet(str(case/'result/models/movies'))
                # 小数据采样概率饱和，近似路径也应得到相同邻居。
                cfg.model.sampling_threshold=1.0
                approximate=itemcf.fit_neighbors(spark,ratings.where("split='train'"),users,movies,cfg,case/'approximate')
                self.assertEqual(approximate['directed_neighbors'],4)
                approx=spark.read.parquet(str(case/'approximate'))
                self.assertEqual({r.movieId:list(r.neighbor_ids) for r in approx.collect()},
                                 {movie:entries[0] for movie,entries in found.items()})
                # 所有用户评分完全相同时，中心化后无非零向量，保留均值回退。
                constant=ratings.where("split='train'").withColumn('rating',F.lit(3.0))
                empty=itemcf.fit_neighbors(spark,constant,users,movies,cfg,case/'empty')
                self.assertEqual(empty['directed_neighbors'],0)
                saved=spark.read.parquet(str(case/'result/predictions'))
                self.assertEqual(saved.count(),6)
                with self.assertRaisesRegex(ValueError,'NaN'):
                    itemcf.evaluate(saved.withColumn('prediction',F.lit(float('nan'))))
                self.assertTrue(all(fingerprint(Path(item['path']))==item for item in report['output_files']))
                with self.assertRaises(FileExistsError):
                    itemcf.run_itemcf(cfg,case/'result')
            finally:
                spark.stop()


if __name__=='__main__':
    unittest.main()
