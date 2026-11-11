"""核对联合更新、正则损失、最佳轮次恢复及独立测试评估。"""

import csv
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import biased_mf as mf
from prepare_data import fingerprint, start_spark


def config(overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="biased_mf", overrides=["model.rank=2", "model.max_epochs=3",
                       "model.batch_size=2", "model.learning_rate=0.03", "model.num_threads=2",
                       "data.read_partitions=2", "spark.shuffle_partitions=2", *overrides])


class JointMFTest(unittest.TestCase):
    def test_joint_gradients_regularization_and_unknown_ids(self):
        model = mf.BiasedMF(3,3,1,3.5,0.05)
        with torch.no_grad():
            model.known_users[1] = model.known_movies[1] = True
            model.user_factors.weight[1] = 0.5
            model.movie_factors.weight[1] = 0.4
            model.user_bias.weight[1] = 0.2
            model.movie_bias.weight[1] = 0.1
        users, movies = torch.tensor([1]), torch.tensor([1])
        target = torch.tensor([4.5])
        self.assertAlmostEqual(model(users,movies).item(),4.0)
        self.assertAlmostEqual(model.loss(users,movies,target,0.01).item(),0.25+0.01*0.46,places=6)
        optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
        (0.5*(model(users,movies)-target).square().mean()).backward()
        optimizer.step()
        self.assertAlmostEqual(model.user_bias.weight[1].item(),0.25)
        self.assertAlmostEqual(model.movie_bias.weight[1].item(),0.15)
        self.assertAlmostEqual(model.user_factors.weight[1].item(),0.52)
        self.assertAlmostEqual(model.movie_factors.weight[1].item(),0.425)
        self.assertAlmostEqual(model(users,movies).item(),4.121,places=6)
        raw = model(torch.tensor([1,99,99]),torch.tensor([99,1,99])).detach().tolist()
        for actual, expected in zip(raw,[3.75,3.65,3.5]):
            self.assertAlmostEqual(actual,expected,places=6)
        data = mf.Ratings(np.array([1,99,99],np.int32),np.array([99,1,99],np.int32),np.array([3.,3.,3.],np.float32))
        score = mf.evaluate_model(model,data,config().model)
        self.assertEqual((score['rows'],score['user_fallback_rows'],score['movie_fallback_rows'],score['fallback_rows']),(3,2,2,3))

    def test_restores_best_epoch_instead_of_last_epoch(self):
        cfg = config(["model.max_epochs=5","model.patience=2"])
        train = mf.Ratings(np.array([1,1,2,2],np.int32),np.array([1,2,1,2],np.int32),np.array([1.,2.,4.,5.],np.float32))
        snapshots, errors = [], iter([0.5,0.2,0.3,0.4])
        def validation(model,*args):
            snapshots.append({name: value.detach().cpu().clone() for name,value in model.state_dict().items()})
            return {'rows':4,'rmse':next(errors)}
        with tempfile.TemporaryDirectory(prefix='mf-epoch-test-',dir=ROOT/'.runtime') as folder:
            report = {}
            path = Path(folder)/'best.pt'
            with patch.object(mf,'evaluate_model',side_effect=validation):
                best = mf.fit_model(train,train,cfg.model,path,report)
            self.assertEqual((report['best_epoch'],report['epochs_trained']),(2,4))
            _, checkpoint = mf.load_model(path)
            self.assertEqual(checkpoint['epoch'],2)
            for name,value in best.state_dict().items():
                self.assertTrue(torch.equal(value,snapshots[1][name]),name)
            self.assertFalse(torch.equal(best.user_factors.weight,snapshots[-1]['user_factors.weight']))
            self.assertTrue(torch.count_nonzero(best.user_bias.weight).item()>0)
            self.assertTrue(torch.count_nonzero(best.movie_bias.weight).item()>0)

    def test_pipeline_freezes_before_test_and_reads_back_saved_predictions(self):
        with tempfile.TemporaryDirectory(prefix='mf-pipeline-test-',dir=ROOT/'.runtime') as folder:
            case = Path(folder)
            cfg = config([f"paths.splits_dir={(case/'splits').as_posix()}",
                          f"paths.preparation_reports_dir={(case/'preparation').as_posix()}"])
            spark = start_spark(cfg,'JointMFFixture')
            try:
                rows = [(1,10,1.,1,'fixture','train'),(1,20,3.,2,'fixture','train'),
                        (2,10,3.,1,'fixture','train'),(2,20,5.,2,'fixture','train'),
                        (1,99,2.,3,'fixture','validation'),(2,10,4.,3,'fixture','validation'),
                        (9,20,3.,4,'fixture','test'),(9,99,3.,4,'fixture','test')]
                ratings = spark.createDataFrame(rows,'userId long, movieId long, rating double, timestamp long, source string, split string')
                ratings.write.partitionBy('split').parquet(str(case/'splits/ratings'))
                counts = [row.asDict() for row in ratings.groupBy('split','source').count().collect()]
                manifest = {'status':'passed','paths':{'splits':str(case/'splits')},
                            'summary':{'by_source_and_split':counts},
                            'output_files':[fingerprint(path) for path in (case/'splits').rglob('*.parquet')]}
                (case/'preparation').mkdir()
                (case/'preparation/data_preparation.json').write_text(json.dumps(manifest),encoding='utf-8')
                events=[]
                original_load, original_fit = mf.load_ratings, mf.fit_model
                def load(*args):
                    events.append(args[2])
                    return original_load(*args)
                def fit(*args):
                    result = original_fit(*args)
                    events.append('fit_done')
                    return result
                with patch.object(mf,'load_ratings',side_effect=load),patch.object(mf,'fit_model',side_effect=fit):
                    report = mf.run_biased_mf(cfg,case/'result')
                self.assertEqual(events,['train','validation','fit_done','test'])
                self.assertEqual(report['status'],'passed')
                self.assertEqual(report['training']['rows'],4)
                self.assertAlmostEqual(report['training']['global_mean'],3.0)
                self.assertEqual(report['best_epoch'],min(report['history'],key=lambda row:row['validation']['rmse'])['epoch'])
                best, checkpoint = mf.load_model(case/'result/models/best.pt')
                self.assertEqual((checkpoint['num_users'],checkpoint['num_movies']),(3,21))
                self.assertEqual([(r['split'],r['rows'],r['fallback_rows']) for r in report['metrics']],
                                 [('validation',2,1),('test',2,2)])
                for phase in ('validation','test'):
                    with gzip.open(case/f'result/predictions/{phase}.csv.gz','rt',encoding='utf-8') as source:
                        saved=list(csv.DictReader(source))
                    self.assertEqual(len(saved),2)
                    sse=0.0
                    for row in saved:
                        user,movie=int(row['userId']),int(row['movieId'])
                        raw=best(torch.tensor([user]),torch.tensor([movie])).item()
                        self.assertAlmostEqual(float(row['raw_prediction']),raw)
                        self.assertAlmostEqual(float(row['prediction']),min(5.,max(0.5,raw)))
                        sse+=(float(row['prediction'])-float(row['rating']))**2
                        self.assertEqual(int(row['timestamp']),3 if phase=='validation' else 4)
                    metric=next(row for row in report['metrics'] if row['split']==phase)
                    self.assertAlmostEqual(metric['squared_error_sum'],sse)
                self.assertTrue(all(fingerprint(Path(item['path']))==item for item in report['output_files']))
                with self.assertRaises(FileExistsError):
                    mf.run_biased_mf(cfg,case/'result')
            finally:
                spark.stop()


if __name__=='__main__':
    unittest.main()
