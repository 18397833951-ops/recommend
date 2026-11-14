"""MMF 手算、官方数值/梯度对照、最佳轮次及完整保存流程。"""

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import mmf
from biased_mf import Ratings
import test_biased_mf as shared


def config(overrides=()):
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT/'configs')):
        return compose(config_name='mmf', overrides=['model.rank=3','model.components=2',
            'model.max_epochs=3','model.batch_size=2','model.num_threads=2',
            'data.read_partitions=2','spark.shuffle_partitions=2',*overrides])


class MMFTest(unittest.TestCase):
    def test_hand_formula_and_unknown_ids(self):
        model = mmf.MaskedMF(3,3,3,2,3.5,.02)
        with torch.no_grad():
            model.known_users[1] = model.known_movies[1] = True
            model.user_factors.weight[1] = torch.tensor([.2,.4,.6])
            model.movie_factors.weight[1] = torch.tensor([.3,.5,.7])
            model.user_bias.weight[1] = .2
            model.movie_bias.weight[1] = -.1
            model.user_shifts.weight[1] = torch.tensor([.2,-.1])
            model.movie_shifts.weight[1] = torch.tensor([-.3,.4])
        expected = 3.5+.2-.1
        for k,(us,ms) in enumerate(zip([.2,-.1],[-.3,.4])):
            for r,(p,q) in enumerate(zip([.2,.4,.6],[.3,.5,.7])):
                a=(.5+math.exp(-((r-1.5*us)*(k+1)/2)**2/2)/2.507)/2
                b=(.5+math.exp(-((r-1.5*ms)*(k+1)/2)**2/2)/2.507)/2
                expected += p*q*a*b
        self.assertAlmostEqual(model(torch.tensor([1]),torch.tensor([1])).item(),expected,places=6)
        values=model(torch.tensor([1,99,99,2]),torch.tensor([99,1,99,1])).tolist()
        for actual,wanted in zip(values,[3.7,3.4,3.5,3.4]):
            self.assertAlmostEqual(actual,wanted,places=6)
        model.loss(torch.tensor([1]),torch.tensor([1]),torch.tensor([5.]),0).backward()
        self.assertGreater(model.user_shifts.weight.grad.abs().sum().item(),0)
        self.assertGreater(model.movie_shifts.weight.grad.abs().sum().item(),0)

    @unittest.skipUnless((ROOT/'.runtime/mmf_reference.py').exists(),'可选官方源码对照文件未下载')
    def test_matches_pinned_official_predictions_and_gradients(self):
        from prepare_data import fingerprint
        self.assertEqual(fingerprint(ROOT/'.runtime/mmf_reference.py')['sha256'],mmf.REFERENCE['source_sha256'])
        spec=importlib.util.spec_from_file_location('mmf_official_reference',ROOT/'.runtime/mmf_reference.py')
        reference=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=reference
        spec.loader.exec_module(reference)
        torch.manual_seed(7)
        ours=mmf.MaskedMF(4,5,3,2,3.5,.02)
        official=reference.MMF(4,5,3,K=2,mu=3.5,mask_mode_A='gaussian',
                               mask_mode_B='gaussian',shift_mode='learned')
        pairs=[(ours.user_factors.weight,official.A),(ours.movie_factors.weight,official.B),
               (ours.user_bias.weight,official.bu),(ours.movie_bias.weight,official.bi),
               (ours.user_shifts.weight,official.shiftA),(ours.movie_shifts.weight,official.shiftB)]
        with torch.no_grad():
            ours.known_users[:]=ours.known_movies[:]=True
            for index,(a,b) in enumerate(pairs):
                a.uniform_(-.5,.5)
                b.copy_(a.T if index>=4 else a.squeeze(1) if index>=2 else a)
        u,i,y=torch.tensor([1,2,3,1]),torch.tensor([4,2,1,3]),torch.tensor([4.,2.,1.,5.])
        torch.testing.assert_close(ours(u,i),official.predict(u,i),atol=1e-6,rtol=1e-6)
        ours.loss(u,i,y,0).backward()
        (official.predict(u,i)-y).square().mean().backward()
        for index,(a,b) in enumerate(pairs):
            wanted=b.grad.T if index>=4 else b.grad[:,None] if index>=2 else b.grad
            torch.testing.assert_close(a.grad,wanted,atol=1e-6,rtol=1e-5)

    def test_restores_best_epoch_and_trains_shifts(self):
        cfg=config(['model.max_epochs=5','model.patience=2'])
        train=Ratings(np.array([1,1,2,2],np.int32),np.array([1,2,1,2],np.int32),np.array([1.,2.,4.,5.],np.float32))
        snapshots=[]
        errors=iter([.5,.2,.3,.4])
        def evaluate(model,*args):
            snapshots.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
            return {'rows':4,'rmse':next(errors)}
        with tempfile.TemporaryDirectory(prefix='mmf-epochs-',dir=ROOT/'.runtime') as folder:
            report={}
            with patch.object(mmf,'evaluate_model',side_effect=evaluate):
                model=mmf.fit_model(train,train,cfg.model,Path(folder)/'best.pt',report)
            self.assertEqual((report['best_epoch'],report['epochs_trained']),(2,4))
            for name,value in model.state_dict().items():
                self.assertTrue(torch.equal(value,snapshots[1][name]),name)
            self.assertGreater(model.user_shifts.weight.abs().sum().item(),0)
            self.assertGreater(model.movie_shifts.weight.abs().sum().item(),0)

    def test_pipeline_freezes_before_test_and_checks_saved_predictions(self):
        # 同一套独立验收：原记录、冷启动、时间戳、最佳模型、读回 RMSE 和文件指纹。
        with patch.object(shared,'config',side_effect=lambda overrides=():config(overrides)), \
             patch.object(shared,'mf',mmf), \
             patch.object(mmf,'run_biased_mf',mmf.run_mmf,create=True):
            shared.JointMFTest.test_pipeline_freezes_before_test_and_reads_back_saved_predictions(self)


if __name__=='__main__':
    unittest.main()
