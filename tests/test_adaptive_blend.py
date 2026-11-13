"""检查输入含义、权重边界、验证恢复，以及训练/验证/测试读取顺序。"""

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
import adaptive_blend as ab
from biased_mf import BiasedMF
from prepare_data import fingerprint, start_spark


def config(overrides=()):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name="adaptive_blend", overrides=["model.hidden_size=2", "model.batch_size=2",
            "model.max_epochs=3", "model.num_threads=2", "spark.shuffle_partitions=2", "data.read_partitions=2", *overrides])


class NetworkTest(unittest.TestCase):
    def test_count_changes_weight_and_prediction_is_convex(self):
        model = ab.AdaptiveBlend(2, [0.]*4, [1.]*4)
        with torch.no_grad():
            model.network[0].weight.zero_()
            model.network[0].bias.zero_()
            model.network[0].weight[0, 2] = 1.
            model.network[-1].weight[0, 0] = 1.
        features = torch.tensor([[4., 2., 0., 0.], [4., 2., 2., 0.]])
        alpha = model(features)
        self.assertAlmostEqual(alpha[0].item(), .5)
        self.assertAlmostEqual(alpha[1].item(), torch.sigmoid(torch.tensor(2.)).item())
        self.assertTrue(torch.all((alpha > 0) & (alpha < 1)).item())
        score = features[:, 1] + alpha * (features[:, 0]-features[:, 1])
        self.assertTrue(torch.all((score >= 2) & (score <= 4)).item())
        loss = (score-torch.tensor([2., 3.])).square().mean()
        loss.backward()
        self.assertGreater(model.network[0].weight.grad.abs().sum().item(), 0.)
        self.assertGreater(model.network[-1].weight.grad.abs().sum().item(), 0.)

    def test_train_only_statistics_and_best_validation_epoch(self):
        x = np.array([[1., 2., 1., 2.], [2., 4., 2., 3.], [4., 3., 3., 4.]], np.float32)
        train = ab.BlendData(x.copy(), np.array([1., 3., 4.]), x[:, :2].copy())
        validation = ab.BlendData(x+10, np.array([1., 3., 4.]), x[:, :2].copy())
        scores, snapshots = iter([.8, .7, .6, .65, .66]), []
        def evaluate(model, *args):
            snapshots.append({k: v.detach().cpu().clone() for k,v in model.state_dict().items()})
            return {"rmse": next(scores)}
        with tempfile.TemporaryDirectory(prefix="adaptive-epoch-", dir=ROOT/".runtime") as folder:
            path, report = Path(folder)/"best.pt", {}
            cfg = config(["model.max_epochs=5", "model.patience=2"])
            with patch.object(ab, "evaluate", side_effect=evaluate):
                model = ab.fit_model(train, validation, cfg.model, .4, path, report)
            self.assertEqual((report["best_epoch"], report["epochs_trained"]), (2, 4))
            np.testing.assert_allclose(model.mean.cpu().numpy(), x.mean(0))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), snapshots[2][key]), key)
            np.testing.assert_array_equal(train.features, x)
            np.testing.assert_array_equal(train.predictions, x[:, :2])


class PipelineTest(unittest.TestCase):
    def test_small_pipeline_base_models_frozen_and_test_loaded_last(self):
        with tempfile.TemporaryDirectory(prefix="adaptive-pipeline-", dir=ROOT/".runtime") as folder:
            case = Path(folder)
            cfg = config([f"paths.raw_dir={(case/'raw').as_posix()}", f"paths.splits_dir={(case/'splits').as_posix()}",
                          f"paths.als_run={(case/'als').as_posix()}", f"paths.mf_run={(case/'mf').as_posix()}",
                          f"paths.blend_run={(case/'blend').as_posix()}"])
            spark = start_spark(cfg, "AdaptiveBlendFixture")
            try:
                root = case/"splits/ratings"
                spark.createDataFrame([(1,1,2.,1,"train"), (1,2,4.,2,"train"),
                    (2,1,3.,1,"train"), (2,2,5.,2,"train")],
                    "userId int,movieId int,rating double,timestamp long,split string").write.partitionBy("split").parquet(str(root))
                als_files = []
                for name in ("userFactors", "itemFactors"):
                    path = case/"als/models/als"/name
                    spark.createDataFrame([(1,[1.]), (2,[2.])], "id int,features array<float>").write.parquet(str(path))
                    als_files += [fingerprint(p) for p in path.rglob("*.parquet")]
                mf = BiasedMF(3, 3, 1, 3., .05)
                mf.known_users[1:] = mf.known_movies[1:] = True
                mf_path = case/"mf/models/best.pt"
                mf_path.parent.mkdir(parents=True)
                torch.save({"num_users":3,"num_movies":3,"rank":1,"global_mean":3.,"state_dict":mf.state_dict()},mf_path)
                before = fingerprint(mf_path)
                source = {"files":[{"path":str(case/"fixed.parquet"), "sha256":"fixture"}],
                          "counts":[{"split":p,"count":n} for p,n in (("train",4),("validation",2),("test",2))]}
                runs = {"als":(case/"als", {"source":source,"output_files":als_files}),
                        "mf":(case/"mf", {"source":source,"output_files":[before]})}
                inputs=[]
                for name, filename in (("als","als.json"),("mf","biased_mf.json")):
                    path=case/name/filename
                    runs[name][1]["status"]="passed"
                    path.write_text(json.dumps(runs[name][1]),encoding="utf-8")
                    inputs.append(fingerprint(path))
                blend_report = {"status":"passed","input_reports":inputs,"weights":{"als":.4,"mf":.6},"metrics":[],"output_files":[]}
                for phase, ids in (("validation", [(1,2),(2,1)]), ("test",[(9,2),(1,9)])):
                    path = case/"blend/predictions"/f"split={phase}"
                    rows = [(u,i,3,4.,3.5,4.5) for u,i in ids]
                    spark.createDataFrame(rows,"userId int,movieId int,timestamp long,rating double,als_prediction double,mf_prediction double").write.parquet(str(path))
                    blend_report["output_files"] += [fingerprint(p) for p in path.rglob("*.parquet")]
                    for name in ("als","mf"):
                        blend_report["metrics"].append({"model":name,"split":phase,"rows":2,"rmse":.5})
                (case/"blend/blend.json").write_text(json.dumps(blend_report),encoding="utf-8")
                events=[]
                original_prepare, original_load, original_fit = ab.prepare_training, ab.load_phase, ab.fit_model
                def prepare(*args):
                    events.append("train")
                    data, users, movies = original_prepare(*args)
                    np.testing.assert_array_equal(users,[0,2,2])
                    np.testing.assert_array_equal(movies,[0,2,2])
                    self.assertEqual(data.features.shape,(4,4))
                    np.testing.assert_allclose(data.features[:,2:],np.log(3))
                    return data,users,movies
                def load(*args):
                    events.append(args[3])
                    if args[3]=="test":
                        self.assertTrue((case/"result/models/parameters.json").is_file())
                    return original_load(*args)
                def fit(*args):
                    result=original_fit(*args)
                    events.append("best_restored")
                    return result
                with patch.object(ab,"check_input",return_value=(root,source)), \
                     patch.object(ab,"prepare_training",side_effect=prepare), patch.object(ab,"load_phase",side_effect=load), patch.object(ab,"fit_model",side_effect=fit):
                    report=ab.run_adaptive_blend(cfg,case/"result")
                self.assertEqual(events,["train","validation","best_restored","test"])
                self.assertEqual(report["status"],"passed")
                self.assertEqual(before,fingerprint(mf_path))
                self.assertEqual([r["rows"] for r in report["metrics"]],[2,2])
                self.assertTrue(all(fingerprint(Path(f["path"]))==f for f in report["output_files"]))
                for row in report["metrics"]:
                    self.assertGreaterEqual(row["als_weight"]["min"],0.)
                    self.assertLessEqual(row["als_weight"]["max"],1.)
            finally:
                spark.stop()


if __name__=="__main__":
    unittest.main()
