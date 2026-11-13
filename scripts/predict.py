"""加载已选定的两个模型和固定融合权重，导出正式 test_pairs 的逐行预测。"""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import OmegaConf
import torch

from als import predict_als
from biased_mf import load_model
from blend import check_runs
from prepare_data import fingerprint, log, start_spark


def verify_files(report, files):
    expected = {Path(f['path']).resolve(): f for f in report['output_files']}
    for path in files:
        if path not in expected or fingerprint(path) != expected[path]:
            raise ValueError(f'输入文件不匹配：{path}')


def write_predictions(raw_rows, predictions, output):
    if len(raw_rows) != len(predictions) or not np.isfinite(predictions).all():
        raise ValueError('预测数量错误或存在非法数值。')
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['userId', 'movieId', 'timestamp', 'prediction'])
        for row, score in zip(raw_rows, predictions):
            writer.writerow([*row, f'{score:.10f}'])
    # 独立读回，检查原字段、顺序和保存精度。
    with output.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        n = 0
        for n, row in enumerate(reader, 1):
            if n > len(raw_rows) or [row[k] for k in ('userId', 'movieId', 'timestamp')] != raw_rows[n-1]:
                raise ValueError('提交表字段或顺序发生变化。')
            if not math.isclose(float(row['prediction']), predictions[n-1], abs_tol=6e-11, rel_tol=0):
                raise ValueError('保存预测读回不一致。')
        if n != len(raw_rows):
            raise ValueError('提交表行数错误。')


def run_prediction(cfg, output):
    from pyspark.ml.recommendation import ALSModel
    output = Path(output).resolve()
    for path in (cfg.paths.raw_dir, cfg.paths.processed_dir, cfg.paths.als_run, cfg.paths.mf_run, cfg.paths.blend_run):
        path = Path(path).expanduser().resolve()
        if path.is_relative_to(output) or output.is_relative_to(path):
            raise ValueError('输出目录与输入目录不可重叠。')
    if any((output/name).exists() for name in ('predictions.csv', 'submission.json')):
        raise FileExistsError('已有结果，不覆盖。')
    runs, inputs = check_runs(cfg, output)
    blend_root = Path(cfg.paths.blend_run).resolve()
    blend_report = json.loads((blend_root/'blend.json').read_text(encoding='utf-8'))
    if blend_report['status'] != 'passed' or blend_report['input_reports'] != inputs:
        raise ValueError('融合模型输入不一致。')
    parameters_path = blend_root/'models/parameters.json'
    verify_files(blend_report, [parameters_path])
    parameters = json.loads(parameters_path.read_text(encoding='utf-8'))
    weights = parameters['weights']
    if weights != blend_report['weights'] or not all(math.isfinite(v) and 0 <= v <= 1 for v in weights.values()) or not math.isclose(sum(weights.values()), 1.):
        raise ValueError('融合权重无效。')
    prep_path = Path(runs['als'][1]['source']['preparation_report'])
    prep = json.loads(prep_path.read_text(encoding='utf-8'))
    if prep['status'] != 'passed':
        raise ValueError('数据准备未通过。')
    raw = Path(cfg.paths.raw_dir).resolve()/'test_pairs.csv'
    if fingerprint(raw) not in prep['input_files']:
        raise ValueError('正式预测清单与数据准备记录不一致。')
    with raw.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        if next(reader) != ['userId', 'movieId', 'timestamp']:
            raise ValueError('正式预测清单表头不匹配。')
        raw_rows = list(reader)
    expected = json.loads((raw.parent/'summary.json').read_text(encoding='utf-8-sig'))['hidden_test_ratings']
    if len(raw_rows) != expected or any(len(r) != 3 for r in raw_rows):
        raise ValueError('正式预测清单行数或字段错误。')
    pairs_path = Path(cfg.paths.processed_dir).resolve()/'test_pairs'
    files = sorted(pairs_path.rglob('*.parquet'))
    if not files or set(files) != {Path(f['path']).resolve() for f in prep['output_files'] if Path(f['path']).resolve().is_relative_to(pairs_path)}:
        raise ValueError('正式预测 Parquet 清单不一致。')
    verify_files(prep, files)
    for name, (root, report) in runs.items():
        paths = [Path(f['path']).resolve() for f in report['output_files'] if Path(f['path']).resolve().is_relative_to(root/'models')]
        verify_files(report, paths)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    report = {'status':'running', 'generated_at':datetime.now(timezone.utc).isoformat(), 'weights':weights,
              'input_reports':inputs, 'test_pairs':fingerprint(raw), 'model_parameters':fingerprint(parameters_path),
              'configuration':OmegaConf.to_container(cfg, resolve=True),
              'training_scope':'original split=train only; no full-data refit', 'formal_rmse':None}
    spark = start_spark(cfg, 'OfficialRatingPrediction')
    try:
        log(f'Predicting {expected:,} official pairs with frozen ALS and MF')
        pairs = spark.read.parquet(str(pairs_path))
        als_root = runs['als'][0]/'models'
        als_parameters = json.loads((als_root/'parameters.json').read_text(encoding='utf-8'))
        fitted = ALSModel.load(str(als_root/'als'))
        saved = predict_als(pairs, fitted, spark.read.parquet(str(als_root/'user_means')),
                            als_parameters['training']['global_mean'], OmegaConf.create(als_parameters['model']))
        rows = saved.select('row_id','userId','movieId','timestamp','prediction','user_fallback','movie_fallback').orderBy('row_id').collect()
        if len(rows) != expected:
            raise ValueError('ALS 输出行数不一致。')
        for index, row in enumerate(rows):
            if row.row_id != index or [row.userId, row.movieId, row.timestamp] != list(map(int, raw_rows[index])):
                raise ValueError('Parquet 或模型预测与原清单顺序不一致。')
        als_scores = np.array([r.prediction for r in rows], np.float64)
        als_fallback = sum(r.user_fallback or r.movie_fallback for r in rows)
        fitted.userFactors.unpersist()
        fitted.itemFactors.unpersist()
    finally:
        spark.stop()
    try:
        torch.set_num_threads(4)
        mf_root = runs['mf'][0]/'models'
        mf_parameters = json.loads((mf_root/'parameters.json').read_text(encoding='utf-8'))['model']
        model, checkpoint = load_model(mf_root/'best.pt')
        mf_scores = np.empty(expected, np.float64)
        mf_fallback = 0
        with torch.no_grad():
            for start in range(0, expected, cfg.model.batch_size):
                end = min(start+cfg.model.batch_size, expected)
                users = torch.tensor([r.userId for r in rows[start:end]], dtype=torch.long)
                movies = torch.tensor([r.movieId for r in rows[start:end]], dtype=torch.long)
                prediction = model(users, movies)
                if mf_parameters['clip_predictions']:
                    prediction = prediction.clamp(mf_parameters['rating_min'], mf_parameters['rating_max'])
                mf_scores[start:end] = prediction.numpy()
                known_u, known_i = model.known(users, movies)
                mf_fallback += (~known_u | ~known_i).sum().item()
        prediction = weights['als']*als_scores + weights['mf']*mf_scores
        if not np.isfinite(als_scores).all() or not np.isfinite(mf_scores).all() or not np.isfinite(prediction).all() or np.any(prediction < cfg.model.rating_min) or np.any(prediction > cfg.model.rating_max):
            raise ValueError('正式预测包含非法数值或超出评分范围。')
        write_predictions(raw_rows, prediction, output/'predictions.csv')
        report.update(status='passed', rows=expected, prediction_min=float(prediction.min()),
                      prediction_max=float(prediction.max()), prediction_mean=float(prediction.mean()),
                      als_fallback_rows=int(als_fallback), mf_fallback_rows=int(mf_fallback), mf_epoch=checkpoint['epoch'],
                      output_files=[fingerprint(output/'predictions.csv')])
        return report
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = round(time.perf_counter()-started, 2)
        (output/'submission.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


@hydra.main(version_base='1.3', config_path='../configs', config_name='predict')
def main(cfg):
    output = Path(HydraConfig.get().runtime.output_dir)
    report = run_prediction(cfg, output)
    log(f"SUBMISSION_STATUS={report['status']}; rows={report['rows']}; file={output/'predictions.csv'}")


if __name__ == '__main__':
    main()
