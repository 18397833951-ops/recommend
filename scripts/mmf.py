"""MMF：高斯掩码、可学习行偏移；仅使用教师固定划分。"""

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
from torch import nn

from baselines import check_input
from biased_mf import BiasedMF, batch_tensors, evaluate_model, load_ratings
from bias import evaluate as evaluate_saved
from prepare_data import fingerprint, log, start_spark

REFERENCE = {
    'paper': 'https://doi.org/10.1145/3770855.3817710',
    'repository': 'https://github.com/snudatalab/MMF',
    'commit': 'c86307711752dff78fc7f8ade4ed4ffdc38c81c8',
    'source_sha256': 'dbaad8c88caa53b2f9cad5ace0af9dfa8ace8dc7f43201963b5caa74fc9ee4db',
}


class MaskedMF(BiasedMF):
    def __init__(self, users, movies, rank, components, global_mean, init_std):
        super().__init__(users, movies, rank, global_mean, init_std)
        self.components = components
        self.rank = rank
        self.user_shifts = nn.Embedding(users, components)
        self.movie_shifts = nn.Embedding(movies, components)
        nn.init.zeros_(self.user_shifts.weight)
        nn.init.zeros_(self.movie_shifts.weight)
        self.register_buffer('positions', torch.arange(rank, dtype=torch.float32))

    def mask(self, shifts, component):
        distance = (self.positions - self.rank * shifts[:, None] / 2) * ((component + 1) / self.components)
        return (0.5 + torch.exp(-distance.square() / 2) / 2.507) / self.components

    def forward(self, users, movies):
        p, q, bu, bi = self.parts(users, movies)
        us = self.user_shifts(users.clamp(0, len(self.known_users)-1))
        ms = self.movie_shifts(movies.clamp(0, len(self.known_movies)-1))
        interaction = torch.zeros_like(bu)
        for k in range(self.components):
            weight = self.mask(us[:, k], k) * self.mask(ms[:, k], k)
            interaction = interaction + (p * q * weight).sum(1)
        return self.global_mean + bu + bi + interaction

    def loss(self, users, movies, ratings, regularization):
        mse = (self(users, movies)-ratings).square().mean()
        if regularization:
            p, q, bu, bi = self.parts(users, movies)
            mse = mse + regularization * (p.square().sum(1)+q.square().sum(1)+bu.square()+bi.square()).mean()
        return mse


def validate_config(cfg):
    for key in ('rank', 'components', 'batch_size', 'max_epochs', 'patience', 'num_threads'):
        v = cfg.model[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f'{key} 必须为正整数。')
    if cfg.model.components < 2:
        raise ValueError('components 至少为 2。')
    for key in ('learning_rate', 'shift_learning_rate', 'init_std'):
        v = cfg.model[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'{key} 必须为有限正数。')
    if not math.isfinite(cfg.model.regularization) or cfg.model.regularization < 0:
        raise ValueError('regularization 必须为有限非负数。')
    if not all(math.isfinite(v) for v in (cfg.model.rating_min,cfg.model.rating_max)) or cfg.model.rating_min >= cfg.model.rating_max:
        raise ValueError('评分范围无效。')
    if list(cfg.evaluation.splits) != ['validation','test']:
        raise ValueError('使用固定 validation、test 评测。')
    if isinstance(cfg.data.read_partitions,bool) or not isinstance(cfg.data.read_partitions,int) or cfg.data.read_partitions < 1:
        raise ValueError('read_partitions 必须为正整数。')
    device = torch.device(cfg.model.device)
    if device.type not in ('cpu','cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('请选择可用的 cpu 或 cuda。')


def load_model(path, device='cpu'):
    c = torch.load(path,map_location=device,weights_only=True)
    model = MaskedMF(c['num_users'],c['num_movies'],c['rank'],c['components'],c['global_mean'],.02).to(device)
    model.load_state_dict(c['state_dict'])
    return model.eval(), c


def fit_model(train, validation, cfg, path, report, on_epoch=None):
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.num_threads)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(cfg.seed)
    mean = float(train.ratings.mean(dtype=np.float64))
    model = MaskedMF(int(train.users.max())+1,int(train.movies.max())+1,cfg.rank,cfg.components,mean,cfg.init_std)
    model.known_users[torch.from_numpy(train.users.astype(np.int64))] = True
    model.known_movies[torch.from_numpy(train.movies.astype(np.int64))] = True
    report['training'] = {'rows':len(train),'global_mean':mean,'users':int(model.known_users.sum()),
                          'movies':int(model.known_movies.sum()),'parameters':sum(p.numel() for p in model.parameters())}
    model.to(cfg.device)
    shifts = [model.user_shifts.weight,model.movie_shifts.weight]
    factors = [model.user_factors.weight,model.movie_factors.weight,model.user_bias.weight,model.movie_bias.weight]
    optimizer = torch.optim.Adam([{'params':factors,'lr':cfg.learning_rate},
                                  {'params':shifts,'lr':cfg.shift_learning_rate}])
    path.parent.mkdir(parents=True,exist_ok=True)
    report['history'], report['best_epoch'] = [], 0
    best, stale = float('inf'), 0
    for epoch in range(1,cfg.max_epochs+1):
        started = time.perf_counter()
        model.train()
        order = rng.permutation(len(train))
        total = 0.
        for start in range(0,len(train),cfg.batch_size):
            indices = order[start:start+cfg.batch_size]
            u,i,y = batch_tensors(train,indices,cfg.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(u,i,y,cfg.regularization)
            if not torch.isfinite(loss):
                raise ValueError('训练损失包含 NaN 或无穷值。')
            loss.backward()
            optimizer.step()
            total += loss.item()*len(indices)
        del order
        score = evaluate_model(model,validation,cfg)
        report['history'].append({'epoch':epoch,'online_training_loss':total/len(train),'validation':score,
                                   'elapsed_seconds':round(time.perf_counter()-started,2)})
        if score['rmse'] < best:
            best,stale = score['rmse'],0
            report['best_epoch'] = epoch
            torch.save({'state_dict':model.state_dict(),'num_users':len(model.known_users),
                        'num_movies':len(model.known_movies),'rank':cfg.rank,'components':cfg.components,
                        'global_mean':mean,'epoch':epoch,'validation_rmse':best},path)
        else:
            stale += 1
        log(f"MMF epoch={epoch}; validation_RMSE={score['rmse']:.8f}; best_epoch={report['best_epoch']}")
        if on_epoch:
            on_epoch()
        if stale >= cfg.patience:
            break
    report['epochs_trained'] = len(report['history'])
    return load_model(path,cfg.device)[0]


def write_report(report, output):
    (output/'mmf.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# MMF 高斯掩码矩阵分解','',f"状态：{report['status']}。",'',
             '共享用户、电影基础向量，各自学习每个成分的行偏移；按官方高斯函数生成掩码并求和。',
             '掩码参数：delta=(r-rank/2*shift)*(k+1)/K；mask=(0.5+exp(-delta²/2)/2.507)/K。',
             '预测=训练全局均值+用户偏置+电影偏置+sum_k dot(p*mask_user_k,q*mask_movie_k)。',
             '原训练集学习向量、偏置和偏移，原验证集选择最佳轮次，恢复后才读取本地测试。',
             '与官方默认示例的区别：启用 learned shifts，固定均值取教师训练集均值而非示例常数 3.0；使用本项目冷启动规则。',
             '未知 ID 的向量和偏置取零，已知侧偏置保留；默认仅使用 MSE，不增加正则或 weight decay。','',
             f"最佳轮次：{report.get('best_epoch','待定')}；实际轮次：{report.get('epochs_trained',len(report.get('history',[])))}。",'',
             '| 划分 | 记录数 | RMSE | 回退记录 |','| --- | ---: | ---: | ---: |']
    for r in report.get('metrics',[]):
        label = '验证集' if r['split']=='validation' else '本地测试集'
        lines.append(f"| {label} | {r['rows']:,} | {r['rmse']:.8f} | {r['fallback_rows']:,} |")
    lines += ['', '## 逐轮训练', '', '| 轮次 | 在线训练损失 | 验证 RMSE | 耗时（秒） |',
              '| ---: | ---: | ---: | ---: |']
    for row in report.get('history',[]):
        lines.append(f"| {row['epoch']} | {row['online_training_loss']:.8f} | {row['validation']['rmse']:.8f} | {row['elapsed_seconds']:.2f} |")
    if 'training' in report:
        lines += ['', f"训练记录：{report['training']['rows']:,}；可训练参数（含 ID 索引空位）：{report['training']['parameters']:,}。"]
    lines += ['',f"耗时：{report.get('elapsed_seconds',0):.2f} 秒。",'',
              f"本次是 rank={report['configuration']['model']['rank']}、K={report['configuration']['model']['components']} 的单组工程实验，不是按官方示例 R=1200、K=64 的完整基准复现；精确配置见 JSON。",
              '[论文](https://doi.org/10.1145/3770855.3817710)；[官方源码固定版本](https://github.com/snudatalab/MMF/blob/'+REFERENCE['commit']+'/main_completion.py)。']
    if 'error' in report:
        lines.append('失败原因：'+report['error'])
    (output/'mmf.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_mmf(cfg, output):
    from pyspark.sql import functions as F
    validate_config(cfg)
    output = Path(output).resolve()
    for source_path in (cfg.paths.raw_dir,cfg.paths.splits_dir):
        source_path = Path(source_path).expanduser().resolve()
        if output.is_relative_to(source_path) or source_path.is_relative_to(output):
            raise ValueError('输出与输入目录不可重叠。')
    if any((output/name).exists() for name in ('models','predictions','mmf.json')):
        raise FileExistsError('已有结果，不会覆盖。')
    started = time.perf_counter()
    root,source = check_input(cfg)
    expected = {p:sum(r['count'] for r in source['counts'] if r['split']==p) for p in ('train','validation','test')}
    output.mkdir(parents=True,exist_ok=True)
    report = {'status':'running','generated_at':datetime.now(timezone.utc).isoformat(),'source':source,
              'reference':REFERENCE,'configuration':OmegaConf.to_container(cfg,resolve=True),'torch_version':str(torch.__version__)}
    try:
        write_report(report,output)
        spark = start_spark(cfg,'MMFTrainingInput')
        try:
            train = load_ratings(spark,root,'train',expected['train'],cfg.data.read_partitions)
            validation = load_ratings(spark,root,'validation',expected['validation'],cfg.data.read_partitions)
        finally:
            spark.stop()
        path = output/'models/best.pt'
        model = fit_model(train,validation,cfg.model,path,report,lambda:write_report(report,output))
        del train
        folder = output/'predictions'
        folder.mkdir()
        vscore = evaluate_model(model,validation,cfg.model,folder/'validation.csv.gz')
        if not math.isclose(vscore['rmse'],report['history'][report['best_epoch']-1]['validation']['rmse'],abs_tol=1e-12,rel_tol=0):
            raise ValueError('最佳模型读回误差不一致。')
        del validation
        log('MMF best weights restored; reading local test')
        spark = start_spark(cfg,'MMFFinalEvaluation')
        try:
            test = load_ratings(spark,root,'test',expected['test'],cfg.data.read_partitions)
            tscore = evaluate_model(model,test,cfg.model,folder/'test.csv.gz')
            del test
            report['metrics'] = []
            schema = 'userId int,movieId int,rating double,timestamp long,raw_prediction double,prediction double,user_fallback int,movie_fallback int'
            for phase,original in (('validation',vscore),('test',tscore)):
                saved = spark.read.schema(schema).option('header',True).option('mode','FAILFAST').csv(str(folder/f'{phase}.csv.gz'))
                for flag in ('user_fallback','movie_fallback'):
                    saved = saved.withColumn(flag,F.col(flag).cast('boolean'))
                score = evaluate_saved(saved)
                if score['rows'] != expected[phase] or not math.isclose(score['rmse'],original['rmse'],abs_tol=1e-10,rel_tol=0):
                    raise ValueError('预测读回数量或误差不一致。')
                report['metrics'].append({'split':phase,**score})
        finally:
            spark.stop()
        (output/'models/parameters.json').write_text(json.dumps({'training':report['training'],'best_epoch':report['best_epoch'],
            'model':OmegaConf.to_container(cfg.model,resolve=True),'reference':REFERENCE},ensure_ascii=False,indent=2),encoding='utf-8')
        report['output_files'] = [fingerprint(p) for p in [path,output/'models/parameters.json',*sorted(folder.glob('*.csv.gz'))]]
        report['status'] = 'passed'
        return report
    except Exception as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = round(time.perf_counter()-started,2)
        write_report(report,output)


@hydra.main(version_base='1.3',config_path='../configs',config_name='mmf')
def main(cfg):
    output = Path(HydraConfig.get().runtime.output_dir)
    report = run_mmf(cfg,output)
    log(f"MMF_STATUS={report['status']}; report={output/'mmf.md'}")


if __name__=='__main__':
    main()
