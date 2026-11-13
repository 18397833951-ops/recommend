"""从项目真实 JSON 报告生成离线 HTML 汇报，保留完整报告及原始数值。"""
import csv
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    'data':'artifacts/data_prepare/2026-11-09/09-12-34-286216/data_preparation.json',
    'means':'artifacts/baselines/2026-11-09/09-36-55-105832/baselines.json',
    'bias':'artifacts/bias/2026-11-09/10-38-19-095645/bias.json',
    'als':'artifacts/als/2026-11-09/12-09-14-872852/als.json',
    'mf':'artifacts/biased_mf/2026-11-09/15-07-48-034939/biased_mf.json',
    'itemcf':'artifacts/itemcf/2026-11-09/16-21-09-845229/itemcf.json',
    'blend':'artifacts/blend/2026-11-12/14-46-10-662025/blend.json',
    'gate':'artifacts/adaptive_blend/2026-11-12/15-31-50-871545/adaptive_blend.json',
    'submission':'artifacts/submission/2026-11-13/09-44-46-741346/submission.json',
}
R = {key:json.loads((ROOT/path).read_text(encoding='utf-8')) for key,path in PATHS.items()}
NAMES = {'data':'数据检查与划分','means':'三种均值基线','bias':'正则化偏置','als':'显式 ALS','mf':'带偏置矩阵分解','itemcf':'近似 ItemCF','blend':'固定权重融合','gate':'自适应融合网络','submission':'正式预测导出'}
esc = lambda x: html.escape(str(x), quote=True)
def value(x):
    if isinstance(x, float):
        return f'<span class="numeric" title="报告原始值：{x!r}">{x:.8f}</span>'
    return esc(f'{x:,}' if isinstance(x,int) and not isinstance(x,bool) else x)
def table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{esc(x)}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def metric(key, model=None, extra=()):
    rows=[r for r in R[key]['metrics'] if model is None or r.get('model')==model]
    cols=[('记录数','rows'),('平方误差和','squared_error_sum'),('RMSE','rmse')]
    if any('fallback_rows' in r for r in rows): cols += [('回退记录','fallback_rows')]
    cols += list(extra)
    return table(['划分']+[c[0] for c in cols], [[('验证集' if r['split']=='validation' else '本地测试集')]+[value(r[c[1]]) if c[1] in r else '—' for c in cols] for r in rows])
def params(data, labels=None):
    labels=labels or {}
    return '<dl class="params">'+''.join(f'<div><dt>{esc(labels.get(k,k))}</dt><dd>{value(v)}</dd></div>' for k,v in data.items())+'</dl>'
def text(s): return f'<p class="body-text">{s}</p>'
def call(s): return f'<div class="callout">{s}</div>'
def cols(a,b): return f'<div class="columns"><div>{a}</div><div>{b}</div></div>'
def formula(s): return f'<div class="formula">{s}</div>'
def chart(points, labels):
    lo=min(points)-.002; hi=max(points)+.002
    coords=[(65+i*930/max(1,len(points)-1),260-(v-lo)/(hi-lo)*210) for i,v in enumerate(points)]
    out='<svg class="chart" viewBox="0 0 1080 330" role="img" aria-label="验证 RMSE 曲线">'
    for t in range(4):
        y=50+t*70; val=hi-(hi-lo)*t/3
        out+=f'<line x1="65" x2="1030" y1="{y}" y2="{y}" stroke="#cbd4d3"/><text x="3" y="{y+5}" class="axis">{val:.4f}</text>'
    out+='<polyline fill="none" stroke="#087f78" stroke-width="4" points="'+' '.join(f'{x},{y}' for x,y in coords)+'"/>'
    for (x,y),v,label in zip(coords,points,labels):
        out+=f'<circle cx="{x}" cy="{y}" r="7" fill="#087f78"/><text x="{x}" y="{y-19}" text-anchor="middle" class="plot-value">{v:.8f}</text><text x="{x}" y="304" text-anchor="middle" class="axis">{esc(label)}</text>'
    return out+'</svg>'

slides=[]
def slide(chapter,title,sub,body,source,notes='',kind=''):
    slides.append({'chapter':chapter,'title':title,'sub':sub,'body':body,'source':source,'notes':notes or sub,'kind':kind})

blend_v=next(r for r in R['blend']['metrics'] if r['model']=='blend' and r['split']=='validation')
blend_t=next(r for r in R['blend']['metrics'] if r['model']=='blend' and r['split']=='test')
slide('项目汇报','电影评分预测','基于 Spark ALS、带偏置矩阵分解与固定权重融合',
      '<div class="cover-title">用真实实验<br>选择评分模型<span class="dot">。</span></div><div class="cover-rule"></div><div class="cover-bottom"><div><span class="caption">本地测试 RMSE</span><strong>'+f'{blend_t["rmse"]:.8f}'+'</strong></div><div><span class="caption">正式预测记录</span><strong>100,000</strong></div><div><span class="caption">本地划分</span><strong>8 : 1 : 1</strong></div></div>',
      'blend','页面中的成绩是独立本地测试结果。正式 10 万条测试由教师保留答案，当前只能交付预测，不能报告正式 RMSE。','cover')
slide('任务','任务与最终交付','对 test_pairs 的每一条用户电影组合输出连续评分',
      cols('<div class="big">31,900,204</div><div class="caption">教师提供的有标签评分</div>'+text('约一半来自 MovieLens，一半来自合成数据。用户、电影 ID 统一使用教师提供的版本。'),
           '<div class="big">100,000</div><div class="caption">指定的待预测记录</div>'+text('保留 userId、movieId、timestamp，新增 prediction，保持原始行顺序。'))+formula('RMSE = √[ Σ(r̂ − r)² / N ]')+call('评价目标是评分回归。真实评分未公布，正式 RMSE 由教师计算。'), 'submission')
d=R['data']; summary=d['summary']; counts={p:sum(r['count'] for r in summary['by_source_and_split'] if r['split']==p) for p in ('train','validation','test')}
slide('数据','数据检查与类型整理','原始输入先检查，再保存 Parquet 与文件指纹',
      '<div class="flow"><div><b>01</b><h3>输入与 ID</h3><p>字段、SHA256<br>教师电影映射</p></div><div><b>02</b><h3>数据内容</h3><p>缺失、重复组合<br>评分范围与引用</p></div><div><b>03</b><h3>整理保存</h3><p>明确字段类型<br>Parquet 分区</p></div><div><b>04</b><h3>正式清单</h3><p>保留 row_id<br>导出恢复原顺序</p></div></div>'+params({'实际评分用户数':summary['users'],'元数据电影数':175170,'数据准备耗时 / 秒':d['elapsed_seconds'],'检查状态':d['status']})+call('评分范围 0.5～5.0。来源字段用于核对，模型统一训练并评估整体 RMSE。'), 'data')
slide('数据','按用户时间划分','同秒评分保持为完整组，边界取最接近累计 80% 与 90% 的时间组',
      '<div class="split-band"><i>训练 80%</i><i>验证 10%</i><i>测试 10%</i></div>'+table(['划分','完整记录数','实际占比','用途'],[[name,value(counts[p]),f'{counts[p]/sum(counts.values())*100:.8f}%',purpose] for name,p,purpose in [('训练集','train','拟合统计、向量、偏置与融合网络'),('验证集','validation','选择超参数、轮次和固定融合权重'),('本地测试集','test','评价已经确定的方案')]])+params({'无验证评分的用户':summary['users_without_validation'],'无本地测试评分的用户':summary['users_without_local_test']})+call('至少保留第一个时间组用于训练。实际比例略有偏差，所有算法复用同一份划分。'), 'data')
slide('数据','冷启动记录完整保留','训练未见过的电影不会从评测中删除',
      table(['划分','训练未见电影记录','涉及电影数'],[['验证集',value(summary['validation_cold_movie_rows']),value(summary['validation_cold_movies'])],['本地测试集',value(summary['local_test_cold_movie_rows']),value(summary['local_test_cold_movies'])]])+
      cols(text('均值模型：缺少对应统计时使用训练全局均值。<br>偏置模型：未知一侧偏置为零，保留已知一侧。'),text('ALS：优先回退训练用户均值，再回退全局均值。<br>矩阵分解：未知一侧向量及偏置为零。'))+call('所有预测先完成既定回退，再计入同一份整体 RMSE。'), 'data')
slide('实现','Spark 与 PyTorch 的职责','Spark local[2] 执行本地并行任务，PyTorch 模型在单机 CPU 训练',
      '<div class="flow"><div><b>DATA</b><h3>Parquet</h3><p>类型转换<br>时间划分</p></div><div><b>SPARK</b><h3>聚合与 ALS</h3><p>评分统计<br>相似度计算</p></div><div><b>TORCH</b><h3>矩阵分解</h3><p>向量与偏置<br>Adam 联合训练</p></div><div><b>OUTPUT</b><h3>融合与导出</h3><p>冻结权重<br>完整行数核对</p></div></div>'+params({'Python':'3.11','Spark':'3.5.7','Java':'JDK 17','运行方式':'local[2]','公共 Java 堆':'1g','ALS / ItemCF Java 堆':'4g','内部压缩':'LZF','Parquet 压缩':'GZIP'})+call('当前实现运行在一台电脑上，未部署多机集群。Hydra 保存配置，SHA256 记录输入输出。'), 'als')
slide('基线','全局均值','给全部记录预测训练集平均分',
      formula('r̂ᵤᵢ = μ')+params({'训练评分数':R['means']['training']['rows'],'μ / 报告原始值':str(R['means']['training']['global_mean'])})+metric('means','global_mean')+text('提供最基础的评分误差参照，不需要训练向量。'), 'means')
slide('基线','用户均值与电影均值','分别利用用户打分习惯和电影总体评分',
      '<h3>用户均值</h3>'+metric('means','user_mean')+'<h3>电影均值</h3>'+metric('means','movie_mean')+call('统计只来自训练评分。未知用户或电影回退全局均值，不丢弃记录。'), 'means')
b=R['bias']; slide('基线','正则化偏置模型','在全局均值上叠加用户偏置与电影偏置',
      formula('r̂ᵤᵢ = μ + bᵤ + bᵢ')+params({'正则候选':str(b['configuration']['model']['regularizations']),'每组最大轮数':b['configuration']['model']['max_epochs'],'选中正则强度':b['selected']['regularization'],'选中轮次':b['selected']['epoch']})+metric('bias')+call('固定 μ，交替更新两类偏置。完整 30 个验证试验保留在本页“完整报告”的 trials 中。'), 'bias')
a=R['als']; slide('模型','显式 ALS','用户与电影各有一组潜在向量，点积得到预测评分',
      formula('r̂ᵤᵢ = pᵤᵀ qᵢ')+params({k:a['configuration']['model'][k] for k in ('rank','reg_param','max_iter','seed','num_user_blocks','num_item_blocks','block_size')})+metric('als')+text('交替固定电影向量求用户向量、固定用户向量求电影向量。显式评分模式 implicitPrefs=false，最终评分截断到 0.5～5.0。'), 'als')
m=R['mf'];mc=m['configuration']['model']; slide('模型','联合训练的带偏置矩阵分解','四类参数一起学习，全局均值固定为训练平均分',
      formula('r̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤᵀ qᵢ')+params({k:mc[k] for k in ('rank','learning_rate','regularization','batch_size','max_epochs','patience','seed','num_threads')})+metric('mf')+call('优化器 Adam。训练目标为每批 MSE 加参数正则项；不额外叠加 weight decay。'), 'mf')
slide('模型','矩阵分解的逐轮验证','训练了 5 轮，恢复验证误差最低的第 2 轮',
      chart([r['validation']['rmse'] for r in m['history']],[f'第 {r["epoch"]} 轮' for r in m['history']])+table(['轮次','在线训练损失','验证 RMSE','验证平方误差和','本轮耗时 / s'],[[value(r['epoch']),value(r['online_training_loss']),value(r['validation']['rmse']),value(r['validation']['squared_error_sum']),value(r['elapsed_seconds'])] for r in m['history']]), 'mf','在线训练损失是各批更新前损失的加权均值，包含正则项。它与纯 RMSE 不是同一个指标。完整报告保留逐轮统计。')
i=R['itemcf']; slide('模型','近似评分版 ItemCF','相似电影的评分差值经加权后校正目标电影均值',
      formula('r̂ᵤᵢ = 电影均值 + Σ[相似度 × 邻居评分差] / Σ相似度')+params({k:i['configuration']['model'][k] for k in ('sampling_threshold','neighbors','min_neighbors','min_item_ratings','min_similarity')})+metric('itemcf',extra=(('平均匹配邻居','mean_actual_neighbors'),))+call('相似度采用用户均值中心化后的电影列余弦；通过 Spark DIMSUM 近似计算。'), 'itemcf')
slide('模型','ItemCF 的图规模与回退','本组邻域配置未超过电影均值，不用于最终提交',
      params(i['graph'])+table(['划分','回退条数','回退比例','平均匹配邻居'],[[('验证集' if r['split']=='validation' else '本地测试集'),value(r['fallback_rows']),f'{r["fallback_rows"]/r["rows"]*100:.8f}%',value(r['mean_actual_neighbors'])] for r in i['metrics']])+call('回退数量包含邻居不足的记录；这些记录仍完整参与评测。'), 'itemcf')
comparisons=[]
for name,key,model in [('全局均值','means','global_mean'),('用户均值','means','user_mean'),('电影均值','means','movie_mean'),('偏置模型','bias',None),('显式 ALS','als',None),('带偏置矩阵分解','mf',None),('近似 ItemCF','itemcf',None),('固定权重融合','blend','blend')]:
    rows=[r for r in R[key]['metrics'] if model is None or r.get('model')==model]
    comparisons.append([name]+[value(next(r['rmse'] for r in rows if r['split']==p)) for p in ('validation','test')])
slide('结果','所有模型使用同一份评测划分','统一保留全部记录，报告数值保留 8 位小数',table(['评分方法','验证 RMSE','本地测试 RMSE'],comparisons)+text('完整浮点原值可悬停查看；每页的完整 JSON 报告保留全部原始精度。'), 'blend')
weights=R['blend']['weights'];slide('最终方案','固定权重融合','两个模型最终分数的凸组合，权重仅由验证集确定',
      '<div class="weight-bar"><div>ALS <b>'+f'{weights["als"]*100:.8f}%</b></div><div>矩阵分解 <b>'+f'{weights["mf"]*100:.8f}%</b></div></div>'+formula('r̂ = α · ALS + (1 − α) · MF')+params({'α / 报告原始值':str(weights['als']),'1−α / 报告原始值':str(weights['mf'])})+metric('blend','blend')+call('α = −Σ(e·d) / Σ(d²)，限制在 [0,1]。d = ALS−MF，e = MF−真实评分。'), 'blend')
g=R['gate'];gc=g['configuration']['model'];slide('扩展实验','自适应权重融合网络','从固定的一个比例，扩展为逐条记录输出不同权重',
      '<div class="network"><div><strong>4</strong><h3>输入特征</h3><p>ALS 分数、MF 分数<br>log1p 用户训练评分数<br>log1p 电影训练评分数</p></div><span>›</span><div><strong>16</strong><h3>隐藏神经元</h3><p>Linear + ReLU<br>训练集标准化<br>共 97 个可训练参数</p></div><span>›</span><div><strong>1</strong><h3>融合权重</h3><p>Sigmoid 输出 α<br>α·ALS + (1−α)·MF</p></div></div>'+params({k:gc[k] for k in ('learning_rate','batch_size','max_epochs','patience','seed','num_threads')})+call('基础模型保持固定。原训练集训练融合网络，原验证集选轮次，原测试集最后评价。'), 'gate')
slide('扩展实验','网络训练过程完整记录','训练误差下降，但验证误差持续上升',
      chart([r['validation']['rmse'] for r in g['history']],[f'第 {r["epoch"]} 轮' for r in g['history']])+table(['轮次','训练 MSE','验证 RMSE','ALS 权重最小值','ALS 权重均值','ALS 权重最大值'],[[value(r['epoch']),value(r['training_mse']),value(r['validation']['rmse'])]+[value(r['validation']['als_weight'][k]) for k in ('min','mean','max')] for r in g['history']])+call('固定融合验证 RMSE：'+f'{blend_v["rmse"]:.8f}'+'。三轮训练均未超过，继续采用固定融合。'), 'gate')
slide('扩展实验','初始化参照与训练后结果分开报告','选中的第 0 轮是未训练的 50/50 权重，不是学出的自适应模型',
      metric('gate')+params({'实际训练轮次':g['epochs_trained'],'最佳检查点轮次':g['best_epoch'],'验证集选定方案':g['selected_method'],'完整运行耗时 / s':g['elapsed_seconds']})+call('上表对应恢复后的 50/50 初始化。训练后各轮的真实验证成绩见上一页，不能把此表当作训练网络的改进成绩。'), 'gate')
slide('工程记录','各阶段完整运行耗时','含各脚本自身的读取、训练、保存与核对，不等同于纯优化器时间',
      table(['阶段','运行耗时 / 秒','记录说明'],[[NAMES[k],value(R[k]['elapsed_seconds']),note] for k,note in [('data','CSV 检查、Parquet 与划分'),('means','三种均值模型'),('bias','3 组正则，每组 10 轮'),('als','当前单组重跑，非历史 8 组网格'),('mf','实际训练 5 轮，最佳第 2 轮'),('itemcf','近似相似图与全部评分'),('blend','复用已有预测，不重新训练'),('gate','训练输入构造、3 轮网络与评测'),('submission','加载既有模型，生成正式预测')]]), 'submission')
s=R['submission'];csv_path=ROOT/'artifacts/submission/2026-11-13/09-44-46-741346/predictions.csv'
with csv_path.open(encoding='utf-8-sig',newline='') as stream:
    reader=csv.reader(stream);head=next(reader);preview=[next(reader) for _ in range(3)]
slide('正式交付','完整的 100,000 条正式预测','直接使用已经选定的原训练子集模型，不做完整教师评分重训',
      params({k:s[k] for k in ('rows','prediction_min','prediction_max','prediction_mean','als_fallback_rows','mf_fallback_rows','mf_epoch','elapsed_seconds')})+table(head,[[esc(x) for x in row] for row in preview])+call('表中为实际输出的前三行。正式清单中的顺序与原字段保持不变，730 条回退记录完整保留。'), 'submission')
slide('正式交付','输出文件与完整性记录','保留连续精度，正式测试 RMSE 尚未知',
      params({'提交文件':'predictions.csv','字段':'userId, movieId, timestamp, prediction','文件字节数':s['output_files'][0]['bytes'],'SHA256':s['output_files'][0]['sha256'],'检查状态':s['status'],'正式 RMSE':'隐藏答案未提供，由教师计算'})+
      '<div class="check-row"><span>原行顺序</span><span>字段逐行对应</span><span>100% 预测覆盖</span><span>有限数值与评分范围</span></div>'+text('提交列名与文件名如有教师统一模板，可按最终模板调整。'), 'submission')
slide('复现','训练与正式预测的入口','Hydra 参数快照与输入输出指纹对应每一次实验',
      cols('<pre class="code">python scripts/prepare_data.py\npython scripts/baselines.py\npython scripts/bias.py\npython scripts/als.py\npython scripts/biased_mf.py\npython scripts/itemcf.py</pre>', '<pre class="code">python scripts/blend.py\npython scripts/adaptive_blend.py\npython scripts/predict.py\n\n# 查看实际配置\npython scripts/predict.py --cfg job --resolve</pre>')+call('已有结果不会覆盖。每次运行保存到日期时间目录，模型文件和正式预测分别保留。'), 'submission')
slide('结论','保留验证收益明确的模型','固定融合仍是最终提交方案',
      '<div class="closing">验证选择模型<br>独立测试评价<br><em>按原清单交付评分</em></div>'+params({'最终验证 RMSE':blend_v['rmse'],'独立本地测试 RMSE':blend_t['rmse'],'正式预测记录数':s['rows'],'正式测试成绩':'待教师计算'})+text('已完成均值、偏置、ALS、矩阵分解、ItemCF 和两种融合实验。当前为单机运行，未做完整教师评分重训。'), 'blend')
slide('数据档案','完整报告随汇报一并保留','原始 JSON 全字段嵌入本文件，可离线查看、复制与下载',
      '<div class="report-grid">'+''.join(f'<button onclick="openReport(\'{k}\')"><b>{esc(NAMES[k])}</b><span>{esc(Path(p).name)} ↗</span></button>' for k,p in PATHS.items())+'</div>'+text('包含原始浮点精度、训练配置、每轮历史、试验列表、回退统计与全部文件指纹。正文按整体指标汇报，不按数据来源拆分成绩。'), 'blend')

CSS = r'''
@import none;
:root{--bg:#f4f5f0;--ink:#142927;--dim:#6b7b77;--line:#d1dad3;--green:#087f78;--accent:#bcf9dc;--rail:102px}*{box-sizing:border-box}body{margin:0;background:var(--bg);font-family:'Segoe UI','Microsoft YaHei',sans-serif;color:var(--ink);overflow:hidden}button{font:inherit;cursor:pointer}button:focus-visible{outline:3px solid #14b8a6;outline-offset:3px}.rail{position:fixed;inset:0 auto 0 0;width:var(--rail);background:#102d2a;color:#b8d2c9;display:flex;align-items:center;flex-direction:column;padding:30px 0;z-index:3}.mark{font-size:24px;font-weight:800;letter-spacing:-2px;color:var(--accent)}.vertical{writing-mode:vertical-rl;letter-spacing:3px;font-size:10px;margin-top:24px;height:150px;max-height:150px;overflow:hidden;flex-shrink:0}.rail nav{margin:auto 0;display:flex;flex-direction:column;gap:6px}.rail nav button{width:22px;height:5px;border:0;background:#54756c;border-radius:5px;padding:0}.rail nav button.on{background:var(--accent);width:34px}.rail .edition{font:11px Consolas,monospace;line-height:1.8;text-align:center}.top{position:fixed;left:150px;right:46px;top:26px;display:flex;justify-content:space-between;font-size:11px;letter-spacing:2px;color:var(--dim);z-index:2}main{position:fixed;inset:70px 0 78px var(--rail)}.slide{display:none;height:100%;padding:22px 64px 16px 48px;overflow:auto}.slide.active{display:flex;flex-direction:column;animation:enter .25s ease}@keyframes enter{from{opacity:.3;transform:translateY(8px)}to{opacity:1;transform:none}}.heading{margin-bottom:28px}.kicker{font-size:13px;color:var(--green);font-weight:700;letter-spacing:3px;margin-bottom:14px}h1{font-size:clamp(32px,3.3vw,56px);line-height:1.2;letter-spacing:-1.8px;margin:0 0 14px;font-weight:700}h3{font-size:20px;margin:20px 0 14px}.subtitle{font-size:18px;line-height:1.6;color:var(--dim)}.content{flex:1;min-height:0}.body-text{font-size:20px;line-height:1.8;margin:18px 0;color:#51655f}.columns{display:grid;grid-template-columns:1fr 1fr;gap:48px}.big{font:clamp(36px,5vw,76px) Consolas,monospace;letter-spacing:-3px;color:var(--green);margin:30px 0 12px}.caption{font-size:14px;letter-spacing:1px;color:var(--dim)}.params{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px 28px;margin:24px 0 30px}.params>div{border-top:1px solid var(--line);padding-top:12px}.params dt{font-size:12px;color:var(--dim);margin-bottom:9px;overflow-wrap:anywhere}.params dd{margin:0;font:18px Consolas,'Microsoft YaHei',monospace;line-height:1.5;overflow-wrap:anywhere}.table-wrap{overflow-x:auto;margin:18px 0 26px}table{width:100%;border-collapse:collapse;font-size:17px}th{padding:12px 16px;border-bottom:2px solid var(--ink);font-size:12px;letter-spacing:.5px;color:var(--dim);text-align:left;white-space:nowrap}td{padding:14px 16px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;white-space:nowrap}td:not(:first-child){font-family:Consolas,'Microsoft YaHei',monospace}tr:last-child td{background:#e7eee7}tbody tr:hover{background:#e1eee7}.numeric{font-family:Consolas,monospace}.callout{margin-top:22px;border-left:4px solid var(--green);padding:14px 20px;background:#e6eee5;font-size:18px;line-height:1.8}.formula{font-family:Georgia,'Microsoft YaHei',serif;font-size:clamp(24px,2.6vw,42px);padding:22px 0;line-height:1.6;color:var(--green)}.flow{display:grid;grid-template-columns:repeat(4,1fr);margin:30px 0;gap:30px}.flow>div{border-top:4px solid var(--green);padding-top:20px}.flow b{font:14px Consolas,monospace;color:var(--green);letter-spacing:3px}.flow h3{font-size:27px;margin:25px 0 16px}.flow p{line-height:1.8;color:var(--dim);font-size:18px}.split-band{display:flex;height:70px;margin:24px 0 34px;overflow:hidden}.split-band i{font-style:normal;display:grid;place-items:center;background:var(--green);color:white;font-size:20px;flex:8}.split-band i:nth-child(2){flex:1;background:#bceacb;color:var(--ink)}.split-band i:nth-child(3){flex:1;background:#dbe3d9;color:var(--ink)}.weight-bar{display:flex;margin:26px 0 12px;height:96px}.weight-bar>div{display:flex;justify-content:center;align-items:center;gap:25px;font-size:20px;background:#087f78;color:white;width:41.52908341%}.weight-bar>div+div{background:#bceacb;color:var(--ink);width:58.47091659%}.weight-bar b{font:25px Consolas,monospace}.chart{display:block;width:100%;height:clamp(200px,26vh,320px);margin:10px 0 20px;overflow:visible}.axis{font:12px Consolas,'Microsoft YaHei',monospace;fill:#75827c}.plot-value{font:16px Consolas,monospace;fill:#142927}.network{display:flex;align-items:center;gap:30px;margin:30px 0}.network>div{flex:1;border-top:1px solid var(--line);padding-top:20px}.network strong{font:72px Consolas,monospace;line-height:1;color:var(--green)}.network p{font-size:17px;color:var(--dim);line-height:1.8}.network>span{font-size:40px;color:var(--green)}.code{padding:26px;background:#102d2a;color:#c8efdb;font:16px/2 Consolas,monospace;border-radius:2px;white-space:pre-wrap;overflow-wrap:anywhere}.check-row{display:flex;gap:20px;margin:30px 0;flex-wrap:wrap}.check-row span{border:1px solid var(--green);padding:12px 18px;color:var(--green)}.closing{font-size:clamp(36px,4vw,64px);font-weight:600;line-height:1.35;letter-spacing:-2px;margin:15px 0 28px}.closing em{font-style:normal;color:var(--green)}.report-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}.report-grid button{border:1px solid var(--line);background:transparent;text-align:left;padding:25px}.report-grid button:hover{background:#dbeee0}.report-grid b{font-size:19px;display:block;margin-bottom:14px}.report-grid span{font:13px Consolas,monospace;color:var(--green)}
.slide.cover .heading{display:none}.slide.cover{background:#102d2a;color:#f1f5ee;position:relative;padding-top:42px}.slide.cover:after{content:'';width:490px;height:490px;border:1px solid #7db19a44;border-radius:50%;position:absolute;right:-65px;top:-140px;box-shadow:0 0 0 50px #84b4a010,0 0 0 100px #84b4a00c;pointer-events:none}.cover-title{font-size:clamp(45px,6vw,94px);letter-spacing:-4px;font-weight:650;line-height:1.18;margin:25px 0 34px;position:relative;z-index:1}.dot{color:var(--accent)}.cover-rule{width:90px;height:5px;background:var(--accent);margin:26px 0 35px}.cover-bottom{display:flex;gap:65px;border-top:1px solid #638675;padding-top:32px}.cover-bottom strong{font:clamp(28px,3.3vw,52px) Consolas,monospace;color:var(--accent);display:block;margin-top:16px}.cover-bottom .caption{color:#a1bfb0}
.bottom{position:fixed;left:150px;right:46px;bottom:20px;display:flex;align-items:center;gap:9px;font-size:13px;color:var(--dim)}.bottom button{border:1px solid var(--line);background:transparent;color:var(--ink);padding:8px 12px}.bottom button:hover{background:#deebe0}.bottom button:disabled{opacity:.3;cursor:default}.counter{font:15px Consolas,monospace;min-width:70px;text-align:center;color:var(--ink)}#source{margin-right:auto;font:11px Consolas,monospace;max-width:44%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.progress{position:fixed;bottom:0;left:102px;right:0;height:3px;background:#e0e6de}.progress i{display:block;background:var(--green);height:100%;transition:width .2s}.overlay{position:fixed;inset:0;background:#001a16b0;z-index:20;display:flex;align-items:center;justify-content:center;padding:35px}.dialog{background:var(--bg);width:min(1180px,95vw);height:min(850px,90vh);display:flex;flex-direction:column;padding:30px;box-shadow:0 20px 100px #0008}.dialog-head{display:flex;align-items:center;gap:15px;margin-bottom:16px}.dialog-head strong{margin-right:auto;font-size:23px}.dialog button,.dialog select{padding:9px 13px;font:14px inherit;border:1px solid var(--line);background:#fff;color:var(--ink)}.dialog pre{overflow:auto;background:#102d2a;color:#d2efde;padding:24px;font:13px/1.8 Consolas,monospace;flex:1;margin:0;white-space:pre-wrap;overflow-wrap:anywhere}.outline{overflow:auto;display:grid;grid-template-columns:1fr 1fr;gap:8px}.outline button{text-align:left;padding:15px;font-size:17px}.notes{line-height:1.9;font-size:21px;overflow:auto}.toast{font-size:12px;color:var(--green)}[hidden]{display:none!important}@media(max-width:900px){:root{--rail:55px}.rail{width:55px}.rail .vertical{display:none}.rail .edition{font-size:9px}.top{left:78px;right:22px}.slide{padding:14px 25px}.bottom{left:78px;right:18px;gap:5px}.bottom button{font-size:12px;padding:8px}#source{display:none}.params{grid-template-columns:repeat(2,1fr)}.columns{gap:22px}.flow{gap:16px}.cover-bottom{gap:30px;flex-wrap:wrap}.progress{left:55px}.report-grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:600px){.flow,.columns{grid-template-columns:1fr 1fr}.flow p{font-size:15px}.network{gap:12px}.network strong{font-size:44px}.network p{font-size:14px}.weight-bar>div{font-size:14px;gap:8px;flex-direction:column}.weight-bar b{font-size:18px}h1{font-size:31px}.subtitle{font-size:15px}.body-text,.callout{font-size:16px}.outline{grid-template-columns:1fr}.cover-title{font-size:43px}.report-grid{grid-template-columns:1fr}}
@media print{@page{size:A4 landscape;margin:10mm}body{overflow:visible;background:white}.rail,.top,.bottom,.progress,.overlay{display:none!important}main{position:static}.slide,.slide.active{display:block!important;position:relative;height:auto;min-height:170mm;padding:6mm;overflow:visible;page-break-after:always;break-after:page;animation:none}.slide h1{font-size:30px}.slide .subtitle{font-size:15px}.heading{margin-bottom:18px}table{font-size:13px}.params{gap:12px}.params dd{font-size:14px}.content{overflow:visible}.chart{height:200px}.cover-title{font-size:65px}.cover{print-color-adjust:exact;-webkit-print-color-adjust:exact}}
'''
out='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>电影评分预测 · 完整实验汇报</title><style>'+CSS.replace('@import none;','')+'</style></head><body>'
out+='<aside class="rail"><div class="mark">MR.</div><div class="vertical">MOVIE RATING / EXPERIMENT REPORT</div><nav id="dots"></nav><div class="edition">COURSE<br>2026.11</div></aside><div class="top"><span>电影评分预测 / 大数据技术课程项目</span><span id="chapter"></span></div><main>'
for index,s in enumerate(slides):
    out+=f'<section class="slide {s["kind"]}" data-report="{s["source"]}" data-title="{esc(s["title"])}" data-note="{esc(s["notes"])}"><header class="heading"><div class="kicker">{index+1:02d} / {esc(s["chapter"])}</div><h1>{esc(s["title"])}</h1><div class="subtitle">{esc(s["sub"])}</div></header><div class="content">{s["body"]}</div></section>'
out+='</main><div class="bottom"><span id="source"></span><button onclick="openOutline()">目录 G</button><button onclick="openReport()">完整报告 D</button><button onclick="openNotes()">讲稿 N</button><button onclick="fullscreen()">全屏 F</button><button onclick="show(current-1)" id="prev">←</button><span class="counter" id="counter"></span><button onclick="show(current+1)" id="next">→</button><button onclick="window.print()">打印</button></div><div class="progress"><i id="progress"></i></div>'
out+='<div class="overlay" id="modal" hidden><div class="dialog"><div class="dialog-head"><strong id="modalTitle"></strong><select id="selectReport" hidden></select><button id="copy" hidden>复制</button><button id="download" hidden>下载 JSON</button><span id="toast" class="toast"></span><button onclick="closeModal()">关闭 Esc</button></div><pre id="raw" hidden></pre><div class="outline" id="outline" hidden></div><div class="notes" id="notes" hidden></div></div></div>'
payload={'reports':R,'names':NAMES,'paths':PATHS}
out+='<script type="application/json" id="reportData">'+json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')+'</script>'
out+=r'''<script>
const DATA=JSON.parse(document.getElementById('reportData').textContent),slides=[...document.querySelectorAll('.slide')];let current=0,reportKey='blend';const byId=id=>document.getElementById(id);
function show(n){current=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,i)=>{s.classList.toggle('active',i===current);s.setAttribute('aria-hidden',String(i!==current))});document.querySelectorAll('#dots button').forEach((d,i)=>d.classList.toggle('on',i===current));byId('counter').textContent=`${String(current+1).padStart(2,'0')} / ${slides.length}`;byId('chapter').textContent=slides[current].dataset.title;byId('source').textContent=DATA.paths[slides[current].dataset.report];byId('source').title=byId('source').textContent;byId('prev').disabled=current===0;byId('next').disabled=current===slides.length-1;byId('progress').style.width=`${(current+1)/slides.length*100}%`;history.replaceState(null,'',`#${current+1}`)}
slides.forEach((s,i)=>{const dot=document.createElement('button');dot.title=`${i+1}. ${s.dataset.title}`;dot.setAttribute('aria-label',dot.title);dot.onclick=()=>show(i);byId('dots').appendChild(dot);const b=document.createElement('button');b.textContent=`${String(i+1).padStart(2,'0')}　${s.dataset.title}`;b.onclick=()=>{show(i);closeModal()};byId('outline').appendChild(b)});
function modal(mode,title){byId('modal').hidden=false;byId('modalTitle').textContent=title;['raw','outline','notes'].forEach(k=>byId(k).hidden=k!==mode);['selectReport','copy','download'].forEach(k=>byId(k).hidden=mode!=='raw');byId('toast').textContent=''}
function closeModal(){byId('modal').hidden=true}function openOutline(){modal('outline','汇报目录')}function openNotes(){modal('notes','本页讲稿');byId('notes').textContent=slides[current].dataset.note}
function renderReport(k){reportKey=k;byId('selectReport').value=k;byId('raw').textContent=JSON.stringify(DATA.reports[k],null,2);byId('modalTitle').textContent=DATA.names[k]+' · 完整报告'}
function openReport(k){modal('raw','完整报告');renderReport(k||slides[current].dataset.report)}
Object.keys(DATA.reports).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=DATA.names[k];byId('selectReport').appendChild(o)});byId('selectReport').onchange=e=>renderReport(e.target.value);
byId('copy').onclick=async()=>{try{await navigator.clipboard.writeText(byId('raw').textContent);byId('toast').textContent='已复制完整报告'}catch{const sel=getSelection(),range=document.createRange();range.selectNodeContents(byId('raw'));sel.removeAllRanges();sel.addRange(range);byId('toast').textContent='已选中，请按 Ctrl+C'}};
byId('download').onclick=()=>{const url=URL.createObjectURL(new Blob([byId('raw').textContent],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=DATA.paths[reportKey].split('/').pop();a.click();setTimeout(()=>URL.revokeObjectURL(url),500)};
function fullscreen(){if(document.fullscreenElement)document.exitFullscreen?.();else document.documentElement.requestFullscreen?.().catch(()=>{})}byId('modal').onclick=e=>{if(e.target===byId('modal'))closeModal()};
addEventListener('keydown',e=>{if(e.ctrlKey||e.metaKey||e.altKey||/SELECT|INPUT|TEXTAREA/.test(e.target.tagName))return;const k=e.key.toLowerCase();if(k==='escape'){closeModal();return}if(!byId('modal').hidden)return;if(['arrowright','pagedown',' '].includes(k)){e.preventDefault();show(current+1)}else if(['arrowleft','pageup'].includes(k)){e.preventDefault();show(current-1)}else if(k==='home'){e.preventDefault();show(0)}else if(k==='end'){e.preventDefault();show(slides.length-1)}else if(k==='g')openOutline();else if(k==='n')openNotes();else if(k==='d')openReport();else if(k==='f')fullscreen()});
let touchX=0;addEventListener('touchstart',e=>touchX=e.changedTouches[0].screenX,{passive:true});addEventListener('touchend',e=>{if(!byId('modal').hidden||e.target.closest('table,button,select'))return;const delta=e.changedTouches[0].screenX-touchX;if(Math.abs(delta)>80)show(current+(delta<0?1:-1))},{passive:true});const first=Number(location.hash.slice(1));show(Number.isInteger(first)&&first>0?first-1:0);
</script></body></html>'''
(ROOT/'docs').mkdir(exist_ok=True)
(ROOT/'docs/presentation.html').write_text(out,encoding='utf-8')
print(f'Generated {len(slides)} slides with {len(R)} complete reports: docs/presentation.html')

if __name__=='__main__':
    pass
