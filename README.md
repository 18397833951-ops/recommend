# 电影评分预测项目

使用老师提供的混合数据集训练模型，为 `test_pairs.csv` 中的 100,000 条用户与电影组合预测评分。验收指标为 RMSE，交付内容为预测表格和介绍算法实现思路的 PPT。

教师数据准备、8∶1∶1 划分、三个均值评分基线、正则化偏置、显式 ALS、联合训练的带偏置矩阵分解、MMF 掩码矩阵分解、评分版 ItemCF、固定与自适应融合已实现，正式预测导出已完成。实施步骤与验收条件见 [PROJECT_PLAN.md](PROJECT_PLAN.md)。

## 数据

教师数据已解压到 `data/raw/`，数据集版本为 `hybrid-half-half-latest100k-v1`。

| 文件 | 用途 |
| --- | --- |
| ratings.csv | 31,900,204 条有真实评分的训练记录 |
| test_pairs.csv | 100,000 条需要预测评分的记录，含 userId、movieId、timestamp |
| movies.csv | 电影名称与类别 |
| movie_mapping.csv | 教师数据的电影 ID 与来源映射 |
| tags.csv、links.csv | 可选辅助信息 |
| summary.json、id_ranges.json | 数据规模、来源和 ID 范围说明 |

训练、验证和预测统一使用教师数据包中的 ID。模型参数和统计信息均从教师训练数据学习，预测目标为 0.5～5.0 范围内的真实评分。

## 评测与训练

从教师 `ratings.csv` 中按每个用户的时间顺序划分训练、验证和本地测试集。训练集用于拟合，验证集用于选择算法与参数，本地测试集在方案冻结后评估。

本地划分以每用户 8∶1∶1 为目标。同秒评分作为完整分组，分别选择最接近累计 80% 和 90% 的两条边界，距离相同时取较早边界。至少保留首个时间组训练，单时间组用户全部训练。因时间组限制，部分用户可能没有验证或本地测试记录，数量及冷电影情况写入报告。

\[
RMSE=\sqrt{\frac{1}{N}\sum_{j=1}^{N}(\hat r_j-r_j)^2}
\]

RMSE 越低，评分预测误差越小。选好参数后评估本地测试集，默认使用已训练、选定的模型为正式 `test_pairs.csv` 逐行预测。完整教师评分重训作为可选步骤，训练范围与重新拟合前的独立评测成绩分别记录。

算法依次比较全局均值、用户均值、电影均值、带正则的用户与电影偏置，以及显式 ALS。评分版 ItemCF 和模型融合根据验证结果决定是否加入。统一使用整体逐记录 RMSE 比较模型。

## 环境与配置

沿用 Conda `dl`、Python 3.11、Spark 3.5.7 和 JDK 17。Python 依赖版本统一记录在 [requirements.txt](requirements.txt)。

```powershell
conda activate dl
python -m pip install -r requirements.txt
```

Windows 的 Java、Hadoop 使用系统环境变量 `JAVA_HOME`、`HADOOP_HOME`。当前公共 Spark 配置为 `local[2]`、1 GB Java 堆，Spark 内部压缩为 LZF，Parquet 压缩为 GZIP。

公共配置保存在 `configs/default.yaml`、`configs/paths/default.yaml` 和 `configs/spark/`。后续数据处理、评分训练、预测提交各自增加 Hydra 入口和 YAML 参数，运行时保存完整配置快照。

## 数据检查与划分

从项目根目录运行，入口配置为 `configs/data_prepare.yaml`，划分参数在 `configs/data/prepare.yaml`。

```powershell
python scripts/prepare_data.py --cfg job --resolve
python scripts/prepare_data.py
```

脚本检查训练评分、正式待预测清单、电影及 ID 映射的内容、来源范围、重复和引用。tags、links 仅核对字段并记录文件指纹。检查成功后报告状态为 `passed`。已有输出会阻止覆盖。调整划分时可用 `data.reuse_processed=true` 复用已通过检查的 Parquet，脚本核对原始输入和整理文件指纹，并要求新的划分输出目录。

```text
data/processed/
  ratings/                  完整教师评分，可用于可选全量重训
  test_pairs/               正式待预测清单，含保留输入顺序的 row_id
  movies/                   电影信息
  movie_mapping/            教师电影来源映射
data/splits/
  ratings/split=train/       本地训练评分
  ratings/split=validation/  本地验证评分
  ratings/split=test/        本地测试评分，供方案冻结后评估
  user_summary/             每用户条数与时间边界
artifacts/data_prepare/日期/时间/
  data_preparation.md        可读检查与划分报告
  data_preparation.json      分来源数量、配置和输入输出 SHA256
  .hydra/                   本次配置快照
```

评分 Parquet 保留 userId、movieId、rating、timestamp，并增加 source 标识来源。正式测试仅检查数据关系和保留输入顺序，验证划分由教师 ratings.csv 的用户与时间计算。

小样本检查命令如下。

```powershell
python -m unittest discover -s tests -p test_prepare_data.py -v
```

## 评分均值基线

在数据整理与划分报告通过后运行，入口配置为 `configs/baselines.yaml`，评测划分由 `configs/evaluation/ratings.yaml` 管理。

```powershell
python scripts/baselines.py --cfg job --resolve
python scripts/baselines.py
```

全局均值由全部训练评分计算，用户均值和电影均值分别按 ID 聚合训练评分。未知用户或电影使用训练全局均值回退。统计仅来自 `split=train`，预测保留连续精度。

三个均值基线是固定规则，默认分别保存验证集、本地测试集的整体 RMSE。每个模型一张表，包含两个划分的记录数、RMSE 和回退数量。整体计算采用全部记录的平方误差之和与记录数，每条评分权重相同。JSON 同时保存样本数、平方误差和回退数，便于核对。

后续需要选参数或训练轮次的模型在固定验证集上选择最佳方案，再使用同一本地测试集与基线比较。正式 `test_pairs.csv` 的评分预测属于后续提交步骤。

```text
artifacts/baselines/日期/时间/
  models/global_mean.json        全局均值与训练评分数
  models/users/                 用户均值与评分数
  models/movies/                电影均值与评分数
  predictions/split=validation/  验证真实评分和三种预测、回退标记
  predictions/split=test/        本地测试真实评分和三种预测、回退标记
  baselines.md                  可读 RMSE 对照报告
  baselines.json                误差统计、输入输出校验值与运行配置
  .hydra/                       配置快照
```

预测表的三列为 `global_mean_prediction`、`user_mean_prediction`、`movie_mean_prediction`，对应全局、用户、电影均值。`user_mean_fallback`、`movie_mean_fallback` 标记回退记录。原用户、电影、评分、时间戳、来源和划分均保留，另记录模型版本与生成时间。

脚本核对当前划分报告和 Parquet 的 SHA256，保存预测后读回核对每个划分及模型的评测数量。已有运行结果会阻止覆盖，每次 Hydra 运行会使用独立的时间目录。

小样本测试包含手算均值、整体 RMSE、冷启动回退、训练统计隔离、非有限预测报错和完整文件读回。

```powershell
python -m unittest discover -s tests -p test_baselines.py -v
```

## 正则化偏置基线

入口配置为 `configs/bias.yaml`，模型参数集中在 `configs/model/bias.yaml`。

```powershell
python scripts/bias.py --cfg job --resolve
python scripts/bias.py
```

预测公式为 `全局均值 + 用户偏置 + 电影偏置`。全局均值固定为训练平均分，用户和电影偏置初始化为零，每轮先求电影偏置，再求用户偏置：

```text
电影偏置 = sum(评分 - 全局均值 - 用户偏置) / (该电影训练评分数 + 正则强度)
用户偏置 = sum(评分 - 全局均值 - 电影偏置) / (该用户训练评分数 + 正则强度)
```

默认分别比较正则强度 5、10、20，各训练 10 轮，以整体验证 RMSE 最小的强度和轮次保存模型。每个正则强度均从零偏置重新训练。训练目标是评分平方误差之和，加上正则强度乘以全部用户与电影偏置的平方和。验证评分只参与选择方案，本地测试评分在选优完成后才参与评估。

未知用户或电影的对应偏置取零，另一项已知偏置仍然保留；两项都未知时使用训练全局均值。默认将预测截断到 0.5～5.0，不四舍五入。所有记录均参与整体 RMSE，回退数量按用户或电影偏置任一缺失的记录去重统计。

```text
artifacts/bias/日期/时间/
  models/parameters.json        训练全局均值、选定参数、轮次及评分范围规则
  models/users/                 选定轮次的 userId、user_bias
  models/movies/                选定轮次的 movieId、movie_bias
  predictions/split=validation/  原评分、预测和回退标记
  predictions/split=test/        原评分、预测和回退标记
  bias.md                       一个模型一张表的验证及本地测试结果
  bias.json                     全部方案的验证成绩、最终评测及文件 SHA256
  .hydra/                       完整配置快照
```

训练仅使用教师数据的 `split=train`，沿用固定划分文件及 SHA256 检查。报告中的最终成绩由保存后读回的模型与预测计算；训练轮次和正则强度只根据验证结果选取。

当前完整运行选中正则强度 5、第 10 轮，验证 RMSE 为 0.91452301，本地测试 RMSE 为 0.92096893。模型和报告保存在 `artifacts/bias/2026-11-09/10-38-19-095645/`，可读结果见该目录中的 `bias.md`。

小样本测试核对手算偏置与正则目标、无标签预测、未知 ID、评分截断、独立标量计算的验证选优，以及模型和预测保存读回。

```powershell
python -m unittest discover -s tests -p test_bias.py -v
```

## 显式 ALS

入口为 `scripts/als.py`，配置为 `configs/als.yaml` 和 `configs/model/als.yaml`。使用 Spark 3.5.7 的显式评分 ALS，`implicitPrefs=false`，预测为用户向量与电影向量的点积。此版本直接拟合真实评分。

```powershell
python scripts/als.py --cfg job --resolve
python scripts/als.py
```

一次运行只训练配置中的一组超参数，不自动搜索组合。在 `configs/model/als.yaml` 中修改：

```yaml
rank: 16         # 用户和电影特征向量的维数
reg_param: 0.1   # 正则强度
max_iter: 10     # 训练轮数
```

随机种子为 20261106。每次运行从头训练，使用验证集整体 RMSE 比较不同配置，本地测试成绩用于评估固定方案，不用于调整超参数。运行配置会随结果保存到 `.hydra/`。

ALS 正则系数按用户或电影的训练评分数缩放，是 Spark 的 ALS-WR 规则，其数值与前述偏置模型的正则强度含义不同。算法参数详见 [Spark ALS 文档](https://spark.apache.org/docs/3.5.7/ml-collaborative-filtering.html)。

未知用户或电影保留在评测中，优先采用训练用户均值，用户统计缺失时再用训练全局均值。固定将最终预测截断到 0.5～5.0，不四舍五入。冷启动原始 `als_prediction` 为 NaN，回退后的 `prediction` 必须为有限数值。RMSE 使用最终预测并计入所有记录。

ALS 的评分块与特征矩阵需要更多内存，因此在本任务配置中单独设置 `spark.driver_memory: 4g`。仍使用 `local[2]`，其余任务沿用公共 Spark 配置。所有模型和回退统计只从 `split=train` 拟合，输入划分文件沿用 SHA256 核对。

```text
artifacts/als/日期/时间/
  models/als/                   本次训练的 Spark ALSModel，含用户、电影特征及元数据
  models/user_means/            冷启动回退用的训练用户均值
  models/parameters.json        全局均值、本次超参数、种子与评分范围规则
  predictions/split=validation/  真实评分、原始 ALS 预测、最终预测、回退标记
  predictions/split=test/        本地测试的同类字段
  als.md                        一个模型一张表的验证和本地测试结果
  als.json                      单次训练的验证成绩、最终误差、输入输出 SHA256
  .hydra/                       配置快照
```

模型保存后重新加载，再保存、读回预测核对验证误差与评测数量。JSON 的 `fit` 记录本次超参数、验证误差和耗时，`metrics` 记录模型读回后的验证与本地测试结果。小样本测试确认一次运行仅调用一次 ALS 训练，超参数传入正确，并检查特征点积、低秩评分还原、未知 ID、无标签输入、评分标签隔离及完整文件读回。

此前的 8 组参数实验选出了维数 16、正则系数 0.1、10 轮，因此将其设为当前默认配置。该实验验证 RMSE 为 0.84739646，本地测试 RMSE 为 0.85926135，较偏置基线降低约 6.70%。原实验总耗时约 30.86 分钟，报告保存在 `artifacts/als/2026-11-09/11-00-27-726460/`。这些成绩属于原实验记录；当前脚本每次只训练配置中的一个模型。

```powershell
python -m unittest discover -s tests -p test_als.py -v
```

## 联合训练的带偏置矩阵分解

入口为 `scripts/biased_mf.py`，配置为 `configs/biased_mf.yaml` 和 `configs/model/biased_mf.yaml`。PyTorch 在单机上使用 Adam 联合学习用户、电影特征向量及两个偏置，Spark 负责读取既有 Parquet 和核对保存的预测。PyTorch 依赖也在 `requirements.txt` 中。

```powershell
python scripts/biased_mf.py --cfg job --resolve
python scripts/biased_mf.py
```

预测为 `训练全局均值 + 用户偏置 + 电影偏置 + 用户向量与电影向量的点积`。全局均值从训练集计算并固定；两个偏置从零开始，向量从随机小数开始，四类参数在同一训练过程中更新。模型原理可参考 [Surprise 的带偏置矩阵分解公式](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)。

一次只训练配置中的一组超参数：

```yaml
rank: 16
learning_rate: 0.01
regularization: 0.01
max_epochs: 20
batch_size: 131072
patience: 3
```

每批训练损失为原始预测与真实评分的 MSE，加上 `regularization` 乘以该批记录所对应的用户向量、电影向量、用户偏置、电影偏置的平方和均值。重复出现的 ID 按评分记录次数计入这项正则，Adam 不额外使用 weight decay。这一正则定义与 Spark ALS 的系数不直接等同。训练时使用原始分数计算损失，评测时固定截断到 0.5～5.0，保留连续值。

每轮打乱完整训练记录，训练后计算完整验证集的整体 RMSE；误差改善就保存最佳权重，连续 `patience` 轮没有改善则提前停止，最多训练 `max_epochs` 轮。保存后恢复最佳轮次，再读取本地测试评分。测试成绩不参与轮次选择。

默认使用 CPU、4 个计算线程，也可在配置中指定可用的 CUDA 设备。读取训练和验证后关闭 Spark 会话，再进行 PyTorch 训练；训练数据只在内存中保留紧凑数值数组，不额外落盘复制。ID 掩码和参数表仅由训练集构造。未知 ID 的对应向量、偏置取零，已知一侧的偏置继续使用；两侧都未知时使用训练全局均值。全部评分保留并计入误差。

```text
artifacts/biased_mf/日期/时间/
  models/best.pt                最佳轮次的四类权重、训练 ID 掩码及全局均值
  models/parameters.json        本次配置、训练统计及最佳轮次
  predictions/validation.csv.gz 验证评分、时间戳、原始和最终预测、回退标记
  predictions/test.csv.gz       本地测试的同类字段
  biased_mf.md                 一个模型一张表的验证和本地测试结果
  biased_mf.json               逐轮 history、最佳轮次、最终评测与输入输出 SHA256
  .hydra/                      配置快照
```

`models/best.pt` 可通过 `scripts/biased_mf.py` 中的 `load_model` 加载，模型输入用户和电影 ID，不需要真实评分。两个预测文件使用标准 gzip CSV，避免增加 Parquet Python 读取库依赖；保存后用 Spark 读回核对数量和 RMSE。时间戳随每条评分保存。

当前完整运行训练了 5 轮，在连续 3 轮未改善后提前停止，恢复第 2 轮权重。验证 RMSE 为 0.83976814，本地测试 RMSE 为 0.84958087，较基础 ALS 的本地测试误差降低约 1.13%。完整流程耗时 424.40 秒，模型与报告位于 `artifacts/biased_mf/2026-11-09/15-07-48-034939/`。

小样本测试核对四类参数的手算更新、正则目标、未知 ID、最佳轮次恢复、测试集读取顺序和预测文件读回。

```powershell
python -m unittest discover -s tests -p test_biased_mf.py -v
```

## MMF 掩码矩阵分解

入口为 `scripts/mmf.py`，配置为 `configs/mmf.yaml`、`configs/model/mmf.yaml`，复用现有 PyTorch 依赖。

```powershell
python scripts/mmf.py --cfg job --resolve
python scripts/mmf.py
python -m unittest discover -s tests -p test_mmf.py -v
```

普通矩阵分解直接计算用户与电影向量的点积。MMF 共享一对基础向量，在点积前给它们乘上不同掩码，再把各成分的点积相加：`训练均值 + 用户偏置 + 电影偏置 + sum_k dot(p * mask_user_k, q * mask_movie_k)`。本实现采用官方高斯掩码，并为每个用户、电影学习各成分的偏移量；向量、偏置、偏移一起更新。

默认维数 16、成分数 4，向量及偏置学习率 0.005，偏移学习率 0.005，Adam、批大小 32768、最多 20 轮、连续 3 轮验证无改善停止。不使用调度器；默认损失为评分 MSE，不额外加正则或 weight decay。以上参数全部在模型 YAML 中配置。训练使用原 25,517,798 条训练评分，原验证集选最佳轮次，恢复后才读取本地测试。未知 ID 的向量和偏置为零，已知侧偏置保留，最终分数截断到 0.5～5.0。

结果保存在 `artifacts/mmf/日期/时间/`：`models/best.pt` 可由 `mmf.load_model` 加载；`models/parameters.json` 记录模型配置；`predictions/validation.csv.gz`、`test.csv.gz` 保存全部逐行预测；`mmf.md`、`mmf.json` 记录逐轮训练、整体 RMSE、回退数量、输入输出指纹和官方源码版本。保存后用 Spark 读回核对数量和误差。

依据为 [MMF 论文](https://doi.org/10.1145/3770855.3817710) 与 [官方实现](https://github.com/snudatalab/MMF/blob/c86307711752dff78fc7f8ade4ed4ffdc38c81c8/main_completion.py)。这是适配教师数据的单组小规模实验：启用可学习偏移，均值使用本项目训练均值；没有采用官方示例的 1200 维、64 成分，也不代表复现论文基准成绩。测试包含三维手算、未知 ID、偏移更新、最佳模型恢复和全流程验收；另在本地下载固定版本官方代码后核对了预测与梯度，其他机器没有该可选参考文件时会跳过此项对照。

全量实验实际训练 7 轮，验证集选中第 4 轮。MMF 的结果如下：

| 划分 | 记录数 | RMSE | 回退记录 |
| --- | ---: | ---: | ---: |
| 验证集 | 3,187,481 | 0.83567061 | 12,291 |
| 本地测试集 | 3,194,925 | 0.84569140 | 19,777 |

全流程耗时 1120.53 秒，可训练参数共 12,117,315 个（含 ID 索引空位）。相较原带偏置矩阵分解的 0.83976814 / 0.84958087，两个划分的误差略有下降；验证仍高于固定融合的 0.83193072，因此依据验证集继续保留固定融合。MMF 完整模型、预测和报告在 `artifacts/mmf/2026-11-14/09-57-08-179493/`。本次仅运行这一组参数，没有据本地测试结果继续调参。

## 评分版 ItemCF

入口为 `scripts/itemcf.py`，配置为 `configs/itemcf.yaml` 和 `configs/model/itemcf.yaml`。只使用训练评分计算统计和相似度，用用户对相似电影的训练评分预测目标电影评分。

```powershell
python scripts/itemcf.py --cfg job --resolve
python scripts/itemcf.py
```

先将训练评分减去各用户的训练平均分，得到中心化评分矩阵，再计算电影列之间的余弦相似度。相似度采用全列的范数，未评分位置视为零，不称为共同评分子集上的 Pearson 相关。Spark 的 `RowMatrix.columnSimilarities` 在阈值为 0 时精确计算，正值时使用 DIMSUM 采样近似计算。参见 [Spark API](https://spark.apache.org/docs/3.5.7/api/python/reference/api/pyspark.mllib.linalg.distributed.RowMatrix.html)。

```yaml
sampling_threshold: 1.0
neighbors: 40
min_neighbors: 2
min_item_ratings: 20
min_similarity: 0.05
```

`sampling_threshold` 控制近似计算的成本与精度，不是最终相似度的过滤阈值；正式邻居要求相似度大于 `min_similarity`，近似估计超过 1 时截断到 1。每部电影保留相似度最高的最多 40 个邻居，相同相似度时按电影 ID 排序。训练评分少于 20 条的电影不参与相似度计算，但仍保留其训练均值和完整用户历史用于回退及预测。

对用户与目标电影，匹配这些邻居中用户训练时评分过的电影，按下式预测：

```text
预测 = 目标电影训练均值
     + sum(相似度 × (用户对邻居的评分 - 邻居电影训练均值)) / sum(相似度)
```

这是使用均值校正的评分邻域方法，见 [KNNWithMeans 公式](https://surprise.readthedocs.io/en/stable/knn_inspired.html)。匹配不足 `min_neighbors` 个时使用目标电影训练均值；目标电影没有训练记录时使用用户训练均值，两者均未见时使用全局均值。全部评分保留，最终预测截断到 0.5～5.0，不四舍五入。回退数量包含邻居不足的记录。

训练评分有 25,517,798 条，精确枚举约产生 33.5 亿次共评分电影组合，因此默认使用近似相似度而非全量精确计算。原始训练历史不额外复制，预测时复用 `data/splits/ratings/split=train/`；相似度图和均值统计单独保存。一次运行一组配置，不自动搜索参数。

```text
artifacts/itemcf/日期/时间/
  models/global_mean.json        全局训练统计
  models/users/                  用户训练均值与记录数
  models/movies/                 电影训练均值与记录数
  models/neighbors/              每部电影的邻居 ID 和相似度数组
  models/parameters.json         模型配置、图规模和训练历史路径
  predictions/split=validation/  原评分、时间戳、原始/最终预测、邻居数及回退标记
  predictions/split=test/        本地测试的同类字段
  itemcf.md                      一个模型一张表的验证和本地测试结果
  itemcf.json                    图规模、误差、耗时、配置及输入输出 SHA256
  .hydra/                        配置快照
```

`scripts/itemcf_scoring.py` 包含逐用户的评分计算，也会分发给 Spark Python 工作进程。保存预测后读回核对评测数量和有限数值。小样本检查涵盖手算加权评分、精确余弦、近似计算路径、高电影 ID、零向量、冷启动和评分标签隔离。

训练向量缓存为 Spark 原生稀疏向量，供相似度计算重复扫描。`configs/itemcf.yaml` 中的 `data.prediction_partitions` 默认设为 16，合并评分阶段的 Python 任务，减少进程启动和广播加载开销；SQL 分组仍使用公共 Spark 分区设置。

已完成的全量实验保留了 30,381 部电影的 1,194,931 条有向邻居，验证 RMSE 为 0.97064105，本地测试 RMSE 为 0.97727760。验证和本地测试平均匹配邻居为 2.21、2.02 个，因邻居不足回退的记录占 50.39%、53.99%。本组配置未超过电影均值基线或联合矩阵分解，保留为算法对照。模型及报告在 `artifacts/itemcf/2026-11-09/16-21-09-845229/`，该次全量实验耗时 3043.83 秒。

```powershell
python -m unittest discover -s tests -p test_itemcf.py -v
```

## ALS 与矩阵分解加权融合

入口为 `scripts/blend.py`，配置为 `configs/blend.yaml` 和 `configs/model/blend.yaml`。修改 `paths.als_run`、`paths.mf_run` 可以指定已有的模型运行目录。复用保存的最终预测，不重新训练模型，不需要新增依赖。

```powershell
python scripts/blend.py --cfg job --resolve
python scripts/blend.py
```

预测公式为 `alpha × ALS预测 + (1-alpha) × 带偏置矩阵分解预测`。两个输入都是已完成回退及评分截断的分数，非负权重之和为 1，因此融合分数仍在 0.5～5.0 范围内。

权重只通过完整验证集选择。令 `d=ALS预测-MF预测`、`e=MF预测-真实评分`，先计算 `alpha=-sum(e*d)/sum(d*d)`，再将权重限制在 [0,1]；两份预测完全相同时取 0。这个公式直接得到该范围内平方误差最小的权重，不需要重复尝试很多组权重。允许权重取 0 或 1，以保留单模型方案。

脚本核对两个模型报告中的划分指纹、输入预测文件 SHA256，并按用户和电影 ID 对齐，检查时间戳、真实评分、唯一性和全部记录覆盖。读取顺序不需要相同。验证集选出的权重写入 `models/parameters.json` 后才读取本地测试预测，测试误差不参与权重选择。

```text
artifacts/blend/日期/时间/
  models/parameters.json        冻结的权重、输入运行目录与评分范围
  predictions/split=validation/ 验证集的两个模型预测与融合预测
  predictions/split=test/       固定权重的本地测试预测
  blend.md                      每个模型一张表的验证和本地测试结果
  blend.json                    权重、误差统计、输入输出指纹及配置
  .hydra/                       本次配置快照
```

融合预测保存后重新读取，核对数量与平方误差。小样本测试覆盖手算权重、端点、相同预测、缺失/重复/错位记录，以及验证集选择与测试集隔离。

已完成的全量实验选出 ALS 权重 0.4152908341、带偏置矩阵分解权重 0.5847091659。验证集 3,187,481 条，融合 RMSE 为 0.83193072；本地测试 3,194,925 条，融合 RMSE 为 0.84191037。相较单独的带偏置矩阵分解，验证与本地测试误差分别降低约 0.93%、0.90%。权重仅由验证集确定，当前将融合方案作为最佳候选。

本次运行耗时 70.19 秒，权重、预测和完整报告位于 `artifacts/blend/2026-11-12/14-46-10-662025/`。使用的是 ALS 单组重跑结果及矩阵分解第 2 轮的已保存预测。

```powershell
python -m unittest discover -s tests -p test_blend.py -v
```

## 自适应权重融合网络

入口为 `scripts/adaptive_blend.py`，配置为 `configs/adaptive_blend.yaml`、`configs/model/adaptive_blend.yaml`，无需新增依赖。

```powershell
python scripts/adaptive_blend.py --cfg job --resolve
python scripts/adaptive_blend.py
```

网络输入 4 个数：ALS 预测、带偏置矩阵分解预测、`log1p(用户训练评分数)` 和 `log1p(电影训练评分数)`。评分数量和标准化均值、标准差均仅从原训练集计算。网络为 `Linear(4,16) → ReLU → Linear(16,1) → Sigmoid`，共 97 个可训练参数，输出每条记录的 ALS 权重 alpha，最后计算 `alpha × ALS + (1-alpha) × MF`。

沿用原有 8∶1∶1 数据划分。两个基础模型保持固定，用它们对原训练集的预测训练融合网络，目标为评分 MSE；Adam 学习率 0.001，批大小 131072，最多 20 轮，验证连续 3 轮无改善停止。网络从固定 50/50 权重开始，初始状态作为第 0 轮候选保存。每轮用原验证集选最佳权重，恢复并保存选定方案后才读取本地测试。不额外拆验证集，不重新训练基础模型。

原训练集的基础预测通过保存的 ALS 特征向量和 MF 权重分批计算，仅保留在内存；验证与测试复用已核对的固定融合预测，继续核对文件指纹、记录覆盖和基础模型 RMSE。训练时基础模型已见过这些评分，验证与测试则是未见记录，是否能改进以实际验证结果为准。

```text
artifacts/adaptive_blend/日期/时间/
  models/best.pt                 融合网络、训练标准化统计、最佳轮次
  models/counts.npz              用户和电影的原训练评分数量
  models/parameters.json         配置、基础模型目录、验证选出的方案
  predictions/validation.csv.gz 验证逐行评分、预测和自适应权重
  predictions/test.csv.gz       本地测试的同类字段
  adaptive_blend.md             每种融合方法各一张结果表
  adaptive_blend.json           逐轮记录、最终误差及输入输出指纹
  .hydra/                       本次配置快照
```

预测 CSV 保留用户、电影、时间戳、真实评分、两个基础预测、ALS 权重、自适应预测和固定融合预测。保存后读回核对完整数量、评分范围及误差。`load_model` 可加载独立网络；部署时还需两个基础模型与 counts.npz 中的训练数量统计。

本次全量实验使用全部 25,517,798 条训练评分，运行 624.70 秒，实际训练 3 轮。训练 MSE 从 0.54312382 降至 0.53586629，验证 RMSE 却依次为 0.83311554、0.83477388、0.83632939，均未超过固定权重融合的 0.83193072。因此在读取本地测试前已选定继续使用固定融合，本次不替换现有方案。

检查点恢复为第 0 轮，即未训练的 50/50 初始参照，其验证、本地测试 RMSE 为 0.83225828、0.84239418；这两个数不是训练后学到自适应权重的成绩。原固定融合在本次读回的本地测试 RMSE 仍为 0.84191037。完整逐轮记录、候选检查点和预测在 `artifacts/adaptive_blend/2026-11-12/15-31-50-871545/`，可读结论见 `adaptive_blend.md`。小样本和全量读回、5 个输出文件指纹检查均通过。

```powershell
python -m unittest discover -s tests -p test_adaptive_blend.py -v
```

## 交付

预测表格需要完整对应原始 `test_pairs.csv` 的 100,000 行，保留用户 ID、电影 ID、时间戳和行顺序，增加连续预测评分。最终表头、文件格式及命名以老师提供的提交模板为准，目前待确认。

同时准备算法 PPT，说明数据来源、验证划分、算法原理、参数、RMSE 对照和最终预测流程。


## 正式预测导出

配置为 `configs/predict.yaml`。直接加载选定的 ALS 与矩阵分解模型及固定融合权重，不重新训练。

```powershell
python scripts/predict.py --cfg job --resolve
python scripts/predict.py
```

输出位于 `artifacts/submission/日期/时间/`，`predictions.csv` 包含 userId、movieId、timestamp、prediction。脚本核对原清单、输入文件和模型指纹，保存后逐行读回检查顺序与连续预测精度，完整统计保存在 `submission.json`。

本次结果在 `artifacts/submission/2026-11-13/09-44-46-741346/`：100,000 行，预测范围 0.6503717882547817～5.0，均值 3.5129418065562685。两个模型分别有 730 条记录触发既定回退，全部保留。完整流程耗时 31.30 秒。正式真实评分未提供，因此正式 RMSE 由教师计算。
