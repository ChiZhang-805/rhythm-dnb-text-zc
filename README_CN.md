# DNB 中文文本评分

线上页面：https://chizhang-805.github.io/rhythm-dnb-text-zc/ 。代码仓库仅上传代码及隔离测试，不上传数据集、用户原文、模型权重或历史结果。GitHub Pages只运行静态前端；未配置独立模型服务时显示“模型服务未连接”，不产生评分。部署方式见根目录README.md。

当前只有一套有效训练代码：**MacBERT 中文编码器 + 五个固定回归输出层**。四个正式输入类别对应原始38项中的四段描述；压力保留为辅助任务，不新增原始采集字段。

## 文件入口

- 数据：`dataset/master.sqlite` 是唯一总库，`dataset/master_view.xlsx` 是同一份数据的一张白底工作表。
- 配置：`train_config.json`，基础模型固定为 `hfl/chinese-macbert-base` 及不可变提交版本。
- 命令：`run.py`；Windows用 `run.ps1`，所有命令复用同一Python环境。
- 页面：`web/`，本地HTTP服务同时提供测试页面及JSON接口。
- 结果：`results/macbert/`，不把工程测试结果当成模型准确率。
- 当前模型指针：`models/macbert/current.json`，仅在正式训练成功结束后发布。
- 旧代码恢复包：`archive/code-before-macbert-20260929-094904.zip`。旧代码已从有效源码和测试目录移除；旧数据、权重及实验记录未删除，也不会被新入口加载。

## 输入与输出

| 类别 | 固定输出 |
| --- | --- |
| 情绪 emotion | 总体心情、愉快程度、悲伤程度、焦虑程度、烦躁程度 |
| 社交 social | 社交意愿、互动满意度、孤独程度、社交负担程度 |
| 饮食 diet | 食欲下降程度、进食过量程度、进餐不规律程度 |
| 睡眠 sleep | 主观睡眠质量、入睡困难程度、睡眠中断程度、醒后疲劳程度 |
| 压力 stress（辅助） | 压力程度 |

每次输入一个类别、一段中文文本。程序固定字段，模型仅预测对应数值；各分数独立，不需要相加等于100。返回值包括0～100整数、除以100后的值和模型身份。它们是文本语义程度估计，不是疾病概率，也不覆盖用户直接填写的0～10主观压力等原始指标。

不把来源、序号、用途、其他类别的分数送入模型。空文本、未知类别及超过模型token上限的文本会明确报错，不静默截断或补默认分数。

## 环境

本机复用已验证的CUDA环境：
`Q:/NAVA-Workspace/Caches/VirtualEnvs/rhythm-dnb-qwen35-gpu/Scripts/python.exe`。
这个历史环境名称不代表仍在运行Qwen；新代码不依赖PEFT或LoRA。

另一台机器先安装匹配显卡的PyTorch，再安装 `requirements.txt`；把环境变量 `DNB_PYTHON` 指向该Python。也可以直接运行 `python run.py ...`。模型目录与数据路径相对项目根目录，无需修改源代码。

## 运行顺序

在本项目目录运行：

```powershell
.\run.ps1 verify
.\run.ps1 prepare
.\run.ps1 test
.\run.ps1 smoke
.\run.ps1 train
.\run.ps1 evaluate --split test
.\run.ps1 serve --port 8765
```

- `prepare`：下载固定版本的中文基础模型，校验文件hash和所有3190条样本的token长度；已下载时可用 `prepare --offline`。
- `test`：隔离单元/集成测试，包括完整训练循环和HTTP契约。
- `smoke`：实际MacBERT在五类各一条样本上执行两次优化步骤，验证梯度、保存加载；不发布权重、不报告模型准确率。这不是正式训练集缩减。
- `train`：每轮使用全部3120条训练记录；30条验证用于选择最佳epoch。40条测试不用于选模型。默认最多12轮，验证类别平均MAE连续3轮无改善即停止。
- `evaluate`：评估完整测试分区，输出逐指标MAE、RMSE、误差在5/10分内的比例、Spearman及训练集均值基线；同时报告连续值和接口取整值的表现。
- `serve`：默认只监听 `127.0.0.1:8765`。端口占用时选另一端口。粘贴文本、选择类别后评分。没有正式模型时页面明确显示尚未训练，API返回503，不生成占位分数。
- `summary / export / audit`：统计、刷新Excel、全量数据与来源审计。
- `import --input 路径`：导入已有逐条审定的中文文本与分数稿件；不会自动造句或打分。

训练过程中可看到每轮已使用的记录数和验证误差。配置、数据指纹、固定分区ID、每轮指标及最佳模型均留档；新训练不会覆盖之前的实验，也不产生另一套训练数据。正式训练失败不会替换已发布模型。当前实现从基础模型开始新训练，不支持断点续训。

浏览器自动测试：Node环境安装Playwright且系统安装Edge后，启动服务，再运行 `node tests/browser.cjs`。`DNB_TEST_URL` 可覆盖服务地址，`DNB_BROWSER` 可改为已安装的Chrome通道；本机也可通过 `NODE_PATH` 使用Codex附带的Playwright。测试覆盖桌面、手机、类别切换、结果字段、错误和清空状态；模拟返回值仅存在于隔离浏览器拦截中，不代表模型准确率，也不写入总库。

## 测试页面及接口

页面与API共用同一个 `Predictor`，不维护另一份预测逻辑。服务不记录或保存用户粘贴的文本。

```powershell
.\run.ps1 predict --category 睡眠 --text "昨晚很难入睡，半夜醒了几次，早上仍然很累。"
```

HTTP：`POST /api/predict`，`Content-Type: application/json`。

```json
{"category":"sleep","text":"昨晚很难入睡，半夜醒了几次，早上仍然很累。"}
```

`GET /api/schema` 返回各类别的固定输出字段；`GET /api/health` 返回模型状态。服务限本机访问，阻止跨站请求，不使用外部模型API。训练完成后在页面点击“刷新状态”即可识别新模型。

批量输入支持CSV、XLSX、JSONL，列为 `record_id/category/text`（也支持总表的中文列名），其他答案列不会作为输入。预测结果是结果文件，不是新数据集：

```powershell
.\run.ps1 predict-file --input .\dataset\master_view.xlsx --output .\results\batch-predictions.jsonl
```

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `contract.py` | 唯一类别、字段、范围与展示名称契约 |
| `corpus.py / chinese_data.py` | 唯一总库、人工语义决策稿件导入、数据检查及Excel |
| `chinese_audit.py` | 数据来源、稿件一致性及近重复审计 |
| `learning_data.py` | 只读分区、数据指纹、类别＋文本编码、批处理 |
| `model.py` | 中文编码器、固定回归层、按样本平均的Huber损失 |
| `training.py` | 基础模型准备、全量微调、早停、模型发布、工程烟雾测试 |
| `evaluation.py` | 逐指标回归评估和仅使用训练集的均值基线 |
| `prediction.py / batch.py` | 单条及批量预测、独立测试评估 |
| `server.py` | 本机HTTP API与页面 |
| `artifacts.py` | 原子保存、文件校验与结果路径保护 |
| `tests/` | 当前任务的隔离测试及浏览器测试 |

## 数据与评估边界

目前3190条：3100条模拟表达、90条真实来源译文，训练3120、验证30、测试40。真实来源的程度分数也是语义参考标注，不能把参考分数上的误差说成临床准确率。测试规模仍小，应继续补充真实中文APP样本，但这不妨碍先训练并报告当前结果。

本次准备阶段不运行正式训练。工程测试权重由临时目录保存并清理，不能被预测服务加载。只有 `train` 成功结束才会启用数值预测。
