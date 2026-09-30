# YOLOv1：以论文为准，使用作者原生 Darknet 核心

当前项目已移除旧 ResNet34 Baseline，以及上一版自行实现的 PyTorch 模型、损失、增强和训练循环。模型计算、损失、SGD、图像处理和采样来自作者的历史 C/CUDA 源码，不再维护另一套 Python 网络。

**代码准备与正式复现结果是两件事：目前没有完成原生 GPU 编译或 135 轮训练，没有最终 mAP/FPS。** 不会把论文结果、旧 Baseline 权重或上一版 PyTorch smoke test 当作当前项目结果。

## 来源和优先级

1. 以 [CVPR 2016 YOLOv1 原论文](https://arxiv.org/html/1506.02640v5) 为准。
2. 论文未给出的实现细节，参考 [作者 pjreddie/darknet 历史提交 42ba5d4](https://github.com/pjreddie/darknet/tree/42ba5d4585a252b344cc737420e46ad93f005dbe)。
3. `darknet/` 保存该提交的原始文件，`upstream-lock.json` 保存下载归档和每个文件的 SHA256。构建前逐个检查，不使用会继续变化的 master。
4. 构建在 `build/darknet/` 中进行。可审查修改位于 `support/build.patch`：输出路径、测试集路径、训练进度/保存频率、测时入口、编译兼容选项，以及论文优先的学习率/停止步数。原始数学计算文件保持不变。

作者这份历史配置确实有 24 Conv、2 FC（4096→1470）、Dropout 0.5、线性输出和 B=2；没有后来 master 配置引入的 BN/local/B=3 改动。

## 固定训练协议

| 参数 | 采用值 |
|---|---|
| 训练目标 | 论文的约 135 个数据集等效轮次 |
| batch | **64**，每次 SGD 更新严格使用 64 张 |
| subdivisions | **64**，保留所锁定作者配置；内部每次处理 1 张后累积，不能误读为把 batch 改成 1 |
| 训练数据 | VOC2007 trainval + VOC2012 trainval，共 16,551 张 |
| 测试数据 | VOC2007 test，共 4,952 张；不用于挑选模型 |
| 优化器 | 原始 SGD，momentum=0.9、decay=0.0005 |
| 输入、网络 | 448×448，24 Conv＋2 FC，LeakyReLU(0.1)，Dropout(0.5)，S=7/B=2/C=20 |
| loss | 作者 detection_layer：λcoord=5，λnoobj=0.5，IoU 负责框/置信度，sqrt 宽高参数化 |
| LR 主要阶段 | 前 75 轮阶段 0.01，接着 30 轮 0.001，最后 30 轮 0.0001 |
| LR 开始阶段 | 0.001；按作者源码第 200/400/600 次更新分段升至 0.01 |
| 精度 | 原始 float/CUDA 路径，无 AMP、AdamW、ResNet 或现代增强 |

**论文与公开代码不一致的地方必须说明，不能冒称完全一致：**

- 作者 cfg 原来是 `max_batches=40000`，不是 135 轮。用户已明确论文优先，因此实际训练配置不使用 40,000 次更新。
- 16,551×135 不能被 64 整除。为保持每次更新 batch=64，在累计轮次边界向上取整：总更新次数 **34,913**，处理 **2,234,432 张次**；135 轮目标为 2,234,385 张次，最后一批多 **47 张次**，即约 **135.00284** 个数据集等效轮次。不会改成 batch=47 或伪称数学上恰好相等。
- 75、105 轮 LR 边界对应累计更新第 **19,396、27,154** 次。采用累计取整，不再按每轮补齐 25 张造成持续额外采样。
- 原作者 loader 随机有放回取图，不是 PyTorch 每轮 shuffle 遍历。保留原代码，因此 epoch 表示数据集等效采样量，不表示每张图片恰好见过一次。
- 论文没有公开 warm-up 精确长度；200/400/600 来自该历史作者配置，并包含在前 75 轮阶段中。已移除上一版自行选择的“5 轮 warm-up”。这依然属于补充解释，无法证明就是论文那次实验的逐步 LR 日志。
- 历史随机种子、完整预训练实验记录没有公开。不能承诺逐位相同训练、固定达到 63.4% mAP，或在 RTX 5060 上得到 Titan X 的 45 FPS。

`protocol.py` 固定上述参数，唯一训练入口 `run.py train` 不提供修改 epochs、batch、subdivisions 或 LR 的参数。`runs/paper/protocol.json` 记录实际协议。

## 文件结构

- `darknet/`：作者原始 C/CUDA/配置/标签转换源码。
- `upstream-lock.json`：上游来源与每文件校验值。
- `build.py`：生成可审查构建目录并编译，保留原始计算核心。
- `prepare_data.py`：下载、MD5 校验并安全解包 VOC。
- `prepare_voc.py`：作者标签转换公式的 Python 3 适配、完整划分核查、训练/测试清单。
- `weights.py`：GitHub 权重下载与结构审计，不允许缺层权重静默当作完整初始化。
- `run.py`：环境预检、原生训练、训练完成后的自动测试/测时。
- `metrics.py`：读取作者 `yolo valid` 输出，计算 VOC2007 11 点 AP。
- `support/benchmark.c`：计时扩展，调用原始推理、解码和 NMS。
- `test_author_source.py`：上游哈希、网络保持一致、训练参数、标签、AP 和缺层拒绝测试。

旧 `.pth` 权重和旧输出属于历史产物，未被新入口使用；它们不兼容原生 `.weights` 格式。数据集继续使用已下载内容。

## GitHub 搜索和预训练权重审计

已实际下载 [frankzhangrui/Darknet-Yolo 的固定提交中的 extraction.conv.weights](https://github.com/frankzhangrui/Darknet-Yolo/blob/1398cf31d9f048558b32d568cd0dba5b140d92be/extraction.conv.weights)，而不仅检查网页说明。

- 文件：`data/pretrained/github-extraction.conv.weights`。
- SHA256：`048f832fbaddec03256f2c1cf380e5694d6b580f24639515587daa358ff32e9f`。
- 大小：47,657,232 字节；对照锁定架构，只够前 **16 Conv**。
- 原论文要求前 **20 Conv** 经过 ImageNet 预训练，该 2015 格式完整前 20 Conv 需要 **89,612,560 字节**。
- 审计：`data/pretrained/github-extraction.conv.audit.json`。
- 此文件与上一轮研究镜像的内容一致。原始 loader 对短文件的 fread 不会主动报缺层，因此程序在启动前另加长度及来源检查，防止“可以运行”被误称“完整预训练”。

还检查了作者历史 cfg/提交、常见 PyTorch/TF 实现和 GitHub release。ResNet 替代、加 BN、Fast YOLO、训练过 VOC 的检测权重都不能替代这里所需的 ImageNet-only 20 层初始化。尚未找到来源、结构均可核验的完整文件；作者完整权重地址实测返回 403。需要取得完整初始化，或在已获访问权限的 ImageNet 数据上执行预训练。原论文未公开预训练精确 epoch 数，项目不会再编造一套预训练日程称为原文。

## 当前环境与验证范围

当前 Windows 主机有 RTX 5060，父目录 `.venv` 中有 PyTorch，但原生 Darknet 构建需要 Linux/WSL、gcc、make 和支持目标显卡的 CUDA **编译工具链**。检查未找到可用的 WSL Linux 发行版或 nvcc；PyTorch 自带运行库不能替代 nvcc。

已完成：

- 上游 109 个文件的哈希验证。
- 7 项源码/配置/标签/AP 测试。
- 生成可审查原生构建目录和配置。
- VOC 标签转换与 16,551/4,952 划分验证，无训练/测试 ID 重叠。

未完成：原生 C/CUDA 编译验证、GPU 前向/反向验证、ImageNet 完整初始化验证、135 轮训练和最终指标。上一版 PyTorch 的 GPU 测试不适用于此原生版本。

## 使用

在本机可以先检查代码与数据：

```powershell
$py = 'C:\GitHub\DeepLearning\.venv\Scripts\python.exe'
& $py -m unittest -v test_author_source
& $py build.py --generate-only
& $py prepare_voc.py
& $py weights.py --download-github
& $py run.py preflight
```

真正训练需在具备原生 CUDA 编译环境的 Linux/WSL 下执行；迁移后必须重新运行 `prepare_voc.py`，不能沿用 Windows 图片路径清单：

```bash
python3 -m pip install -r requirements.txt
python3 build.py --arch sm_120
python3 prepare_voc.py
python3 run.py preflight --pretrained /path/to/verified-first20.weights
python3 run.py train --pretrained /path/to/verified-first20.weights
```

`sm_120` 对应当前目标卡的构建选项；脚本会核验 nvcc 是否支持该值，其他机器应传其实际架构。CPU 构建 `--cpu` 仅用于诊断：历史代码的 HSV 色彩增强位于 GPU crop 路径，训练入口拒绝将 CPU 路径冒充相同方法。

完整预训练文件必须包含该旧格式 16 字节头部、seen=0、全部前 20 Conv；还需同名 `.provenance.json` 记录真实来源：

```json
{"kind":"imagenet_first20","source":"可核验的实际来源或本地预训练记录","sha256":"该文件的实际SHA256"}
```

这份来源记录不是历史真实性的自动证明。不得把检测模型截取的权重、缺层镜像、未知训练来源或后来的 BN 模型写成 ImageNet 初始化。作者 `darknet/cfg/extraction.cfg` 作为预训练参考原样保存；它与论文没有公开的预训练实验细节仍须区分。

训练从有效 ImageNet 初始化开始，连续跑到论文轮次目标，保存 `build/darknet/backup/paper_last.weights` 和 `paper_final.weights`。原版 `.weights` 不保存 SGD 动量/完整 RNG 状态，因此这里不提供“无损严格续训”的虚假承诺；中断后的历史 checkpoint 可以保留，但不能称为重放了同一次不中断实验。

## 最终指标

训练完成后自动执行原生 VOC2007 test 推理、AP 计算和单图测时，写入 `runs/paper/metrics.json`，包含：

- VOC2007 11 点 mAP@0.5 和 20 类 AP。
- 各类别 TP、FP、precision、recall（置信度阈值 0.001）。
- 原生 network API FPS、pipeline FPS、平均/p50/p95 延迟。
- batch=1、FP32、20 次预热、CUDA 同步、测量次数、硬件信息、最终权重 SHA256 和训练样本计数。

network API 本身包括原版内部的 CPU/GPU 传输；pipeline 包括同一测试图片的缓存文件读取、解码、原版 resize、推理、解码和 NMS，不包括绘图。这个范围会写入报告，不能与不同硬件/范围的 FPS 混用。

单独重新评估：

```bash
python3 run.py evaluate --weights build/darknet/backup/paper_final.weights --iterations 100
```

**63.4% mAP、45 FPS 是原论文普通 YOLO 的参考值，不是本项目当前的实测结果。**
