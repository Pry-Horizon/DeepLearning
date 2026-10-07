# YOLOv1-ResNet34：PyTorch 训练与评估基线

当前默认方案按用户最新选择，使用 torchvision ImageNet 预训练 ResNet34、经典 YOLOv1 检测机制、两层全连接检测头。无需自行下载 ImageNet 或编译 Darknet。它是明确标注结构差异的 YOLOv1 基线，不宣称原论文逐项严格复现。

服务器步骤见 [SERVER.md](SERVER.md)。已有 VOC 数据保留在 `data/VOCdevkit`。

## 固定模型与默认协议

| 项目 | 设置 |
|---|---|
| Backbone | ResNet34，torchvision IMAGENET1K_V1 |
| 输入 | 448×448，ImageNet mean/std 归一化 |
| 检测卷积 | 4个3×3 Conv，1024通道，stride=1/1/2/1；LeakyReLU(0.1)，无新增BN |
| 检测头 | 7×7×1024 → FC4096 → LeakyReLU → Dropout0.5 → FC1470 |
| 输出 | 7×7×30，S=7、B=2、C=20，线性训练输出 |
| 宽高 | 网络输出sqrt(w/h)，解码时平方；目标w/h相对整图 |
| 损失 | 平方误差，最高IoU框负责目标，置信度目标为detach的IoU，λcoord=5、λnoobj=0.5 |
| SGD | momentum=0.9，weight_decay=0.0005 |
| 训练轮次 | 默认135轮 |
| LR | 5轮从0.0001逐步升至0.001；第76轮开始0.0001、第106轮开始0.00001 |
| batch | 有效64，默认micro-batch4，累积16次后一次SGD更新 |
| BN | 冻结骨干BN运行统计，affine参数参与训练 |
| 精度 | FP32，无AMP/TF32 |

此LR是本基线的明确实验配置，不是社区统一标准或原论文日程。模型共265,866,494个参数，FC较大；显存不足时可把micro-batch改为2或1，有效batch仍为64。BN统计冻结后，累积不受micro-batch BN统计影响；Dropout随机执行顺序仍可能导致不同micro-batch运行结果不同。

每轮固定随机排列完整训练集，最后补齐至64的倍数。完整VOC训练集每轮16,576张次，其中重复25张次；每轮259次更新，135轮共34,965次更新，与旧原生项目34,913次更新不同。种子、配置、源文件哈希、初始化哈希和环境写入run.json。

增强：随机x/y缩放0.8–1.2、平移±20%、水平翻转、饱和度与亮度调整；不使用Mosaic。困难目标不参与训练。同网格冲突时确定性保留面积最大目标，日志记录冲突数。

## 数据与模型选择

- 默认 `--split full`：VOC2007+2012 trainval，16,551张；训练到最后配置轮次，使用last.pt；VOC2007 test仅最终测试，不用于选择checkpoint。
- 可选 `--split dev`：VOC2007 train+VOC2012 trainval共14,041张训练，VOC2007 val共2,510张独立验证；保存验证mAP最高的best.pt，用它最终测试。
- VOC2007 test共4,952张。若在dev模式调参，再做正式full模式实验，应使用新输出目录重新训练。

## 入口

```bash
python preflight.py
python train.py --device cuda:0
python train.py --device cuda:0 --resume runs/resnet34/last.pt
python evaluate.py --weights runs/resnet34/last.pt --device cuda:0
```

独立验证集实验：`python train.py --split dev --output runs/resnet34-dev`。

续训恢复模型、SGD动量、累计更新和Python/NumPy/Torch/CUDA随机状态；保存每个完整epoch的last.pt。未完成epoch从上个epoch检查点重新执行，不承诺跨硬件或不同软件版本逐位相同。恢复必须使用原epochs、seed、micro-batch、split和输出目录；best.pt仅用于评估。训练入口拒绝覆盖旧输出。

## 输出

- `runs/resnet34/last.pt`：完整训练状态，可续训。
- `runs/resnet34/history.jsonl`：每轮loss、LR、耗时、样本数、更新和冲突数。
- `runs/resnet34/run.json`：完整训练协议与来源。
- `runs/resnet34/metrics.json`：实测mAP、各类AP/TP/FP/precision/recall、FPS和延迟。
- `runs/resnet34/test/detections_*.txt`：可检查的检测结果。
- dev模式另有best.pt和validation指标。

评估使用VOC2007 11点AP@0.5，inclusive坐标、严格IoU>0.5、difficult忽略、重复检测为FP。score_threshold=0.001，NMS=0.5；推理时将线性类别/置信度截断至[0,1]后相乘，按类NMS。

FPS固定batch1、FP32、20次预热、CUDA同步，循环16张不同测试图片。network FPS仅计GPU驻留输入到原始输出；pipeline包括缓存文件读取/解码、resize、归一化、传输、前向、按类NMS，不含绘图。均记录范围及硬件。

## 验证与来源

官方权重已下载并SHA256校验，完整哈希：`b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`。文件87,319,819字节，来源：https://download.pytorch.org/models/resnet34-b627a593.pth

已通过损失梯度、累积等价性、数据划分、网格冲突、NMS、AP和动量/RNG续训检查；并运行真实预训练网络的GPU前向/反向/SGD smoke。完整135轮训练和最终mAP/FPS尚未完成。

```bash
python -m unittest -v test_baseline
python smoke_baseline.py --device cuda:0
python package_resnet34.py --include-weights
```

新包 `dist/yolov1-resnet34-server.zip` 包含当前实现，可包含ResNet34权重；不打包旧训练产物或Darknet。旧历史源码及入口在本机保留为研究记录，旧说明见docs/legacy-darknet-*。当前训练使用train.py，旧run.py和build.py不参与新流程。

参考：[原论文](https://arxiv.org/html/1506.02640v5)、[torchvision ResNet34](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html)、[PyTorch可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)。
