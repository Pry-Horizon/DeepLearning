# 公共服务器：YOLOv1-ResNet34训练

使用公共Python环境，不创建新环境、不自动升级系统依赖。训练不再需要nvcc、原生Darknet或完整ImageNet。先确认管理员允许使用的GPU编号。

## 上传更新

本机新包：`dist/yolov1-resnet34-server.zip`。含源码和已校验的ResNet34权重，约80多MB，不含VOC数据。上传到服务器原项目目录 `/home/dell-station/Qhh/YOLO/YOLOv1`，在该目录执行：

```bash
unzip -o yolov1-resnet34-server.zip
```

保留已有data/VOCdevkit。包不覆盖数据集、checkpoint或训练结果。旧build目录可以留存，当前入口不使用它。

## 公共环境检查

```bash
python preflight.py
```

检查torch、torchvision及其NMS算子、NumPy、Pillow、CUDA运行时、16,551/4,952数据划分和预训练哈希。输出ready=true才能继续。Python3.9可用；NumPy1.24.4无需升级。建议torch≥1.13、torchvision≥0.14且二进制版本匹配、Pillow≥9.1。检查失败应发回输出排查，不要直接pip升级公共环境。

包内已有权重。若使用仅源码包，可运行 `python prepare_resnet34.py` 下载约83MB官方权重。

## GPU选择、检查与训练

下面以“管理员允许使用物理GPU1”为例；先确认这个编号。只在当前终端选择它：

```bash
export CUDA_VISIBLE_DEVICES=1
python smoke_baseline.py
python train.py
```

设置CUDA_VISIBLE_DEVICES=1后，脚本内cuda:0代表物理GPU1，不要再传cuda:1。smoke运行一次micro-batch前后向和SGD更新并丢弃模型，写入runs/resnet34-smoke.json，不产生正式训练权重。

正式训练默认135轮、有效batch64、micro-batch4，FP32/SGD。若smoke显存不足，用 `python smoke_baseline.py --batch 2`；通过后使用 `python train.py --micro-batch 2`，有效batch仍为64。

建议tmux保持运行：

```bash
tmux new -s qhh-yolov1
cd /home/dell-station/Qhh/YOLO/YOLOv1
conda activate yolov10
export CUDA_VISIBLE_DEVICES=1
python train.py
```

Ctrl+B、D离开，`tmux attach -t qhh-yolov1`返回。训练进度实时打印。另一个终端查看已完成epoch记录：`tail -f runs/resnet34/history.jsonl`。

当前模型FC较大，完整模型约1.06GB、带SGD动量的last.pt约2.13GB；原子保存还有临时文件。建议至少预留10GB结果空间，另计VOC数据。

## 中断与评估

每个完整epoch保存last.pt。中断后用相同参数恢复：

```bash
python train.py --resume runs/resnet34/last.pt
```

原来用了micro-batch2、不同seed/epochs/split/output，恢复时必须带回相同参数。没有完成第一个epoch就不会有last.pt；保留旧输出，在新output目录重新启动。

训练结束自动测试和测FPS：

```bash
python -m json.tool runs/resnet34/metrics.json
```

重新评估：`python evaluate.py --weights runs/resnet34/last.pt`。

可选独立验证集模式：`python train.py --split dev --output runs/resnet34-dev`。它使用VOC2007 train+VOC2012 trainval训练，VOC2007 val选best.pt。默认full模式使用最终epoch，测试集不参与模型选择。

正式135轮和最终mAP/FPS仍需服务器执行，本机smoke通过不代表服务器已训练完成。
