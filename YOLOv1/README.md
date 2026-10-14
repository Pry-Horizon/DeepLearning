# YOLOv1 图片预测包

原版 YOLOv1 网络结构，VOC 20 类。最终 raw 权重测试 mAP50：50.30%。

## 使用方法（Windows）

1. 解压到普通文件夹，不要直接在压缩包内运行。
2. 安装 Python 3.10 或 3.11，安装时勾选 Add Python to PATH；已有兼容 PyTorch 的 Python 环境可直接使用。
3. 首次使用双击 install_dependencies.bat，联网安装 PyTorch、torchvision、Pillow。已有这些依赖可以跳过。
4. 确认 checkpoints/raw.pt 存在。目前打包时本机尚未收到该权重，必须另行补入；没有权重无法预测。
5. 把待检测图片放进 images，双击 run_predict.bat。
6. 到 outputs 查看带检测框的图片及 predictions.json。

命令行运行：

```shell
python predict.py
python predict.py --source images/example.jpg --conf 0.3
python predict.py --device cpu
```

自动使用 PyTorch 可用的 GPU，否则使用 CPU。网络约 2.72 亿参数，CPU 运行较慢，需数 GB 可用内存。

## 文件

- predict.py：YOLOv1 网络、权重加载、预处理、NMS 和画框，不需要训练代码。
- checkpoints/raw.pt：最终模型权重（约 1.1 GB）。
- images：输入图片，附带 10 张示例图片。
- outputs：预测输出。
- install_dependencies.bat：首次安装依赖。
- run_predict.bat：启动预测。

## 权重来源（模型持有人操作）

从服务器下载：
/home/dell-station/Qhh/YOLO/YOLOv1/runs/yolov1-finetune-60/raw.pt

放入解压目录的 checkpoints/raw.pt 后即可使用。请使用自己训练或可信来源的权重。
