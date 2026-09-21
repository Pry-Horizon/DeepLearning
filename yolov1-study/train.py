import os
from loss import YOLOv1Loss
from model import YOLOv1
from dataset import VOCYoloDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
# torch.autograd.set_detect_anomaly(True)

# 自动选择GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

# ============ 修正后的数据集根目录 ============
dataset_root = "E:/DeepLearning/yolov1-study/data"
train_dataset = VOCYoloDataset(
    root=dataset_root, year="2007", image_set="train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = YOLOv1().to(device)
loss_fn = YOLOv1Loss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

# 模型保存文件夹
save_dir = "E:/DeepLearning/yolov1-study/weights"
os.makedirs(save_dir, exist_ok=True)

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        pred = model(imgs)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(
                f"Epoch:{epoch+1}, Batch:{batch_idx}, Loss:{loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"==== Epoch {epoch+1} 平均损失：{avg_loss:.4f} ====")
    # 保存权重
    torch.save(model.state_dict(), os.path.join(
        save_dir, f"yolov1_epoch{epoch+1}.pth"))
