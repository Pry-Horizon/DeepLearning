import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from model import YOLOv1
from dataset import class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S = 7


def nms(boxes, thres=0.5):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    res = []
    while boxes:
        best = boxes.pop(0)
        res.append(best)
        keep = []
        for b in boxes:
            x1, y1, x2, y2, score, cls = best
            x1b, y1b, x2b, y2b, scoreb, clsb = b
            inter = max(0, min(x2, x2b)-max(x1, x1b)) * \
                max(0, min(y2, y2b)-max(y1, y1b))
            area1 = (x2-x1)*(y2-y1)
            area2 = (x2b-x1b)*(y2b-y1b)
            union = area1+area2-inter
            if inter/(union+1e-6) < thres:
                keep.append(b)
        boxes = keep
    return res


def detect(img_path, weight_path):
    model = YOLOv1().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = trans(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp).squeeze().cpu().numpy()

    boxes = []
    for grid_y in range(7):
        for grid_x in range(7):
            # 框1
            xc, yc, bw, bh, conf = pred[grid_y, grid_x, 0:5]
            cls_prob = pred[grid_y, grid_x, 10:]
            cls_id = np.argmax(cls_prob)
            score = conf * cls_prob[cls_id]
            if score > 0.3:
                xc = (grid_x + xc)/7 * w
                yc = (grid_y + yc)/7 * h
                bw = bw * w
                bh = bh * h
                x1 = int(xc - bw/2)
                y1 = int(yc - bh/2)
                x2 = int(xc + bw/2)
                y2 = int(yc + bh/2)
                boxes.append([x1, y1, x2, y2, score, cls_id])
            # 框2
            xc2, yc2, bw2, bh2, conf2 = pred[grid_y, grid_x, 5:10]
            score2 = conf2 * cls_prob[cls_id]
            if score2 > 0.3:
                xc2 = (grid_x + xc2)/7 * w
                yc2 = (grid_y + yc2)/7 * h
                bw2 = bw2 * w
                bh2 = bh2 * h
                x1 = int(xc2 - bw2/2)
                y1 = int(yc2 - bh2/2)
                x2 = int(xc2 + bw2/2)
                y2 = int(yc2 + bh2/2)
                boxes.append([x1, y1, x2, y2, score2, cls_id])

    boxes = nms(boxes)
    for b in boxes:
        x1, y1, x2, y2, sc, cls = b
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{class_names[cls]} {sc:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 推理结果保存位置
    out_img_path = "E:/DeepLearning/yolov1-study/result.jpg"
    cv2.imwrite(out_img_path, img)
    print(f"推理图片已保存至：{out_img_path}")


if __name__ == "__main__":
    # 测试图片放在项目根目录
    test_img = "E:/DeepLearning/yolov1-study/test.jpg"
    weight_file = "E:/DeepLearning/yolov1-study/weights/yolov1_epoch10.pth"
    detect(test_img, weight_file)
