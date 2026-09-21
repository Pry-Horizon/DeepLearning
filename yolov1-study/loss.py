import torch
import torch.nn as nn

S = 7
B = 2
C = 20


def iou(box1, box2):
    """计算两个框的IOU，输入：x,y,w,h（归一化）"""
    b1x1 = box1[..., 0] - box1[..., 2]/2
    b1y1 = box1[..., 1] - box1[..., 2]/2
    b1x2 = box1[..., 0] + box1[..., 2]/2
    b1y2 = box1[..., 1] + box1[..., 3]/2

    b2x1 = box2[..., 0] - box2[..., 2]/2
    b2y1 = box2[..., 1] - box2[..., 3]/2
    b2x2 = box2[..., 0] + box2[..., 2]/2
    b2y2 = box2[..., 1] + box2[..., 3]/2

    inter_x1 = torch.max(b1x1, b2x1)
    inter_y1 = torch.max(b1y1, b2y1)
    inter_x2 = torch.min(b1x2, b2x2)
    inter_y2 = torch.min(b1y2, b2y2)

    inter = torch.clamp(inter_x2-inter_x1, 0) * \
        torch.clamp(inter_y2-inter_y1, 0)
    area1 = (b1x2-b1x1)*(b1y2-b1y1)
    area2 = (b2x2-b2x1)*(b2y2-b2y1)
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


class YOLOv1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, pred, target):
        batch_size = pred.shape[0]

        pred_box1 = pred[..., 0:5]
        pred_box2 = pred[..., 5:10]
        pred_cls = pred[..., 10:]

        true_box = target[..., 0:5]
        true_cls = target[..., 10:]

        obj_mask = target[..., 4] > 0
        noobj_mask = target[..., 4] == 0

        iou1 = iou(pred_box1[..., :4], true_box[..., :4])
        iou2 = iou(pred_box2[..., :4], true_box[..., :4])

        # 不再用torch.where，改用索引分别计算两组损失，完全避开原地张量选择
        iou_max = torch.max(iou1, iou2)
        mask_box1 = (iou1 >= iou2) & obj_mask
        mask_box2 = (iou2 > iou1) & obj_mask

        # Box1负责的目标损失
        # 坐标
        box1_xy_loss = torch.sum(
            (pred_box1[mask_box1, 0:2] - true_box[mask_box1, 0:2])**2)
        box1_wh_loss = torch.sum((torch.sqrt(torch.abs(
            pred_box1[mask_box1, 2:4])+1e-6) - torch.sqrt(true_box[mask_box1, 2:4]+1e-6))**2)
        # 置信度
        box1_conf_loss = torch.sum(
            (pred_box1[mask_box1, 4] - true_box[mask_box1, 4])**2)

        # Box2负责的目标损失
        box2_xy_loss = torch.sum(
            (pred_box2[mask_box2, 0:2] - true_box[mask_box2, 0:2])**2)
        box2_wh_loss = torch.sum((torch.sqrt(torch.abs(
            pred_box2[mask_box2, 2:4])+1e-6) - torch.sqrt(true_box[mask_box2, 2:4]+1e-6))**2)
        box2_conf_loss = torch.sum(
            (pred_box2[mask_box2, 4] - true_box[mask_box2, 4])**2)

        loss_coord = box1_xy_loss + box1_wh_loss + box2_xy_loss + box2_wh_loss
        loss_obj = box1_conf_loss + box2_conf_loss

        # 无物体置信损失
        loss_noobj = self.lambda_noobj * \
            (torch.sum(pred_box1[noobj_mask, 4]**2) +
             torch.sum(pred_box2[noobj_mask, 4]**2))

        # 分类损失
        loss_cls = torch.sum((pred_cls[obj_mask] - true_cls[obj_mask])**2)

        total_loss = self.lambda_coord * loss_coord + loss_obj + loss_noobj + loss_cls
        return total_loss / batch_size
