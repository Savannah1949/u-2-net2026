import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import glob
import warnings
import torch.nn.functional as F

# 保持原模型和 DataLoader 导入不变
from model.u2net import U2NETP
from data_loader import RescaleT, RandomCrop, ToTensorLab, SalObjDataset

warnings.filterwarnings('ignore')


# ------- 1. 损失函数优化 (SSIM 仅作用于主输出以提升速度) --------

def ssim(img1, img2, window_size=11, size_average=True):
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

def label_to_edge(label):
    # label: [B,1,H,W]
    dilated = F.max_pool2d(label, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-label, kernel_size=3, stride=1, padding=1)
    edge = dilated - eroded
    edge = (edge > 0).float()
    return edge

def focal_tversky_loss(pred, target, alpha=0.3, beta=0.7, gamma=1.5, eps=1e-6):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = torch.pow((1 - tversky), gamma)
    return loss.mean()

def muti_loss_fusion(preds, target):
    bce_loss = nn.BCELoss(reduction='mean')

    # 前7个是分割输出，第8个是边缘输出
    seg_preds = preds[:7]
    edge_pred = preds[7]

    loss_sum = 0.0
    main_loss = 0.0

    # 分割损失
    for i, pred in enumerate(seg_preds):
        loss_bce = bce_loss(pred, target)
        loss_tversky = focal_tversky_loss(pred, target)

        i_loss = loss_bce + loss_tversky

        # 主输出 d0 再加一点 SSIM，保留结构一致性
        if i == 0:
            loss_ssim = 1 - ssim(pred, target)
            i_loss = i_loss + 0.1 * loss_ssim
            main_loss = i_loss

        loss_sum += i_loss

    # 边缘监督
    edge_gt = label_to_edge(target)
    edge_loss = bce_loss(edge_pred, edge_gt)

    loss_sum = loss_sum + 0.3 * edge_loss
    main_loss = main_loss + 0.3 * edge_loss

    return main_loss, loss_sum


# ------- 2. 训练主程序 --------

if __name__ == '__main__':
    # 配置参数
    model_name = 'U2NETP'
    epoch_num = 100
    batch_size_train = 4
    lr = 0.0001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 路径设置
    data_dir = os.path.join(os.getcwd(), 'train_data' + os.sep)
    tra_image_dir = os.path.join('DUTS', 'DUTS-TR', 'im_aug' + os.sep)
    tra_label_dir = os.path.join('DUTS', 'DUTS-TR', 'gt_aug' + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 获取数据列表
    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*.jpg')
    tra_lbl_name_list = [img_path.replace('im_aug', 'gt_aug').replace('_sat.jpg', '_mask.png') for img_path in
                         tra_img_name_list]

    print(f"--- Total images found: {len(tra_img_name_list)} ---")

    # 构建 DataLoader - 优化参数
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(384),
            RandomCrop(352),
            ToTensorLab(flag=0)
        ])
    )

    # 增加 num_workers，并添加 prefetch_factor 加快预取
    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=10,  # 根据你的 CPU 逻辑核心数调整，建议 4-12 之间
        pin_memory=True,  # 必须开启，加速内存到显存的拷贝
        prefetch_factor=4 ,# 预取几个 batch 的数据
        drop_last = True
    )

    # 初始化模型
    net = U2NETP(3, 1)
    net.to(device)

    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    print(f"--- Start Training {model_name} on {device} ---")

    for epoch in range(epoch_num):
        net.train()
        running_loss = 0.0
        running_tar_loss = 0.0

        for i, data in enumerate(salobj_dataloader):
            inputs, labels = data['image'], data['label']

            # 异步拷贝到 GPU
            inputs = inputs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float()

            optimizer.zero_grad()

            # 前向传播
            preds = net(inputs)

            # 计算 Loss
            loss_main, loss_all = muti_loss_fusion(preds, labels)

            # 反向传播
            loss_all.backward()

            # 梯度裁剪：防止训练不稳
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)

            optimizer.step()

            # 统计
            running_loss += loss_all.item()
            running_tar_loss += loss_main.item()

            if (i + 1) % 10 == 0:
                print(
                    f"[Epoch: {epoch + 1:3d}/{epoch_num}, Batch: {i + 1:5d}] Loss: {running_loss / (i + 1):.4f}, Main: {running_tar_loss / (i + 1):.4f}")

        # 更新学习率
        scheduler.step()

        # 定期保存
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(model_dir, f"{model_name}_epoch_{epoch + 1}.pth")
            torch.save(net.state_dict(), save_path)
            print(f">>> Saved: {save_path}")

    print("Training Done.")