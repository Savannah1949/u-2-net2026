import os
from skimage import io
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import glob
import warnings

# 导入当前模型
from model.u2net import U2NETP
from data_loader import RescaleT, ToTensorLab, SalObjDataset

warnings.filterwarnings("ignore")

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi + 1e-8)
    return dn

def save_output(image_name, pred, d_dir):
    predict = pred.squeeze().cpu().data.numpy()

    im = Image.fromarray((predict * 255).astype('uint8')).convert('L')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)

    # resize 回原图大小
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.LANCZOS)

    im_idx = os.path.splitext(img_name)[0]
    imo.save(os.path.join(d_dir, im_idx + '.png'))

def main():
    model_name = 'U2NETP'

    image_dir = r"E:\U-2-Net-master\test_data\road"
    prediction_dir = r"E:\U-2-Net-master\test_data\road_results"
    model_path = r"E:\U-2-Net-master\saved_models\U2NETP\U2NETP_epoch_90.pth"  # 先测90轮

    os.makedirs(prediction_dir, exist_ok=True)

    img_name_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    print(f"找到 {len(img_name_list)} 张测试图片")

    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([
            RescaleT(384),
            ToTensorLab(flag=0)
        ])
    )

    test_salobj_dataloader = DataLoader(
        test_salobj_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"正在加载模型: {model_name}")
    net = U2NETP(3, 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    print("开始推理...")
    with torch.no_grad():
        for i_test, data_test in enumerate(test_salobj_dataloader):
            img_path = img_name_list[i_test]
            img_filename = os.path.basename(img_path)

            inputs_test = data_test['image'].to(device).float()

            # 当前模型返回 8 个输出
            d0, d1, d2, d3, d4, d5, d6, edge = net(inputs_test)

            # 只保存主输出 d0
            pred = d0[:, 0, :, :]
            pred = normPRED(pred)

            save_output(img_path, pred, prediction_dir)
            print(f"Done: {img_filename}")

            del d0, d1, d2, d3, d4, d5, d6, edge

    print(f"所有结果已保存至: {prediction_dir}")

if __name__ == "__main__":
    main()