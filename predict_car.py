import glob
import numpy as np
import torch
import os
import cv2
import pickle
from model.unet_model import UNet, UNet_light
from model.unetv2_parts import Unet_v2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm 

def predict(in_channel, model_path, data_path, light=False):
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    if light:
        net = UNet_light(n_channels=in_channel, n_classes=1)
    else:
        net = UNet(n_channels=in_channel, n_classes=1)
    # net = Unet_v2(in_channels=in_channel, n_classes=1)
    
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load(model_path, map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    with open(os.path.join(data_path, 'valid.pkl'),"rb") as f:
        valid = pickle.load(f)

    tests_path = [os.path.join(data_path, path) for path in valid]
    # 遍历素有图片
    for test_path in tqdm(tests_path):
        # 保存结果地址
        save_res_path = test_path.replace("train","valid_predict")
        # 读取图片
        img = cv2.imread(test_path)

        img_shape = img.shape

        if in_channel==1:
            # 转为灰度图
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 转为batch为1，通道为1，大小为512*512的数组
            img = transforms.ToTensor()(img)
        else:
            # 转为batch为1，通道为3，大小为512*512的数组
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            PIL_image = Image.fromarray(img)
            transform = transforms.Compose([
                transforms.Resize((img_shape[0]//2, img_shape[1]//2)),
                transforms.ToTensor(), #数据归一化到[0,1],输入通道转换在前
                transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)), # 数据归一化到[-1,1]
            ])
            img = transform(PIL_image)
            img = img.unsqueeze(0) # 加入batch维度

        # # 转为tensor
        # img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img = img.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片

        cv2.imwrite(save_res_path, pred)

        # plt.figure()
        # plt.imshow(pred)
        # plt.title(test_path)
        # plt.show()


if __name__ == "__main__":
    # model_path = "saved_models/BCE_Loss_1/epoch_240.pth"
    model_path = "saved_models/12150404/best_model.pth"
    # model_path = "saved_models/BCE_DICE_Loss_1/epoch_240.pth"
    data_path = "../dataset/"
    in_channel = 3
    with torch.cuda.device(1):
        predict(in_channel, model_path, data_path)