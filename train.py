import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model.unet_model import UNet, UNet_light
from model.unetv2_parts import Unet_v2
from utils.dataset import Car_Image_Loader
import datetime
from torch import optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model.dice_loss import DiceLoss
from model.focal_loss import BinaryFocalLoss,FocalLoss_Ori



def train_net(net, device, in_channel, data_path, epochs=40, batch_size=1, lr=0.0001):
    time_stamp = datetime.datetime.now().strftime("%m%d%H%M")
    
    if not os.path.exists("saved_models/{}".format(time_stamp)):
        os.makedirs("saved_models/{}".format(time_stamp))

    # tensorboard writer
    writer = SummaryWriter('runs/experiment_{}'.format(time_stamp))
    # 加载训练集
    car_dataset = Car_Image_Loader(in_channel,data_path)
    train_loader = torch.utils.data.DataLoader(dataset=car_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # criterion = DiceLoss()
    # criterion = BinaryFocalLoss(alpha=[0.25,0.75],gamma=2)
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 累计更新次数
    step = 0  
    # 记录最好模型是哪个epoch
    best_epoch = -1
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        batch_count = 0
        losses = []
        # 按照batch_size开始训练
        for image, label in train_loader:
            batch_count += 1
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果

            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            # print(loss)
            print('Epoch', epoch, '| Batch',batch_count, '| Loss/train', loss.item())
            writer.add_scalar('training loss',loss.item(),step)
            losses.append(loss.item())
            step+=1 
            # 更新参数
            loss.backward()
            optimizer.step()
        
        # 保存loss值最小的网络参数
        loss_avg = np.mean(losses)
        print('Epoch', epoch, '| Loss/train', loss_avg)
        if loss_avg < best_loss:
            best_model_path = 'saved_models/{}/best_model.pth'.format(time_stamp)
            best_loss = loss_avg
            torch.save(net.state_dict(), best_model_path)
            best_epoch = epoch
        # 每20个epoch保存一次参数
        if (epoch+1)%3==0:
            model_path = 'saved_models/{}/epoch_{:03d}.pth'.format(time_stamp,epoch+1)
            torch.save(net.state_dict(), model_path)

    print("Training Finished! Best epoch: ", best_epoch)

if __name__ == "__main__":
    with torch.cuda.device(1):                  # 使用GPU1
        print(torch.cuda.is_available())        # 查看cuda是否可用
        print(torch.cuda.device_count())        # 返回GPU数目
        print(torch.cuda.get_device_name(1))    # 返回GPU名称，设备索引默认从0开始
        print(torch.cuda.current_device())      # 返回当前设备索引 

        # 输入图片通道数
        in_channel = 3
        # 选择设备，有cuda用cuda，没有就用cpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(device)
        # 加载网络，图片通道3，分类为1。
        # net = UNet(n_channels=in_channel, n_classes=1)
        net = UNet_light(n_channels=in_channel, n_classes=1)
        # net = Unet_v2(in_channels=3,n_classes=1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 指定训练集地址，开始训练
        data_path = "../dataset/"
        train_net(net, device, in_channel, data_path, epochs=12, batch_size=8)