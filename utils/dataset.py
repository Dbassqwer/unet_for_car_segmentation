import torch
import cv2
import imageio
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import pickle

class Car_Image_Loader(Dataset):
    def __init__(self, in_channel, data_path):
        self.in_channel = in_channel
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        # self.imgs_path = glob.glob(os.path.join(data_path, 'train/*.jpg'))

        with open(os.path.join(data_path, 'train_split.pkl'),"rb") as f:
            train_split = pickle.load(f)
        
        train_split = [os.path.join(data_path, path) for path in train_split]
        self.imgs_path = train_split

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为0水平翻转，1垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('train', 'train_masks').replace('.jpg','_mask.gif')
        # 读取训练图片和标签图片
        # print(image_path, label_path)
        image = cv2.imread(image_path)
        # label = cv2.imread(label_path)

        label = imageio.mimread(label_path)
        label = np.array(label[0])



        # 将image的BGR通道顺序转换为RGB通道顺序，将label数据转为单通道的图片
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        # print(image.shape) # (1280, 1918,3)
        # print(label.shape) # (1280, 1918)

        img_shape = image.shape

        # 归一化处理
        # image = image / 255.
        # 处理标签，将像素值为255的改为1
        # if label.max() > 1:
        #     label = label / 255.
        # 随机进行数据增强，为0做水平翻转为2时不做处理
        flipCode = random.choice([1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)

        PIL_image = Image.fromarray(image)
        # 输入图片经过transform转换
        transform = transforms.Compose([
            transforms.Resize((img_shape[0]//2, img_shape[1]//2)),
            transforms.ToTensor(), #数据归一化到[0,1],输入通道转换在前
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)), # 数据归一化到[-1,1]
        ])
        image = transform(PIL_image)

        PIL_label = Image.fromarray(label)
        label = transforms.Resize((img_shape[0]//2, img_shape[1]//2))(PIL_label)
        label = transforms.ToTensor()(label)

        # print(image.shape, label.shape)
        # 将image数据转为输入通道数的图片
        # if self.in_channel == 1:
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #     image = image.reshape(1, image.shape[0], image.shape[1])
        # else:
        #     # image = image.reshape(3, image.shape[0], image.shape[1])
        #     image = np.transpose(image,(2,0,1))
        
        # label = label.reshape(1, label.shape[0], label.shape[1])
        
        # print(image.shape,label.shape)

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    car_dataset = Car_Image_Loader(3,"../../dataset/")
    print("数据个数：", len(car_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=car_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    for image, label in train_loader:
        # img_arr = image.squeeze().numpy()
        # label_arr = label.squeeze().numpy()
        # img_arr.reshape((512,512,3))
        print(image.shape,label.shape) # 为什么不能reshape
        image = np.squeeze(image.numpy())
        
        # image = np.transpose(image,[1,2,0])
        # print(type(image),image.shape)
        # image = (image / 2) + 0.5 # 反归一化
        # plt.figure()
        # plt.imshow(image)
        # plt.show()

        label = np.squeeze(label.numpy())
        # print(label.shape)
        # plt.figure()
        # plt.imshow(label)
        # plt.show()


        print(image.max(),image.min())
        print(label.max(),label.min())
        # print(image.max())
        # cv2.imshow("image",image)
        # cv2.imshow("label",label_arr)
        # cv2.waitKey(0)