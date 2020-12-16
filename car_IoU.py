import glob
import numpy as np
import torch
import os
import cv2
import imageio

def IoU():
    predicts_path = glob.glob('../dataset/valid_predict/*')
    # labels_path = glob.glob('data/test/opencv_predict_mask/*.png')
    predicts_path.sort()
    IoUs = []
    for predict_path in predicts_path:
        # predict_path = label_path.replace("label","opencv_predict_mask")
        label_path = predict_path.replace("valid_predict","train_masks").replace('.jpg','_mask.gif')
        

        predict = cv2.imread(predict_path)
        label = imageio.mimread(label_path)
        label = label[0]
        # print(predict.shape, label.shape,type(label))
        label = cv2.resize(label,(959,640))



        predict = cv2.cvtColor(predict, cv2.COLOR_BGR2GRAY)
        # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # cv2.imshow(predict_path,predict)
        # cv2.imshow(label_path,label)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        overlap = np.logical_and(predict,label)
        union = np.logical_or(predict,label)
        IoUs.append(np.sum(overlap)/np.sum(union))
        # print(predict_path,np.sum(overlap)/np.sum(union))
    
    IoU = np.mean(IoUs)
    # print(IoU)
    return IoU   

if __name__ == "__main__":
    # predicts_path = glob.glob('data/test/predict/*.png')
    print(IoU())
