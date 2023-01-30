"""
https://blog.csdn.net/sinat_29047129/article/details/103642140
https://www.cnblogs.com/Trevo/p/11795503.html
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import os
from PIL import Image

__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        # 混淆矩阵n*n，初始值全0
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    # Accuracy = (TP + TN) / (TP + TN + FP + TN)
    def pixelAccuracy(self):
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        classAcc = np.diag(self.confusionMatrix) / np.maximum(self.confusionMatrix.sum(axis=1), 1)
        return classAcc

    # Precision = TP / (TP + FP)
    def Precision_Score(self):
        classAcc = self.classPixelAccuracy()
        Precision = classAcc[0]
        return Precision

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    # MIoU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.maximum(np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix), 1)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    # Recall = TP / (TP + FN)
    def Recall(self):
        # np.sum(self.confusionMatrix, axis=0) 代指 [TP+FN FP+TN]
        # np.diag(self.confusionMatrix) 代指 [TP TN]
        # recall0 = [TP / (TP + FN)  TN / (FP+TN)]
        # recall1 = recall0[0] = TP / (TP + FN)
        recall0 = np.diag(self.confusionMatrix) / (np.sum(self.confusionMatrix, axis=0))
        recall1 = recall0[0]
        return recall1

    # F1_Score
    def F1_Score(self):
        Precision = self.Precision_Score()
        Recall = self.Recall()
        F1 = (2 * Precision * Recall) / (Precision + Recall)
        return F1

    # Dice
    def Dice(self):
        dice_coef = 2 * np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0))
        return dice_coef

    def genConfusionMatrix(self, imgPredict, imgLabel):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype(int) + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # fwIoU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def evaluate1(pre_path, label_path):
    acc_list = []
    pre_list = []
    macc_list = []
    mIoU_list = []
    fwIoU_list = []
    Recall_list = []
    F1_list = []
    Dice_list = []

    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)

    for i, p in enumerate(pre_imgs):
        imgPredict = Image.open(pre_path + p)
        imgPredict = np.array(imgPredict)
        imgLabel = Image.open(label_path + lab_imgs[i])
        imgLabel = np.array(imgLabel)

        metric = SegmentationMetric(2)
        metric.addBatch(imgPredict, imgLabel)
        acc = metric.pixelAccuracy()
        pre = metric.Precision_Score()
        macc = metric.meanPixelAccuracy()
        mIoU = metric.meanIntersectionOverUnion()
        fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
        Recall = metric.Recall()
        F1 = metric.F1_Score()
        Dice = metric.Dice()

        acc_list.append(acc)
        pre_list.append(pre)
        macc_list.append(macc)
        mIoU_list.append(mIoU)
        fwIoU_list.append(fwIoU)
        Recall_list.append(Recall)
        F1_list.append(F1)
        Dice_list.append(Dice)

    return acc_list, pre_list, macc_list, mIoU_list, fwIoU_list, Recall_list, F1_list, Dice_list


def evaluate2(pre_path, label_path):
    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)

    metric = SegmentationMetric(2)
    for i, p in enumerate(pre_imgs):
        imgPredict = Image.open(pre_path + p)
        imgPredict = np.array(imgPredict)
        imgLabel = Image.open(label_path + lab_imgs[i])
        imgLabel = np.array(imgLabel)

        metric.addBatch(imgPredict, imgLabel)

    return metric

if __name__ == '__main__':
    pre_path = 'C:/Desktop/pre/'
    label_path = 'C:/Desktop/label/'
    
    acc_list, pre_list, macc_list, mIoU_list, fwIoU_list, Recall_list, F1_list, Dice_list = evaluate1(pre_path, label_path)
    print('Final: Accuracy={:.2f}%, Precision={:.2f}%, mAcc={:.2f}%, mIoU={:.2f}%, fwIoU={:.2f}%, Recall={:.2f}%, F1_score={:.2f}%, Dice={:.2f}%'
          .format(np.mean(acc_list) * 100, np.mean(pre_list) * 100, np.mean(macc_list) * 100,
                  np.mean(mIoU_list) * 100, np.mean(fwIoU_list) * 100, np.mean(Recall_list) * 100, np.mean(F1_list) * 100,
                  np.mean(Dice_list) * 100))