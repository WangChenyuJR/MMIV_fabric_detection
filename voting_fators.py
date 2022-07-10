"""
This file is to give all the factors of voting, and their calculation functions
将处理过后的图片分成多个子图，每个子图两两比较，若差异较大，则同时为两个子图记一次瑕疵可能点，即投票；
票数越多，越容易出现瑕疵。再将该瑕疵子图的位置坐标保留并凸显（由于背景一致性高）
可以比较非格状图的频谱、色调空间、结构、边缘、灰度程度

子图mask大小、边界如何确定-
"""
import os
import cv2 as cv
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import sys

pixel_size = 600*800
#gray-median
def gray_mean(img, pixel_size):
    mean_gray = round(np.mean(img))
    return  mean_gray

#contrast mean
"""
对比度包含四邻近和八邻近、十六邻近等
"""
#4-neighbor Contrast
def contrast_mean_4neighbor(img, pixel_size):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#转灰度
    height, width = img.shape
    #类卷积，邻近法需要边界出向外扩展一个像素点
    img_ext = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_REPLICATE) / 1.0
    rows_ext, cols_ext = img_ext.shape
    pixel_diff = 0.0
    for i in range(1, rows_ext - 1):
        for j in range(1, cols_ext - 1):
            pixel_diff += ((img_ext[i, j] - img_ext[i, j - 1])**2
                         + (img_ext[i, j] - img_ext[i + 1, j])**2
                         + (img_ext[i, j] - img_ext[i - 1, j])**2
                         + (img_ext[i, j] - img_ext[i, j + 1])**2)
    num_of_square = 4*(height - 2)*(width - 2) + 3*(2*(height - 2) + 2*(width - 2)) + 2*4
    contrast_mat = pixel_diff / num_of_square
    return contrast_mat
#8-neighbor Contrast
def contrast_mean_8neighbor(img, pixel_size):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)#转灰度
    height, width = img.shape
    #类卷积，邻近法需要边界出向外扩展一个像素点
    img_ext = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_REPLICATE) / 1.0
    rows_ext, cols_ext = img_ext.shape
    pixel_diff = 0.0
    for i in range(1, rows_ext - 1):
        for j in range(1, cols_ext - 1):
            pixel_diff += ((img_ext[i, j] - img_ext[i, j - 1])**2
                         + (img_ext[i, j] - img_ext[i + 1, j])**2
                         + (img_ext[i, j] - img_ext[i - 1, j])**2
                         + (img_ext[i, j] - img_ext[i, j + 1])**2
                         + (img_ext[i, j] - img_ext[i - 1, j - 1])**2
                         + (img_ext[i, j] - img_ext[i + 1, j - 1])**2
                         + (img_ext[i, j] - img_ext[i - 1, j + 1])**2
                         + (img_ext[i, j] - img_ext[i + 1, j + 1])**2)
    num_of_square = 8*(height - 2)*(width - 2) + 5*(2*(height - 2) + 2*(width - 2)) + 3*4
    contrast_mat = pixel_diff / num_of_square
    return contrast_mat
#Saturation，饱和度计算，每个子图的平均饱和度
def saturation_mean(rgb_img, pixel_size):
    img = rgb_img * 1.0
    #min和max代表RGB空间中的R、G、B颜色值中的最小最大值，范围为0-1的实数
    img_min = img.min(axis = 2)
    img_max = img.max(axis = 2)

    delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    Light_Value = value / 2.0#亮度

    #s = L<0.5 ? s1 : s2
    mask_1 = Light_Value < 0.5
    satura1 = delta / value
    satura2 = delta / (2 - value)
    Saturation_Value = satura1 * mask_1 + satura2 * (1 - mask_1)#饱和度

    Saturation_Value_Sum = reduce(lambda x, y:x+y, (reduce(lambda x, y :x+y, Saturation_Value)))#所有点的饱和度加起来
    Saturation_Value_Median = (Saturation_Value_Sum / pixel_size) * 1000#饱和度平均值
    return Saturation_Value_Median
im = cv.imread("E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\JPEGImages\\000001.jpg")
# print(saturation_mean(im, pixel_size))
# print(contrast_mean_4neighbor(im))
# print(contrast_mean_8neighbor(im))
cut_test = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\cut_test"
def read_directory(directory_name):
    aa = 0
    for filename in os.listdir(r"" + directory_name):
        img = cv.imread(directory_name + "/" + filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        saturation_mean(img)
        aa += 1
    print(aa)
        #print(saturation_mean(img))
"""
这部分是iou计算
"""
#box1:预测框坐标 box2：真实框坐标
def IOU_compute(box1, box2):
    """
    :param box1:[x1,y1,x2,y2] 左上角的坐标与右下角的坐标
    :param box2:[x1,y1,x2,y2]
    :return: iou_ratio--交并比
    """
    width1 = abs(box1[2] - box1[0])
    height1 = abs(box1[1] - box1[3]) # 这里y1-y2是因为一般情况y1>y2，为了方便采用绝对值
    width2 = abs(box2[2] - box2[0])
    height2 = abs(box2[1] - box2[3])
    x_max = max(box1[0],box1[2],box2[0],box2[2])
    y_max = max(box1[1],box1[3],box2[1],box2[3])
    x_min = min(box1[0],box1[2],box2[0],box2[2])
    y_min = min(box1[1],box1[3],box2[1],box2[3])
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    if iou_width <= 0 or iou_height <= 0:
        iou_ratio = 0
    else:
        iou_area = iou_width * iou_height # 交集的面积
        box1_area = width1 * height1
        box2_area = width2 * height2
        iou_ratio = iou_area / (box1_area + box2_area - iou_area) # 并集的面积
    return iou_ratio