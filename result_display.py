"""
该.py文件是对数据集中所有图片进行裁剪操作，为了试验不同大小的子图像效果
"""
import os
import cv2
import csv
import math
import heapq
import random
import linecache
import numpy as np
import numba as nb
import inspect, re
import pandas as pd
import elim_low_freq
from numba import jit
from PIL import Image
from tqdm import tqdm
from numba import cuda
import matplotlib.pyplot as plt
import xml.dom.minidom as xmldom
from openpyxl import load_workbook
from pycallgraph import PyCallGraph
import matplotlib.patches as patches
from evaluation import evaluation_judge
from pycallgraph.output import GraphvizOutput
from multiprocessing.dummy import Pool as ThreadPool
from voting_fators import gray_mean, contrast_mean_4neighbor, contrast_mean_8neighbor, saturation_mean

save_path = 'E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\save.txt'
source_path = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted"
source_path_1 = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect"
image_path = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\JPEGImages"
hist_path = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\Hist_avg_txt"#所有子图的灰度平均值txt
gray_median_dist_path = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\gray_distri_plt"#每张图的子图灰度距离分布
result_path = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\result_image"
label_path = "E:\\LIC\\CV_detection\\faster_rcnn\\fabric-defect\\fabric-annoted\\Annotations"#所有缺陷原图的标注文件夹（包含缺陷框）

#改进的ransac算法里epsilon和threshold的取值范围
zsRansac_epsilon_range = 5
zsRansac_threshold_range = 5

#例图获取所有图像的大小和尺寸、子图数量等
example_img = cv2.imread(image_path + "\\000001.jpg")
size = example_img.shape
sub_size = 100#子图大小
#print(size)
height = size[0]
width = size[1]
pixel_size = height * width
#print(height)
#print(width)

sub_total = int((height / sub_size) * (width / sub_size))
row = height / sub_size
col = width / sub_size
subimg_num = int(row * col)#子图初始数量
#save_file = open(hist_path, 'a')


matrix_subimg_diff_abs = np.zeros((sub_total, sub_total))#灰度平均值列表值两两相减后的绝对值差值空矩阵

list_img_name = []#图片文件名的列表
list_xml_name = []#标注文件列表
def get_img_list(path):
    filelist = os.listdir(path)
    for filename in filelist:
        #filename = path + filename
        list_img_name.append(filename)

def get_xml_list(path):
    filelist = os.listdir(path)
    for filename in filelist:
        list_xml_name.append(filename)

"""
分别获取数据集的图片文件名列表和标注文件列表
"""
get_img_list(image_path + '\\')
get_xml_list(label_path + '\\')

img_num_total = 0
    # Read the folder of the pictures to be processed, and save all the picture names in a folder
f = open(save_path, 'w')
for filename in os.listdir(image_path):
    f.write(str(filename))
    f.write('\n')
    img_num_total+=1
f.close()

#该函数可以传入函数或变量，返回对应的函数名和变量名
def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
        return m.group(1)

#检查当前有无该文件路径，没有该路径则新建
def check_mkdir(path):
    mask_folder = os.path.exists(path)
    if not mask_folder:
        os.makedirs(path)

#将数据保存在csv文件中
def save_csv_single(save_path, save_file_name, save_data):
    check_mkdir(save_path)
    dataframe = pd.DataFrame({'data':save_data})
    dataframe.to_csv(save_path + "\\" +save_file_name, index=False)

#多列数据
def save_csv_multi(save_path, save_file_name, save_data):
    check_mkdir(save_path)
    for i in range(len(save_data)):
        file = open(save_path + "\\" +save_file_name, 'a', newline='')
        writer = csv.writer(file)
        writer.writerow(save_data)
        file.close()

#由于只找所有值中最符合的中间值，所以定义斜率为0的ransac
def ransac_threshold_slope0(list, point_total, max_val, iters, epsilon, threshold):
    pregiven_line = np.linspace(max_val / 2, max_val / 2, point_total)#自动生成一条水平的线，用于拟合
    best_std_val = max_val / 2#std_val截距
    pre_total = 0
    for i in range(iters):
        #随机选择所有点中的两个样本点进行迭代
        sample_index = random.sample(range(point_total), 2)

        #预给的值
        pre_1 = pregiven_line[sample_index[0]]
        pre_2 = pregiven_line[sample_index[1]]

        #实际值,原始值是二维列表，实际只需要取二维列表的第二个元素
        real_1 = list[sample_index[0]][1]
        real_2 = list[sample_index[1]][1]

        #omega = (real_2 - real_1) / (pre_2 - pre_1)
        slope = 0
        std_val = real_1 - slope * pre_1

        total_in = 0  # 内点计数器
        for index in range(point_total):
            y_estimate = slope * pregiven_line[index] + std_val
            ind = list[index][1]
            if abs(y_estimate - ind) < epsilon:  # 符合内点条件
                total_in += 1

            if total_in > pre_total:  # 记录最大内点数与对应的参数
                pre_total = total_in
                best_slope = slope
                best_std_val = std_val

            if total_in > point_total * threshold:  # 内点数大于设定的阈值，跳出循环
                break
        #print("Iterate {} times, slope = {}, standard value = {}".format(i, best_slope, best_std_val))

        #list_ext_point.append(best_std_val)
    #标准方差列表
    std_var_list = [[] for i in range(point_total)]
    std_var_sum = 0

    #阈值列表
    threshold_list = []
    for i in range(point_total):
        #标准方差std_var
        ceta = list[i]
        ceta = ceta[1]
        std_var = abs((ceta - best_std_val)) / point_total
        std_var_sum += ceta - best_std_val#所有点与标准值差值的总和
        std_var_list[i].extend([i, std_var])
    #                      标准方差和、         平均标准方差、                   平均标准方差的开方、                 平均标准方差的平方
    threshold_list.extend([std_var_sum, std_var_sum / point_total, np.sqrt(std_var_sum / point_total), (std_var_sum / point_total)**2])

    return threshold_list, std_var_list

#定义逐个计算每个点的值（std_var_list）与标准值不同算术下的阈值相比得到的结果
#输入形参:th_list代表不同的阈值列表，std_list代表所有点计算标准方差后的列表,rate是另三种计数方法的参数，该函数可以忽略
def calculate_std_diff_multi(th_list, std_list):
    all_exterior_point_mat = []#不同阈值下得到的外点集矩阵
    for index_1 in range(len(th_list)):
        single_exterior_point = []#单个阈值所得到的外点集
        for index_2 in range(len(std_list)):
            if th_list[index_1] == None:
                single_exterior_point.append([])
                continue
            if std_list[index_2][1] > th_list[index_1]:
                single_exterior_point.append(std_list[index_2][0])

        all_exterior_point_mat.append(single_exterior_point)
    return all_exterior_point_mat

#单个threshold的计算
def calculate_std_diff(threshold, std_list):
    single_exterior_point = []#单个阈值所得到的外点集
    for index_2 in range(len(std_list)):
        if std_list[index_2][1] > threshold:
            single_exterior_point.append(std_list[index_2][0])
    return single_exterior_point

"""
根据标准值和实际值之间的差值列表生成得分矩阵
"""
def score_matrix(std_list):
    score_mat = [None] * len(std_list)
    for i in range(len(std_list)):
        score_mat[i] = [0] * len(std_list)
    for i in range(len(std_list)):
        for j in range(len(std_list)):
            score_mat[i][j] = std_list[i][1] - std_list[j][1]
    return score_mat

#borda计算外点方法，输入形参为rate_of_extpnt:外点值占比，th_list, std_list：每个点和标准值的差值
def borda_count(rate_of_extpnt, std_list):
    all_exterior_point_mat = []#不同阈值下得到的外点集矩阵
    single_exterior_point = []#单个阈值所得到的的外点集
    borda_score_list = []

    for index in range(len(std_list)):
        score_mat = score_matrix(std_list)
        sum = 0
        for i in range(len(score_mat[index])):
            sum += score_mat[index][i]
        borda_score_list.append(sum)

    num = int(rate_of_extpnt * len(score_mat[index]))#最大的值的数量
    len_borda_score_list = len(borda_score_list)

    #如果num=0就不需要进行计算，直接输出空值
    if num > 0:
        index_max_num = []
        val_max_num = []#最大的几个值
        #先将得分列表里的前num个值直接看做整个得分列表的最大值并保存到index和val里
        for i in range(num):
            index_max_num.append(i)#最大值的下标列表
            val_max_num.append(borda_score_list[i])#最大值的值列表

        #从第num开始，找比现在index和val里更大的值并替换
        for i in range(num, len_borda_score_list - 1):
            min_val = heapq.nsmallest(1, val_max_num)
            index = list(map(val_max_num.index, min_val))

            if borda_score_list[i] > val_max_num[index[0]]:
                val_max_num[index[0]] = borda_score_list[i]
                index_max_num[index[0]] = i

        index_max_num = sorted(index_max_num)
    elif num == 0 :
        index_max_num = []
    return borda_score_list, index_max_num

#copeland计算外点方法
def copeland_count(rate_of_extpnt, std_list):
    copeland_score_list = []
    for ind in range(len(std_list)):
        score_mat = score_matrix(std_list)
        sum = 0
        for i in range(len(score_mat[ind])):
            if score_mat[ind][i] > 0:
                sum += 1
            elif score_mat[ind][i] == 0:
                sum += 0
            elif score_mat[ind][i] < 0:
                sum += -1
        copeland_score_list.append(sum)

    num = int(rate_of_extpnt * len(score_mat[ind]))  # 最大的值的数量
    if num > 0:
        index_max_num = []
        val_max_num = []  # 最大的几个值

        for i in range(num):
            index_max_num.append(i)  # 最大值的下标列表
            val_max_num.append(copeland_score_list[i])  # 最大值的值列表

        # 从第num开始，找比现在index和val里更大的值并替换
        for i in range(num, len(copeland_score_list)):
            min_val = heapq.nsmallest(1, val_max_num)
            index = list(map(val_max_num.index, min_val))

            if copeland_score_list[i] > val_max_num[index[0]]:
                val_max_num[index[0]] = copeland_score_list[i]
                index_max_num[index[0]] = i

        index_max_num = sorted(index_max_num)
    elif num == 0 :
        index_max_num = []
    return copeland_score_list, index_max_num

#maximin计算外点方法
def maximin_count(rate_of_extpnt, std_list):
    maximin_score_list = []
    #for ind in range(len(std_list)):
    score_mat = score_matrix(std_list)

    #直接取每一行的最大值
    for i in range(len(std_list)):
        temp_list = score_mat[i]
        max_num = heapq.nlargest(1, temp_list)
        maximin_score_list.append(max_num[0])

    num = int(rate_of_extpnt * len(std_list))
    if num > 0:
        index_max_num = []
        val_max_num = []  # 最大的几个值

        for i in range(num):
            index_max_num.append(i)  # 最大值的下标列表
            val_max_num.append(maximin_score_list[i])  # 最大值的值列表

        # 从第num开始，找比现在index和val里更大的值并替换
        for i in range(num, len(maximin_score_list)):
            min_val = heapq.nsmallest(1, val_max_num)
            index = list(map(val_max_num.index, min_val))

            if maximin_score_list[i] > val_max_num[index[0]]:
                val_max_num[index[0]] = maximin_score_list[i]
                index_max_num[index[0]] = i

        index_max_num = sorted(index_max_num)
    elif num == 0 :
        index_max_num = []

    return maximin_score_list, index_max_num

#计算外点i是原图片上的哪个子图位置
# def fig_sub_loc(subimage_location):
#     plt.imshow(Image.open(image_path + "\\%d" % list_img_name[img_num]))
#
#     sub_col = subimage_location / col - 1
#     sub_row = subimage_location % col
#     y0 = sub_col * sub_size
#     x0 = sub_row * sub_size
#     upleft = (x0, y0)
#     ax1 = plt.gca()
#     ax1.add_patch(plt.Rectangle(upleft, sub_size, sub_size, color="red", fill=False, linewidth=1))
#     # ax1.text(y0, x0, "defect", bbox={'facecolor':'red', 'alpha':0.5})
#     plt.savefig(result_path + '\\result_%d' % list_img_name[img_num])
#     print(upleft, sub_size)

"""
对每一张原图的标注文件进行处理，得到每个瑕疵在相应的子图计算模式下的编号位置，通过编号位置的对比来查看是否预测准确
"""
def get_bndbox_From_Xml_Multi_Obj(xml_file_path):
    # 读取xml文件
    count = 0
    domobj = xmldom.parse(xml_file_path)
    elementobj = domobj.documentElement
    ob = elementobj.getElementsByTagName('object')
    count += len(ob)#记该xml文件中有多少个obj对象
    bndbox = [[] for n in range(count)]#生成多目标的bndbox列表
    #统一将所有的box的外点都保存为一维数组
    outer_list = []

    sub_element_obj = elementobj.getElementsByTagName('bndbox')
    for i in range(count):
        if sub_element_obj is not None:
            #print(i)
            bndbox[i].append(int(sub_element_obj[i].getElementsByTagName('xmin')[0].firstChild.data))
            bndbox[i].append(int(sub_element_obj[i].getElementsByTagName('ymin')[0].firstChild.data))
            bndbox[i].append(int(sub_element_obj[i].getElementsByTagName('xmax')[0].firstChild.data))
            bndbox[i].append(int(sub_element_obj[i].getElementsByTagName('ymax')[0].firstChild.data))
            #print(sub_element_obj[i])
            x_min = bndbox[i][0]
            y_min = bndbox[i][1]
            x_max = bndbox[i][2]
            y_max = bndbox[i][3]
            #print(bndbox)
            # 标签高度和宽度
            # box_height = y_max - y_min + 1
            # box_width = x_max - x_min + 1
            # 左上角的点和右下角的点所在的子图位置
            left_top_point = int(math.ceil(x_min / sub_size) + math.ceil((y_min / sub_size) - 1) * col)
            right_bottom_point = int(math.ceil(x_max / sub_size) + math.ceil((y_max / sub_size) - 1) * col)

            right_remainder = right_bottom_point % col
            left_remainder = left_top_point % col

            #如果点刚好在最右侧，则余数取最大
            if right_remainder == 0:
                right_remainder = 8
            if left_remainder == 0:
                left_remainder = 8

            #如果刚好一个box框被一个子图完全包裹，则不进行以下计算，直接将该子图编号作为外点编号
            if left_top_point != right_bottom_point:
                hei_num = math.ceil(right_bottom_point / col) - math.ceil(left_top_point / col)  # 高度上标签占了几个子图格
                wid_num = abs(right_remainder - left_remainder)  # 宽度上标签占了几个子图格

                for j in range(hei_num+1):
                    list = np.arange(int(left_top_point + j * col), int(left_top_point + j * col + wid_num + 1), 1)
                    outer_list.extend(list)

            elif left_top_point == right_bottom_point:
                outer_list.extend([left_top_point])

    outer_list = sorted(outer_list)
    return outer_list
"""
计算是第几个子图，跟get_bndbox_From_Xml_Multi_Obj
"""
#def Calculate_Src_Img_Sub_Outer(bnd_box):

"""
对每一张原图进行如下操作，其中会调用以上函数以及其他源程序的函数
"""
#rate_of_extpnts = [0.02, 0.04, 0.06, 0.08,  0.1,  0.12,  0.14,  0.16,  0.18,  0.2]
#print(list_img_name)
#epsilon = 1
#threshold = 2

# evaluation_all_matrix = [[] for i in range(img_num_total)]
# p = image_path + '\\000001.jpg'
# img = cv2.imread(p)
# xml_path = label_path + '\\000001.xml'#找到标注文件
# outer_list = get_bndbox_From_Xml_Multi_Obj(xml_path)  # GroundTruth外点
# 把每个图像在不同尺寸子图下的groundtruth外点保存下来
for img_num in tqdm(range(0, img_num_total - 1), desc='Processing image ', ncols=200, leave=True):
    img = cv2.imread(image_path + '\\' + list_img_name[img_num])
    xml_path = label_path + '\\' + list_xml_name[img_num]  # 找到标注文件
    outer_list = get_bndbox_From_Xml_Multi_Obj(xml_path)  # GroundTruth外点
 #save_csv_single(save_path=groundtruth_list_path, save_file_name=file_save_name, save_data=outer_list)

    fac_val_list = [[] for i in range(subimg_num)]  # 储存不同值的list
    iterations = 0
    max_val = 0
    for i in range(0, height, sub_size):
        for j in range(0, width, sub_size):
            img_cropped = img[i:i + sub_size, j:j + sub_size]  # 将原图按照尺寸大小分为若干个子图
            # 该ij下的子图数据值
            #val = round(saturation_mean(img_cropped, pixel_size))
            #val = gray_mean(img_cropped, pixel_size)
            val = gray_mean(img_cropped, pixel_size)
            if val > max_val:
                max_val = val
            fac_val_list[iterations].extend([iterations, val])

            iterations += 1
    """
    plt.title("Distance distribution of " + '000001.jpg', fontsize=15)
    plt.axis([0, sub_total, 0, 256])
    plt.xlabel('Number of subimage', fontsize=10)
    plt.ylabel('Gray median of subimage ' + '000001.jpg', fontsize=10)
    index = np.linspace(0, sub_total-1, sub_total)
    #print(index)
    plt.scatter(index, fac_val_list, s=5)
    elim_low_freq.check_mkdir(gray_median_dist_path + '_size_%d'%sub_size)
    plt.savefig(gray_median_dist_path + '_size_%d'%sub_size + '\\' + '000001.jpg')#保存每个子图的灰度距离分布
    plt.figure()
    #plt.show()
    """
    threshold_list, std_var_list = ransac_threshold_slope0(list=fac_val_list, point_total=subimg_num, max_val=max_val, iters=1000, epsilon=3, threshold=2)
    # con4_cal_scorelist, con4_cal_predict_point_list = calculate_std_diff(threshold=con4_threshold_list[3], std_list=con4_std_var_list)
    scorelist, predict_point_list = borda_count(rate_of_extpnt=0.1, std_list=std_var_list)


"""
print(predict_point_list)
#显示检测结果
plt.imshow(Image.open(image_path + '\\000001.jpg'))

for i in range(len(predict_point_list)):
    print(predict_point_list[i])
    sub_col = predict_point_list[i] / col
    sub_row = predict_point_list[i] % col + 1
    y0 = sub_col * sub_size
    x0 = sub_row * sub_size
    upleft = (x0, y0)
    ax1 = plt.gca()
    ax1.add_patch(plt.Rectangle(upleft, sub_size, sub_size, color="red", fill=False, linewidth=1))
    # ax1.text(y0, x0, "defect", bbox={'facecolor':'red', 'alpha':0.5})
    plt.savefig(result_path + '\\result_000001.jpg')
    plt.show()
    #print(upleft, sub_size)
# 评估矩阵，包含置信度、混淆矩阵参数


# con4_cal_evaluation_matrix = evaluation_judge(groundtruth_point_list=outer_list, predict_point_list=con4_bo_predict_point_list)
# con4_cal_evaluation_all_matrix[img_num].extend(con4_bo_evaluation_matrix)
# evaluation_matrix = evaluation_judge(groundtruth_point_list=outer_list, predict_point_list=predict_point_list)
# evaluation_all_matrix[img_num].extend(evaluation_matrix)

"""

"""
if __name__ == '__main__':
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput
    with PyCallGraph(output=GraphvizOutput()):
         main_test()
"""
# methods_factors(str(list_img_name[img_num][:-4]), img, outer_list)
# list_gray_median[iterations].extend([iterations, mean_gray])#同时给list添加原始标号和对应标号子图的灰度平均值
# list_contrast4neighber[iterations].extend([iterations, mean_contrast4neighbor])#同上
# list_contrast8neighber[iterations].extend([iterations, mean_contrast8neighbor])#同上
# list_saturation[iterations].extend([iterations, mean_saturation])#同上
# print("----------------------" + str(list_img_name[img_num]) + " processes over ---------------------------")
"""
#将子图灰度平均值分布可视化
plt.title("Distance distribution of " + list_img_name[img_num], fontsize=15)
plt.axis([0, sub_total, 0, 256])
plt.xlabel('Number of subimage', fontsize=10)
plt.ylabel('Gray median of subimage ' + list_img_name[img_num], fontsize=10)
index = np.linspace(0, sub_total-1, sub_total)
#print(index)
plt.scatter(index, list_gray_median, s=5)
elim_low_freq.check_mkdir(gray_median_dist_path + '_size_%d'%sub_size)
plt.savefig(gray_median_dist_path + '_size_%d'%sub_size + '\\' + list_img_name[img_num])#保存每个子图的灰度距离分布
plt.figure()
#plt.show()
"""
"""
evaluation_all_matrix = [[] for i in range(img_num_total)]
for img_num in tqdm(range(0, img_num_total-1), desc='Processing image ', ncols=200, leave=True):
    img = cv2.imread(image_path + '\\' + list_img_name[img_num])
    xml_path = label_path + '\\' + list_xml_name[img_num]#找到标注文件
    outer_list = get_bndbox_From_Xml_Multi_Obj(xml_path)  # GroundTruth外点
    # 把每个图像在不同尺寸子图下的groundtruth外点保存下来

    #save_csv_single(save_path=groundtruth_list_path, save_file_name=file_save_name, save_data=outer_list)

    fac_val_list = [[] for i in range(subimg_num)]  # 储存不同值的list
    iterations = 0
    max_val = 0
    for i in range(0, height, sub_size):
        for j in range(0, width, sub_size):
            img_cropped = img[i:i + sub_size, j:j + sub_size]  # 将原图按照尺寸大小分为若干个子图
            # 该ij下的子图数据值
            #val = round(saturation_mean(img_cropped, pixel_size))
            val = gray_mean(img_cropped, pixel_size)
            if val > max_val:
                max_val = val
            fac_val_list[iterations].extend([iterations, val])
            iterations += 1

    threshold_list, std_var_list = ransac_threshold_slope0(list=fac_val_list, point_total=subimg_num, max_val=max_val,
                                                           iters=1000, epsilon=3, threshold=3)
    score_mat = score_matrix(std_list=std_var_list)
    scorelist, predict_point_list = borda_count(rate_of_extpnt=0.2, std_list=std_var_list)  # calcu_diff方法返回值为多个阈值下的外点矩阵
    # 评估矩阵，包含置信度、混淆矩阵参数
    # name1 = "score_mat.xlsx"
    # path1 = "E:\\LIC\\CV_detection\\experiment_records\\graph\\" + name1
    name2 = "cope_score.xlsx"
    path2 = "E:\\LIC\\CV_detection\\experiment_records\\graph\\" + name2
    # sheet1 = "score_mat"
    sheet2 = "cope_score"
    # frame1 = pd.DataFrame(score_mat)
    frame2 = pd.DataFrame(scorelist)
    # frame1.to_excel(path1, sheet_name=sheet1)
    frame2.to_excel(path2, sheet_name=sheet2)
    evaluation_matrix = evaluation_judge(groundtruth_point_list=outer_list, predict_point_list=predict_point_list)
    evaluation_all_matrix[img_num].extend(evaluation_matrix)
    #print("----------------------" + str(list_img_name[img_num]) + " processes over ---------------------------\n")
"""

