import os
import csv
import pandas as pd
"""
每张图都输入一个真实外点列表和一个预测外点列表，得到每张图的混淆矩阵四个指标值
 True Positive（TP）：真正类。样本的真实类别是正类，并且模型识别的结果也是正类。

 False Negative（FN）：假负类。样本的真实类别是正类，但是模型将其识别为负类。

 False Positive（FP）：假正类。样本的真实类别是负类，但是模型将其识别为正类。

 True Negative（TN）：真负类。样本的真实类别是负类，并且模型将其识别为负类。
"""
def evaluation_judge(groundtruth_point_list, predict_point_list):
    #GT含有且预测也含有为TP,GT含有且预测不包含为FN，GT不包含而预测包含为FP，GT不包含而预测也不包含为TN
    groundtruth_point_list = set(groundtruth_point_list)
    predict_point_list = set(predict_point_list)
    true_num = len(groundtruth_point_list)
    pre_num = len(predict_point_list)

    same_elem_num = len(groundtruth_point_list & predict_point_list)#真实值和预测值重复的值，即预测正确的值
    TP = same_elem_num
    FN = true_num - same_elem_num
    FP = pre_num - same_elem_num
    TN = 0

    confidence = same_elem_num / (true_num + same_elem_num)#类似交并比IoU

    return true_num, pre_num, TP, FN, FP


def check_mkdir(path):
    mask_folder = os.path.exists(path)
    if not mask_folder:
        os.makedirs(path)

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
        writer.writerow(save_data[i])
        file.close()

b = 0.7
a = b < 0.5
c = 2*a + 7*(1-a)
print(a,b,c)




