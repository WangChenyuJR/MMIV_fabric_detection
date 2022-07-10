from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
from tqdm import tqdm


with PyCallGraph(output=GraphvizOutput()):
    rate_of_extpnts = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    img_num_total =
    for rate in range(len(rate_of_extpnts)):
        evaluation_all_matrix = [[] for i in range(img_num_total)]
        for img_num in tqdm(range(0, img_num_total - 1), desc='Processing image ', ncols=200, leave=True):
            img = cv2.imread(image_path + '\\' + list_img_name[img_num])
            xml_path = label_path + '\\' + list_xml_name[img_num]  # 找到标注文件
            outer_list = get_bndbox_From_Xml_Multi_Obj(xml_path)  # GroundTruth外点
            # 把每个图像在不同尺寸子图下的groundtruth外点保存下来

            # save_csv_single(save_path=groundtruth_list_path, save_file_name=file_save_name, save_data=outer_list)

            fac_val_list = [[] for i in range(subimg_num)]  # 储存不同值的list
            iterations = 0
            max_val = 0
            for i in range(0, height, sub_size):
                for j in range(0, width, sub_size):
                    img_cropped = img[i:i + sub_size, j:j + sub_size]  # 将原图按照尺寸大小分为若干个子图
                    # 该ij下的子图数据值
                    # val = round(saturation_mean(img_cropped, pixel_size))
                    val = gray_mean(img_cropped, pixel_size)
                    if val > max_val:
                        max_val = val
                    fac_val_list[iterations].extend([iterations, val])
                    iterations += 1

            threshold_list, std_var_list = ransac_threshold_slope0(list=fac_val_list, point_total=subimg_num,
                                                                   max_val=max_val,
                                                                   iters=1000, epsilon=2,
                                                                   threshold=2)
            predict_point_list = borda_count(rate_of_extpnt=0.2,
                                             std_list=std_var_list)  # calcu_diff方法返回值为多个阈值下的外点矩阵
            # 评估矩阵，包含置信度、混淆矩阵参数
            evaluation_matrix = evaluation_judge(groundtruth_point_list=outer_list,
                                                 predict_point_list=predict_point_list)
            evaluation_all_matrix[img_num].extend(evaluation_matrix)
            # print("----------------------" + str(list_img_name[img_num]) + " processes over ---------------------------\n")
