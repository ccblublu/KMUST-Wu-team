import cv2
import numpy as np
import copy

# def find_range(line_1, line_2):
#     point_min = np.concatenate(([line_1[0]], [line_2[0]]))
#     point_max = np.concatenate(([line_1[1]], [line_2[1]]))
#     x_min = max(point_min[0, 0], point_min[1, 0])
#     x_max = min(point_max[0, 0], point_max[1, 0])
#     y_min = max(point_min[0, 1], point_min[1, 1])
#     y_max = min(point_max[0, 1], point_max[1, 1])
#     if x_min > x_max:
#         select_x = 0
#     else:
#         select_x = 1
#     if y_min > y_max:
#         select_y = 0
#     else:
#         select_y = 1

#     return (x_min, x_max, select_x), (y_min, y_max, select_y)

# def cal_distance(l):
#     theta, datle_x, datle_y, select_x, select_y = l
#     x = np.abs(datle_x[0] - datle_x[1])
#     y = np.abs(datle_y[0] - datle_y[1])
#     distance_1 = y * np.cos(theta)
#     distance_2 = x * np.sin(theta)
#     if select_x and select_y:
#         return (distance_1 + distance_2) / 2
#     elif select_x:
#         return distance_2
#     elif select_y:
#         return distance_1
#     else:
#         raise TypeError("NO such selection")


def point_distance_line(point, line_points):
    #计算向量
    line_point1 = line_points[0]
    line_point2 = line_points[1]
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(
        vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


def cal_all_lines_diatance(base_line, all_lines_points, real_range):
    point_3 = (base_line[0] + base_line[1]) / 2
    points = [base_line[0], point_3, base_line[1]]
    all_lines_with_ori = []
    for i in range(len(all_lines_points)):
        # i += 1
        single_line_distance = 0
        sub_line_point = all_lines_points[i][:2]
        for j in range(3):
            single_line_distance += point_distance_line(
                points[j], sub_line_point)
        single_line_distance /= 3
        all_lines_with_ori.append(single_line_distance)
    image_distence = np.min(all_lines_with_ori) * real_range / 5.0
    return image_distence, all_lines_with_ori


def calculate_real_meters(distance, x_coors, base_p, h):    
    b_0 = np.linalg.norm([base_p[0], 1]) * distance
    new_parameter_1 = base_p + [0, b_0] # to
    new_parameter_2 = base_p - [0, b_0] # from
    y_coor_1 = np.polyval(new_parameter_1, x_coors).astype(np.int16)
    y_coor_2 = np.polyval(new_parameter_2, x_coors).astype(np.int16)
    y_coor_1[y_coor_1 < 0] = 0
    y_coor_2[y_coor_2 < 0] = 0
    y_coor_1[y_coor_1 > h] = h
    y_coor_2[y_coor_2 > h] = h
    return (y_coor_1, y_coor_2)


def full_mask(range_img, coors_depent, mask, coors='x'):
    if coors =='x':
        for x in range(range_img):
            to_ = coors_depent[0][x]
            from_ = coors_depent[1][x]
            mask[from_:to_, x] = 1
    elif coors == 'y':
        for y in range(range_img):
            to_ = coors_depent[0][y]
            from_ = coors_depent[1][y]
            mask[y, from_:to_] = 1
    else:
        raise TypeError("NO such selection {}".format(coors))
    return mask


def calculate_railway_range(mask_input, input_img, rito=0.5, if_plot=False):
    # mask_input1 = copy.deepcopy(mask_input)
    mask_input[mask_input > 1] = 0 # make sure only reserve railway 
    h, w = mask_input.shape
    '''1.提取边缘轮廓信息'''
    mask_img, railway_lines = cv2.findContours(mask_input, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
    lines_list = [
        sub_binary for sub_binary in mask_img if len(sub_binary) > 15
    ]  #剔除无效数据
    '''2.获取最远顶点，直线拟合'''
    all_lines_points = []
    for single_line in lines_list:
        single_line = single_line.squeeze()
        point_distance = np.linalg.norm(single_line, axis=1, keepdims=False)
        point_max = single_line[point_distance.argmax()]
        point_min = single_line[point_distance.argmin()]
        k_xy = np.polyfit((point_min[0], point_max[0]),
                          (point_min[1], point_max[1]), 1)
        k_yx = np.polyfit((point_min[1], point_max[1]),
                          (point_min[0], point_max[0]), 1)
        line_points = [point_max, point_min, k_xy, k_yx]
        all_lines_points.append(line_points)
    '''3.计算直线间距离'''
    image_distence, all_distance = cal_all_lines_diatance(
        all_lines_points[0], all_lines_points[1:], 50)
    '''3.5. 确定基准为x or y'''
    if np.abs(np.arctan(all_lines_points[0][2][0])) < 1.30:
        in_coor = 'x'
        range_img = w
        max_img = h
        p_index = 2
    else:
        in_coor = 'y'
        range_img = h
        max_img = w
        p_index = 3
    '''4.根据距离，y坐标确定x坐标'''
    coors = np.arange(0, range_img)
    coors_depent = calculate_real_meters(image_distence, coors, all_lines_points[0][p_index], max_img)
    '''5.设置掩码'''
    mask_1 = np.zeros_like(mask_input) + rito
    mask = full_mask(range_img, coors_depent, mask_1, in_coor)
    return mask
    # if if_plot:
    #     input_img  = input_img[...,:] * np.expand_dims(mask, axis=-1).repeat(3,axis=-1)
    #     cv2.imwrite('./test.png', input_img)

        
 

if __name__ == "__main__":
    input_mask_file = './datasets/VOC2007/masks/DJI_20210528104530_0068_W.png'
    img_file = './datasets/VOC2007/images/DJI_20210528104530_0068_W.jpg'
    input_mask = cv2.imread(input_mask_file, 0)
    input_img = cv2.imread(img_file)
    mask = calculate_railway_range(input_mask, input_img, if_plot=True)
    print(mask.max())
    print(mask.shape)

    # input_mask =  np.array([[0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         [0,0,0,0,0,0,0,1,1,0,0,0,1,1,1,1],
    #                         ])
    # input_mask = np.expand_dims(input_mask, 0)
    # input_mask = np.load('/media/chen/299D817A2D97AD94/must_done/yoloR/datasets/VOC2007/masks_npy/18123.npy')
    # railway_cluster(input_mask)