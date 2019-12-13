# coding=UTF-8

import os
import cv2

from PIL import Image
from logzero import logger

def get_arguments(height, weight, label, x_left, y_left, x_right, y_right):
    if label == '带电芯充电宝':
        olabel = 0
    else:
        olabel = 1

    x_mid = (x_left + x_right) / 2
    x_mid /= weight

    y_mid = (y_left + y_right) / 2
    y_mid /= height

    x_length = x_right - x_left
    x_length /= weight

    y_length = y_right - y_left
    y_length /= height

    return olabel, x_mid, y_mid, x_length, y_length

dataset_folder = './dataset'
core_folder = dataset_folder + '/coreless_3000'
core_anno = core_folder + '/Annotation/'
core_img = core_folder + '/Image/'

core_list = os.listdir(core_anno)
core_list = sorted(core_list)

core_img_list = os.listdir(core_img)
core_img_list = sorted(core_img_list)

core_cur_idx = 0
core_max_len = len(core_list)

label_list = ['带电芯充电宝','不带电芯充电宝']

for core_cur_idx in range(core_cur_idx, core_max_len):
    anno_file = core_anno
    anno_file += core_list[core_cur_idx]
    img_file = core_img
    img_file += core_img_list[core_cur_idx]

    image = cv2.imread(img_file)
    height = image.shape[0]
    weight = image.shape[1]

    with open(anno_file, 'r') as f:
        lines = f.readlines()
        f.close()

    enter = False
    with open(anno_file, 'w') as f_w:
        for i in range(len(lines)):
            line = lines[i]
            if i < len(lines):
                enter = True
            else:
                enter = False

            str1 = ''
            label = line.split(' ')[0]

            x_left = line.split(' ')[1]
            x_left = int(x_left)

            y_left = line.split(' ')[2]
            y_left = int(y_left)

            x_right = line.split(' ')[3]
            x_right = int(x_right)

            y_right = line.split(' ')[4]
            y_right = int(y_right)

            olabel, x_mid, y_mid, x_length, y_length = get_arguments(height, weight, label, x_left, y_left, x_right,
                                                                     y_right)
            x_mid = '%.5f' % x_mid
            y_mid = '%.5f' % y_mid
            x_length = '%.5f' % x_length
            y_length = '%.5f' % y_length

            str1 += (str(olabel) + ' ' + str(x_mid) + ' ' + str(y_mid) + ' ' + str(x_length) + ' ' + str(y_length))
            if enter:
                str1 += '\n'

            logger.info(core_cur_idx+1)
            logger.info(anno_file)
            logger.info(line)
            logger.info(str1)
            logger.info('')

            f_w.write(str1)
        f_w.close()
