# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :deal_with_json_file
# @File     :COCO2CUB
# @Date     :2022/3/15 13:36
# @Author   :Sun
# @Email    :szqqishi@163.com
# @Software :PyCharm
-------------------------------------------------
"""
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re


class COCO2CUB(object):
    def __init__(self, input_json, save_json):
        '''
        :param input_json: 输入COCO格式json地址
        :param save_json: 输出CUB格式json地址
        '''
        self.input_json = input_json
        self.label_names = 0
        self.img_to_id = {}
        self.id_to_parts = {}
        self.dataset_list = ['base', 'novel']

    def read_json(self):
        with open(self.input_json, "r", encoding='utf-8') as f:
            return json.load(f)

    def get_message(self):
        json_message = self.read_json()
        images = json_message['images']
        categories = json_message['categories']
        annotations = json_message['annotations']
        for i in images:
            file_name = i['file_name']
            file_id = i['id']
            image_class = i['image_class']
            self.img_to_id[str(image_class) + '/' + file_name] = file_id
        part_list = []
        for j in annotations:
            image_id = j['image_id']
            bbox = j['bbox']
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[0] + bbox[2]
            y2 = bbox[1] + bbox[3]
            part = [(x1 + x2) / 2, (y2 + y1) / 2, 1]
            if image_id not in self.id_to_parts.keys():
                del part_list[:]
                part_list.append(part)
                self.id_to_parts[image_id] = part_list
            else:
                part_list.append(part)

            self.id_to_parts[image_id] = part_list
        cwd = os.getcwd()
        data_path = join(cwd, 'FGVC_2\\fgvc_images')
        savedir = './'
        folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
        folder_list.sort()
        label_dict = dict(zip(folder_list, range(0, len(folder_list))))
        classfile_list_all = []
        for i, folder in enumerate(folder_list):
            folder_path = join(data_path, folder)
            classfile_list_all.append([join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path, cf)) and cf[0] != '.')])
            random.shuffle(classfile_list_all)
        for dataset in self.dataset_list:
            file_list = []
            label_list = []
            for i, classfile_list in enumerate(classfile_list_all):
                if 'base' in dataset:
                    if (i % 2 == 0):
                        file_list = file_list + classfile_list
                        label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
                if 'val' in dataset:
                    if (i % 4 == 1):
                        file_list = file_list + classfile_list
                        label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
                if 'novel' in dataset:
                    if (i % 2 != 0):
                        file_list = file_list + classfile_list
                        label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
            part_list = []
            for path in file_list:
                path = path.replace('\\', '/')
                filename = re.search('/fgvc_images/(.*)', path, flags=0).group(1)
                # print(self.img_to_id.get(filename))
                part_list.append(self.id_to_parts[self.img_to_id[filename]])
            with open(savedir + dataset + '.json', 'w') as outfile:
                json.dump({'label_names': folder_list, 'image_names': file_list, 'image_labels': label_list,
                           'part': part_list}, outfile)
            print("%s -OK" % dataset)

if __name__ == '__main__':
    train = COCO2CUB('total.json', '/dataset/')
    train.get_message()
