import os
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split
from labelme import utils
import random

np.random.seed(41)

# 0为背景

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.class_name_to_id = {}
        self.dd = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path, tags):
        obj = self.read_jsonfile(json_path)
        _via_attributes = obj['_via_attributes']
        region = _via_attributes['region']
        takephoto = region['takephoto']
        options = takephoto['options']
        for option in options:
            if option == '东华':
                continue
            self.dd += 1
            self.class_name_to_id[option] = self.dd
        self._init_categories()
        _via_img_metadata = obj['_via_img_metadata']
        _via_image_id_list = obj['_via_image_id_list']
        random.shuffle(_via_image_id_list)
        ll = self._dataset_split(tags, _via_image_id_list, _via_img_metadata)
        for img_id, img in enumerate(ll):
            self.images.append(self._image(ll, img, img_id))
            imga = ll[img]
            regions = imga['regions']
            for r, region in enumerate(regions):
                annotation, point_x = self._annotation(region)
                if not point_x:
                    break
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance, ll

    # 构建类别
    def _init_categories(self):
        for k, v in self.class_name_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, _via_img_metadata, img, img_id):
        image = {}
        # img_x = utils.img_b64_to_arr(obj['imageData'])
        # h, w = img_x.shape[:-1]
        photo = _via_img_metadata[img]
        filename = photo['filename']
        hw = photo['file_attributes']
        image['height'] = hw['height']
        image['width'] = hw['width']
        image['id'] = self.img_id
        image['file_name'] = filename
        return image

    # 构建COCO的annotation字段
    def _annotation(self, region):
        region_attributes = region['region_attributes']
        shape_attributes = region['shape_attributes']
        all_points_x = shape_attributes['all_points_x']
        all_points_y = shape_attributes['all_points_y']
        a = np.array([all_points_x, all_points_y])
        b = np.transpose(a)
        label = region_attributes['takephoto']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(self.class_name_to_id[label])
        annotation['segmentation'] = [np.asarray(b).flatten().tolist()]
        annotation['bbox'] = self._get_box(b)
        annotation['iscrowd'] = 0
        annotation['area'] = annotation['bbox'][-1] * annotation['bbox'][-2]
        return annotation, all_points_x

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    # 将图片分为训练集，验证集和测试集
    def _dataset_split(self, tags, _via_image_id_list, _via_img_metadata):
        bb = {}
        if tags == 'train':
            _via_image_id_list = _via_image_id_list[:int(0.8*len(_via_image_id_list))]
        elif tags == 'valid':
            _via_image_id_list = _via_image_id_list[int(0.8*len(_via_image_id_list)):int(0.9*len(_via_image_id_list))]
        else:
            _via_image_id_list = _via_image_id_list[int(0.9*len(_via_image_id_list)):]
        for i in _via_image_id_list:
            bb[i] = _via_img_metadata[i]
        return bb


if __name__ == '__main__':
    # 将一个文件夹下的照片和labelme的标注文件，分成了train和val的coco json文件和照片

    train_img_out_path = './train_img'
    val_img_out_path = './val_img'
    test_img_out_path = './test_img'
    if not (os.path.exists(train_img_out_path) and os.path.exists(val_img_out_path)):
        os.mkdir(train_img_out_path)
        os.mkdir(val_img_out_path)
        os.mkdir(test_img_out_path)

    json_list_path = 'D:/ylqx/final_image_test/new.json'
    # print('json_list_path=', json_list_path)

    l2c_train = Lableme2CoCo()
    train_instance, train_img_list = l2c_train.to_coco(json_list_path, 'train')
    l2c_train.save_coco_json(train_instance, 'train_a.json')
    # print(train_img_list)
    l2c_valid = Lableme2CoCo()
    valid_instance, valid_img_list = l2c_valid.to_coco(json_list_path, 'valid')
    l2c_train.save_coco_json(valid_instance, 'valid_a.json')
    l2c_test = Lableme2CoCo()
    test_instance, test_img_list = l2c_test.to_coco(json_list_path, 'test')
    l2c_train.save_coco_json(test_instance, 'test_a.json')

    for file in train_img_list:
        shutil.copy(file.replace("json", "jpg"), train_img_out_path)
    for file in valid_img_list:
        shutil.copy(file.replace("json", "jpg"), val_img_out_path)
    for file in test_img_list:
        shutil.copy(file.replace("json", "jpg"), test_img_out_path)