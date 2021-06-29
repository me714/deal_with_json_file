import os
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split
from labelme import utils

np.random.seed(41)


class Lableme2CoCo:

    def to_coco(self, json_root_path, jpg_root):
        dd = 0
        # 将所有json文件地址读取成列表
        json_path_list = glob.glob(json_root_path + "/*/*.json")
        new_via_image_id_list =[]
        new_via_image_metadata = {}
        # 循环读取json文件
        for json_path in json_path_list:
            root_path = os.path.dirname(json_path)
            obj = self.read_jsonfile(json_path)

            _via_img_metadata = obj["_via_img_metadata"]
            _via_image_id_list = obj["_via_image_id_list"]
            # 从json文件中images项读取图片
            for images in _via_image_id_list:
                dd += 1
                old_image_name_path = root_path + '/' + images
                new_name = 'km' + f"{dd}" + '.jpg'
                # 复制改名
                shutil.copyfile(old_image_name_path, jpg_root +'/' + new_name)
                # 生成添加到新的images项
                new_via_image_id_list.append(new_name)

                new_via_image_metadata[new_name] = _via_img_metadata[images]
                new_via_image_metadata[new_name]['filename'] = new_name

                obj_new = self.read_jsonfile(json_path)
                obj_new["_via_image_id_list"] = new_via_image_id_list
                obj_new["_via_img_metadata"] = new_via_image_metadata
                # 保存新json文件
                new_json = jpg_root + '/new.json'
                new_json = open(new_json, "w", encoding='utf-8')
                new_json.write(json.dumps(obj_new))
                new_json.close()

    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            f = json.load(f)
            return f

    def write_jsonfile(self, path):
        with open(path, "w", encoding='utf-8') as f:
            bb = f
            return bb


if __name__ == '__main__':
    # 合并成一个文件保存的位置
    new_root = 'D:/ylqx/final_image_test'
    # 多个json文件所在位置
    json_list_path = 'D:\ylqx\外轮廓标记 - 副本'
    l2c_train = Lableme2CoCo()
    l2c_train.to_coco(json_list_path, new_root)

