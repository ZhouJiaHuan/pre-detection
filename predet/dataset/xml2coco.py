# coding: utf-8
# Description: xml标注及图片转coco
# Author: ZhouJH
# Date: 2019/11/08

import os
import json
import numpy as np
from tqdm import tqdm
from .xml_format import XmlFormat as Xml


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class Xml2Coco(object):
    '''xml换为COCO格式

    xml标注和对应的图片需要放在相同的路径下，图片为jpg格式

    Args:
        xml_dir: [str], xml文件所在路径(支持递归搜索)
        out_path: [str], lmdb格式数据保存路径
        cls_txt: [str or list], 类别名列表或记录类别名的文本，不在cls_txt中的类别会被忽略
    '''

    def __init__(self, xml_dir, out_path, cls_txt):
        super(Xml2Coco, self).__init__()
        self.xml = Xml(xml_dir)
        self.out_path = out_path
        assert out_path.split('.')[-1] == 'json', \
            "output coco annotations must be in json format!"
        if isinstance(cls_txt, str):
            assert os.path.exists(cls_txt)
            self.classes = self._load_classes(cls_txt)
        else:
            assert isinstance(cls_txt, (list, tuple))
            self.classes = cls_txt

        self.img_list = []
        self.images = []
        self.categories = self._categorie()
        self.annotations = []
        
        self.labels = []
        self.obj_num = 0
        self.height = 0
        self.width = 0

    def _load_classes(self, cls_txt):
        '''读取数据的类别信息

        类别需要按行写入到一个txt文本中

        Args:
            cls_txt: 记录类别信息的txt路径
        Return:
            记录类别信息的字符串列表
        '''
        assert os.path.exists(cls_txt)
        with open(cls_txt, 'r') as f:
            return [line.strip() for line in f.readlines()
                if not line.startswith('#')]

    def _image(self, img_info, num):
        '''生成COCO标注的的"images"信息

        Args:
            img_info: [list], [img_name, w, h, c]
            num: 当前图片的标号（从1开始）
        Return:
            image: [dict], 包含height, width, id, file_name
        '''
        image = {}
        width = int(img_info[1])
        height = int(img_info[2])
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = os.path.basename(img_info[0])
        self.height = height
        self.width = width
        return image

    def _annotation(self, obj, img_name):
        '''生成COCO标注中的annotation

        Args:
            obj: [list], [label, bbx]
                 其中label为str, bbx为[xmin, ymin, w, h]
            img_name: [str], 当前obj所在的图片名
        '''
        annotation = {}
        annotation['segmentation'] = []
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.img_list.index(img_name) + 1
        annotation['bbox'] = obj[1:]
        annotation['area'] = obj[3] * obj[4]
        annotation['category_id'] = self.classes.index(obj[0])+1
        annotation['id'] = self.obj_num
        return annotation

    def _categorie(self):
        '''生成COCO标注中的categories
        '''
        categories = []
        for idx, cls_name in enumerate(self.classes):
            categorie = {}
            categorie['supercategory'] = 'Unspecified'
            categorie['id'] = idx + 1  # 0 默认为背景
            categorie['name'] = cls_name
            categories.append(categorie)
        return categories

    def _data2coco(self):
        '''将加载的标注按COCO格式存储
        '''
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco

    def _data_transfer(self):
        '''加载xml标注信息
        '''
        for num, xml_path in enumerate(tqdm(self.xml.xml_list)):
            img_name = os.path.basename(xml_path).replace('.xml', '.jpg')
            self.img_list.append(img_name)
            img_info, obj_info = self.xml.parse_xml_info(xml_path)
            img_info[0] = img_name # 使用xml对应的文件名
            self.images.append(self._image(img_info, num))
            for label, bbox in obj_info.items():
                if label not in self.classes:
                    continue
                for box in bbox:
                    self.obj_num += 1
                    obj = list(box[:2]) + [box[2]-box[0],box[3]-box[1]]
                    obj.insert(0, label)
                    self.annotations.append(self._annotation(obj, img_name))

    def convert(self):
        '''转化COCO格式标注
        '''
        print("loading xml annotations ...")
        self._data_transfer()
        self.coco = self._data2coco()
        print("saving coco annotations ...")
        with open(self.out_path, 'w') as f:
            json.dump(self.coco, f, indent=4, cls=MyEncoder)
        print("convert finished.")
