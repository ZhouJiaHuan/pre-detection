# coding: utf-8
# Author: ZhouJH
# Date: 2023/03/02

import os
import json
from ..utils.file_io import get_file_path

class LabelmeFormat(object):
    ''' Labelme json format dataset analysis
    '''
    @classmethod
    def parse_ann_info(self, json_path: str, with_group=False):
        ann_dict = {}
        try:
            with open(json_path, 'r') as f:
                ann_dict = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f'parse json file failed: {json_path}')
            return ann_dict
        img_info = [ann_dict['imagePath'], ann_dict['imageWidth'], ann_dict['imageHeight']]
        obj_info = {}
        if with_group:
            for shape in ann_dict['shapes']:
                if 'group_id' not in shape.keys():
                    print(f'ignore shape with no group id: {shape}')
                    continue
                group_id = str(shape['group_id'])
                if group_id not in obj_info.keys():
                    obj_info[group_id] = []
                obj_info[group_id].append(shape)
        else:
            for shape in ann_dict['shapes']:
                label = shape['label']
                if label not in obj_info.keys():
                    obj_info[label] = []
                obj_info[label].append(shape)
        return img_info, obj_info

    def __init__(self, json_dir: str) -> None:
        self.json_dir = json_dir
        self.json_list = get_file_path(json_dir, ['.json'])
        self.json_num = len(self.json_list)

    