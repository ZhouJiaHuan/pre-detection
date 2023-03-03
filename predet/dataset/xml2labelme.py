# coding: utf-8
# Author: ZhouJH
# Date: 2023/02/28

import os
import json
import numpy as np
from tqdm import tqdm
from ..utils.file_io import make_dirs
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
        

class Xml2labelme(object):
    ''' xml format -> labelme json format
    '''
    def __init__(self, xml_dir, out_dir, cls_txt):
        super(Xml2labelme, self).__init__()
        self.xml = Xml(xml_dir)
        self.out_dir = out_dir
        if isinstance(cls_txt, str):
            assert os.path.exists(cls_txt)
            self.classes = self._load_classes(cls_txt)
        else:
            assert isinstance(cls_txt, (list, tuple))
            self.classes = cls_txt

    def _init_json(self) -> dict:
        data = {}
        data['version'] = '5.1.1'
        data['flags'] = {}
        data['shapes'] = []
        data['imageData'] = None
        return data

    def convert(self, with_group=False):
        make_dirs(self.out_dir)
        for xml_path in tqdm(self.xml.xml_list):
            data = self._init_json()
            img_info, obj_info = self.xml.parse_xml_info(xml_path)
            data['imagePath'] = img_info[0]
            data['imageHeight'] = img_info[2]
            data['imageWidth'] = img_info[1]
            group_id = 0
            for obj_name, bboxes in obj_info.items():
                for bbox in bboxes:
                    shape = dict(shape_type='rectangle', flags={})
                    shape['label'] = obj_name
                    shape['points'] = [bbox[:2], bbox[2:]]
                    if with_group:
                        shape['group_id'] = group_id
                        group_id += 1
                    data['shapes'].append(shape)
            json_name = os.path.basename(xml_path).replace('.xml', '.json')
            out_json = os.path.join(self.out_dir, json_name) 
            with open(out_json, 'w') as f:
                json.dump(data, f, indent=2, cls=MyEncoder)
        print("convert finished.")