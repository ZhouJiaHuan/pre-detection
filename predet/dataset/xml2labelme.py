import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Union
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
    ''' xml format to labelme json format
    '''
    def __init__(self,
                 xml_dir: str,
                 out_dir: str,
                 cls_txt: Union[str, List],
                 with_group=False):
        super(Xml2labelme, self).__init__()
        self.xml = Xml(xml_dir)
        self.xml_dir = Path(xml_dir)
        assert self.xml_dir.exists(), "xml directory not found!"
        self.out_dir = Path(out_dir)

        if isinstance(cls_txt, str):
            assert Path(cls_txt).exists(), "class file not found!"
            self.classes = self._load_classes(cls_txt)
        else:
            assert isinstance(cls_txt, (list, tuple))
            self.classes = cls_txt
        self.with_group = with_group

    def _load_classes(self, cls_txt: str, ignore='#') -> List[str]:
        ''' load classes info from file
        '''
        with open(cls_txt, 'r') as f:
            return [line.strip() for line in f.readlines()
                if not line.startswith(ignore)]

    def _init_json(self) -> dict:
        data = {}
        data['version'] = '5.1.1'
        data['flags'] = {}
        data['shapes'] = []
        data['imageData'] = None
        return data
    
    def _convert_single(self, xml_path: str):
        data = self._init_json()
        img_info, obj_info = Xml.parse_xml_info(xml_path)
        data['imagePath'] = img_info[0]
        data['imageHeight'] = img_info[2]
        data['imageWidth'] = img_info[1]
        group_id = 0
        for obj_name, bboxes in obj_info.items():
            for bbox in bboxes:
                shape = dict(shape_type='rectangle', flags={})
                shape['label'] = obj_name
                shape['points'] = [bbox[:2], bbox[2:]]
                if self.with_group:
                    shape['group_id'] = group_id
                    group_id += 1
                data['shapes'].append(shape)

        json_name = Path(xml_path).stem + '.json'
        out_json = self.out_dir.joinpath(json_name)
        with open(out_json, 'w') as f:
            json.dump(data, f, indent=2, cls=MyEncoder)     

    def convert(self):
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)

        xml_list = Xml.get_xml_list(str(self.xml_dir))
        for xml_path in tqdm(xml_list):
            self._convert_single(xml_path)

    def convert_thread(self, threads=4):
        from multiprocessing.pool import ThreadPool
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)

        xml_list = Xml.get_xml_list(str(self.xml_dir))
        with ThreadPool(threads) as p:
            return list(tqdm(p.imap(self._convert_single, xml_list), total=len(xml_list)))