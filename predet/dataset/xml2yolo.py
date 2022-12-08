# Author: ZhouJH
# Date: 2022/12/08


import os
import numpy as np
from tqdm import tqdm
from ..utils.file_io import make_dirs, write_list_to_txt
from .xml_format import XmlFormat as Xml


class Xml2Yolo(object):
    ''' transer xml annotations to yolo format
    '''

    def __init__(self, xml_dir, out_dir, cls_txt) -> None:
        super(Xml2Yolo, self).__init__()
        self.xml = Xml(xml_dir)
        self.out_dir = out_dir
        if isinstance(cls_txt, str):
            assert os.path.exists(cls_txt)
            self.classes = self._load_classes(cls_txt)
        else:
            assert isinstance(cls_txt, (list, tuple))
            self.classes = cls_txt
        make_dirs(out_dir)

    def convert(self):
        for xml_path in tqdm(self.xml.xml_list):
            img_info, obj_info = self.xml.parse_xml_info(xml_path)
            xml_name = os.path.basename(xml_path)
            txt_path = os.path.join(self.out_dir, xml_name.replace('.xml', '.txt'))
            _, img_w, img_h, _ = img_info

            data = []
            for label, bbox in obj_info.items():
                if label not in self.classes:
                    continue
                cls_id = self.classes.index(label)
                for box in bbox:
                    x1, y1, x2, y2 = box
                    cx, cy = (x1+x2) * 0.5, (y1+y2) * 0.5
                    bw, bh = x2-x1, y2-y1
                    norm_cx, norm_cy = cx / img_w, cy / img_h
                    norm_bw, norm_bh = bw / img_w, bh / img_h
                    row_list = [cls_id, norm_cx, norm_cy, norm_bw, norm_bh]
                    row_str = ' '.join(list(map(str, row_list)))
                    data.append(row_str)
            write_list_to_txt(data, txt_path)



        