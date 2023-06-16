from pathlib import Path
from tqdm import tqdm
from typing import List, Union
from .xml_format import XmlFormat as Xml


class Xml2Yolo(object):
    ''' transer xml annotations to yolo format
    '''

    def __init__(self,
                 xml_dir: str,
                 out_dir: str,
                 cls_txt: Union[str, List]):
        super(Xml2Yolo, self).__init__()
        # self.xml = Xml(xml_dir)
        self.xml_dir = Path(xml_dir)
        assert self.xml_dir.exists(), "xml or image directory not found!"
        self.out_dir = Path(out_dir)
        if isinstance(cls_txt, str):
            assert Path(cls_txt).exists(), f"{cls_txt} not found!"
            self.classes = self._load_classes(cls_txt)
        else:
            assert isinstance(cls_txt, (list, tuple))
            self.classes = cls_txt

    def _load_classes(self, cls_txt: str, ignore='#') -> List[str]:
        ''' load classes info from file
        '''
        with open(cls_txt, 'r') as f:
            return [line.strip() for line in f.readlines()
                if not line.startswith(ignore)]
        
    def _convert_single(self, xml_path: str):
        img_info, obj_info = Xml.parse_xml_info(xml_path)
        txt_path = self.out_dir.joinpath(Path(xml_path).stem + '.txt')
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
                row_str = ' '.join(list(map(str, row_list))) + '\n'
                data.append(row_str)
        with txt_path.open('w') as f:
            f.writelines(data)

    def convert(self):
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)

        for xml_path in tqdm(Xml.get_xml_list(str(self.xml_dir))):
            self._convert_single(xml_path)

    def convert_thread(self, threads=4):
        from multiprocessing.pool import ThreadPool
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)

        xml_list = Xml.get_xml_list(str(self.xml_dir))
        with ThreadPool(threads) as p:
            return list(tqdm(p.imap(self._convert_single, xml_list), total=len(xml_list)))

     