import sys
sys.path.append('.')
import logging
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from .xml_format import XmlFormat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Dota2Xml(object):
    ''' convert dota txt rect labels to voc xml format
    rect label format:
    x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult

    example (part of P0003.txt in val dataset)
    ...
    ...
    1077.0 949.0 1107.0 949.0 1107.0 987.0 1077.0 987.0 small-vehicle 1
    1085.0 969.0 1119.0 969.0 1119.0 1012.0 1085.0 1012.0 small-vehicle 0
    1088.0 706.0 1126.0 706.0 1126.0 757.0 1088.0 757.0 large-vehicle 0
    1089.0 621.0 1126.0 621.0 1126.0 667.0 1089.0 667.0 small-vehicle 1
    1038.0 620.0 1071.0 620.0 1071.0 664.0 1038.0 664.0 small-vehicle 0
    990.0 504.0 1022.0 504.0 1022.0 548.0 990.0 548.0 small-vehicle 1
    953.0 421.0 978.0 421.0 978.0 463.0 953.0 463.0 small-vehicle 1
    ...
    ...
    '''

    def __init__(self, img_dir: str, txt_dir: str, out_dir: str, with_difficult=True):
        self.img_dir = Path(img_dir)
        self.txt_dir = Path(txt_dir)
        assert self.img_dir.exists() and self.txt_dir.exists(), \
            "image dir or txt dir not found!"

        self.txt_list = sorted(list(self.txt_dir.glob('*.txt')))
        self.out_dir = Path(out_dir)
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
        self.with_difficult = with_difficult
        
    def _convert_single(self, txt_path: Path):
        img_path = self.img_dir.joinpath(txt_path.stem + '.png')
        if not img_path.exists():
            logging.warning(f"{img_path} not found, ignore ...")
            return False
        
        out_path = self.out_dir.joinpath(txt_path.stem + '.xml')
        img_w, img_h = Image.open(img_path).size
        img_info = [img_path.name, img_w, img_h, 3]
        obj_info = {}
        with txt_path.open('r') as f:
            row_list = f.readlines()
        for row in row_list:
            row = row.strip().split(' ')
            if (not self.with_difficult) and (int(row[-1]) == 1):
                continue
            obj_name = row[8]
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[4]), float(row[5])
            if obj_name not in obj_info.keys():
                obj_info[obj_name] = []
            obj_info[obj_name].append([x1, y1, x2, y2])
        XmlFormat.dump_xml(img_info, obj_info, str(out_path))
        return True

    def convert(self):
        for txt_path in tqdm(self.txt_list):
            try:
                self._convert_single(txt_path)
            except Exception as e:
                logger.error(e)

    def convert_threads(self, threads=4):
        from multiprocessing.pool import ThreadPool
        with ThreadPool(threads) as p:
           return list(tqdm(p.imap(self._convert_single, self.txt_list), total=len(self.txt_list))) 

