import sys
sys.path.append('.')
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from .xml_format import XmlFormat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Dota2Xml(object):
    ''' convert dota labels to voc xml format
    rect label format:
    x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult

    example (part of P0000.txt in train dataset):
    imagesource:GoogleEarth
    gsd:0.146343590398
    2753 2408 2861 2385 2888 2468 2805 2502 plane 0
    3445 3391 3484 3409 3478 3422 3437 3402 large-vehicle 0
    3185 4158 3195 4161 3175 4204 3164 4199 large-vehicle 0
    2870 4250 2916 4268 2912 4283 2866 4263 large-vehicle 0
    630 1674 628 1666 640 1654 644 1666 small-vehicle 0
    636 1713 633 1706 646 1698 650 1706 small-vehicle 0
    717 76 726 78 722 95 714 90 small-vehicle 0
    737 82 744 84 739 101 731 98 small-vehicle 0
    658 242 648 237 657 222 667 225 small-vehicle 1
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
            if len(row) != 10:
                continue
            if (not self.with_difficult) and (int(row[-1]) == 1):
                continue
            obj_name = row[8]
            polys = np.array(list(map(float, row[:8]))).reshape(4, 2)
            x1, y1 = polys.min(axis=0)
            x2, y2 = polys.max(axis=0) 
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

