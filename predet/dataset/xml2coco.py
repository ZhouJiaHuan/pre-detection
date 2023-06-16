import json
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Union
from .xml_format import XmlFormat as Xml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    '''xml annotations to coco json annotations
    '''

    def __init__(self,
                 xml_dir: str,
                 out_path: str,
                 cls_txt: Union[str, List],
                 img_ext='.jpg'):
        '''
        Args:
            xml_dir: [str], xml annotation directory
            out_path: [str], COCO json format save path
            cls_txt: [str | list], class list or class file
            img_ext: [str], img format, default '.jpg'
        '''
        super(Xml2Coco, self).__init__()
        self.xml_dir = xml_dir
        assert Path(self.xml_dir).exists(), "xml or image directory not found!"
        self.out_path = out_path
        assert out_path.split('.')[-1] == 'json', \
            "output coco annotations must be in json format!"
        if isinstance(cls_txt, str):
            assert Path(cls_txt).exists(), f"class file not found: {cls_txt}!"
            self.classes = self._load_classes(cls_txt)
        else:
            assert isinstance(cls_txt, (list, tuple))
            self.classes = cls_txt
        self.img_ext = img_ext

        self.img_list = []
        self.images = []
        self.categories = self._categorie()
        self.annotations = []
        
        self.labels = []
        self.obj_num = 0
        self.height = 0
        self.width = 0

    def _load_classes(self, cls_txt: str, ignore='#') -> List[str]:
        ''' load classes info from file
        '''
        with open(cls_txt, 'r') as f:
            return [line.strip() for line in f.readlines()
                if not line.startswith(ignore)]

    def _image(self, img_info: List, num: int) -> Dict:
        '''generate 'image' info in coco json format

        Args:
            img_info: [list], [img_name, w, h, c]
            num: current image idx (start from 1)

        Return:
            image: [dict], dict with height, width, id, file_name
        '''
        image = {}
        width = int(img_info[1])
        height = int(img_info[2])
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = Path(img_info[0]).name
        self.height = height
        self.width = width
        return image

    def _annotation(self, obj, img_name) -> Dict:
        '''generate 'annotation' info in coco json format

        Args:
            obj: [list], [cls_name, xmin, ymin, w, h]
            img_name: [str], image name corresponding to current obj
        '''
        annotation = {}
        annotation['segmentation'] = []
        annotation['iscrowd'] = 0
        annotation['image_id'] = self.img_list.index(img_name) + 1
        annotation['bbox'] = obj[1:]
        annotation['area'] = obj[3] * obj[4]
        annotation['category_id'] = self.classes.index(obj[0]) + 1
        annotation['id'] = self.obj_num
        return annotation

    def _categorie(self) -> List[Dict]:
        '''generate 'categories' info in coco json format
        '''
        categories = []
        for idx, cls_name in enumerate(self.classes):
            categorie = {}
            categorie['supercategory'] = 'Unspecified'
            categorie['id'] = idx + 1  # 0 = background
            categorie['name'] = cls_name
            categories.append(categorie)
        return categories

    def _data_transfer(self):
        '''load xml annotations info
        '''
        for num, xml_path in enumerate(tqdm(Xml.get_xml_list(self.xml_dir))):
            img_name = Path(xml_path).stem + self.img_ext
            self.img_list.append(img_name)
            img_info, obj_info = Xml.parse_xml_info(xml_path)
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
        ''' run convert process
        '''
        logger.info("loading xml annotations ...")
        self._data_transfer()

        logger.info("saving coco annotations ...")
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        with open(self.out_path, 'w') as f:
            json.dump(data_coco, f, indent=4, cls=MyEncoder)

        logger.info("convert finished.")
