import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict, Set


def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i
    return elem

class XmlFormat(object):
    '''xml annotations analysis with specified xml directory
    '''

    @staticmethod
    def get_xml_list(xml_dir: str, sort=False) -> List[str]:
        ''' get a list of xml absolute path in specified xml directory
        '''
        xml_list =list(Path(xml_dir).glob('*.xml'))
        xml_list = [str(p.absolute()) for p in xml_list]
        return sorted(xml_list) if sort else xml_list

    @staticmethod
    def get_obj_num(xml_path: str) -> int:
        '''get object number from xml annotation
        '''
        assert Path(xml_path).exists(), f"{xml_path} not exist!"
        return len(ET.parse(xml_path).getroot().findall('object'))

    @staticmethod
    def get_obj_names(xml_path: str) -> Set:
        '''get a set of object names from xml annotation
        '''
        assert Path(xml_path).exists(), f"{xml_path} not exist!"
        root=ET.parse(xml_path).getroot()
        obj_names = set([obj.find('name').text for obj in root.findall('object')])
        return obj_names
    
    @staticmethod
    def get_main_obj(xml_path: str) -> List:
        '''get main object from xml annotation

        Return:
            [obj_name, obj_num] or [] if no object found
        '''
        assert Path(xml_path).exists(), f"{xml_path} not exist!"
        root=ET.parse(xml_path).getroot()
        obj_names = [obj.find('name').text for obj in root.findall('object')]
        if len(obj_names) > 0:
            return Counter(obj_names).most_common(1)[0]
        else:
            return []

    @staticmethod
    def get_img_info(xml_path: str) -> List:
        '''get image info from xml annotation
    
        Return:
            [img_name, img_width, img_height, img_depth]
        '''

        assert Path(xml_path).exists(), f"{xml_path} not exist!"

        tree=ET.parse(xml_path)
        root=tree.getroot()
        img_name = root.find('filename').text
        img_width = int(root.find('size/width').text)
        img_height = int(root.find('size/height').text)
        img_depth = int(root.find('size/depth').text)
        img_info = [img_name, img_width, img_height, img_depth]
        return img_info     

    @staticmethod
    def parse_xml_info(xml_path: str) -> Tuple[List, Dict]:
        ''' parse xml annotation info

        Return
            img_info: [list], [img_name, W, H, C]
            obj_info: [dict], {obj_name1: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                               obj_name2: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                               ...
                               }
        '''
        assert Path(xml_path).exists(), f"{xml_path} not exist!"

        tree=ET.parse(xml_path)
        root=tree.getroot()
        img_name = root.find('filename').text
        img_width = int(root.find('size/width').text)
        img_height = int(root.find('size/height').text)
        img_depth = int(root.find('size/depth').text)
        img_info = [img_name, img_width, img_height, img_depth]

        obj_info = {}
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            if obj_name not in obj_info.keys():
                obj_info[obj_name] = []
            obj_info[obj_name].append((xmin, ymin, xmax, ymax))
        
        return img_info, obj_info
    
    @staticmethod
    def dump_xml(img_info: List, obj_info: Dict, out_path: str):
        '''dump xml annotation with image info and object info

        Args:
            img_info: [list], [img_name, W, H, C]
            obj_info: [dict], {obj_name1: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                               obj_name2: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
                               ...
                               }

        Note: truncation and difficult info are set to 0.    
        '''
        p = Path(out_path)
        assert p.suffix == '.xml', "invalid output xml path!"
        out_dir, xml_name = p.parent.resolve(), p.name

        root = ET.Element('annotation')
        folder = ET.SubElement(root, 'folder')
        folder.text = str(out_dir)
        filename = ET.SubElement(root, 'filename')
        img_ext = img_info[0].split('.')[-1]
        img_name = xml_name.replace('.xml', '.'+img_ext)
        filename.text = img_name
        path = ET.SubElement(root, 'path')
        path.text = str(out_dir.joinpath(img_name))
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
        width.text = str(img_info[1])
        height.text = str(img_info[2])
        depth.text = str(img_info[3])

        for obj_name, bbox in obj_info.items():
            for box in bbox:
                object_root = ET.SubElement(root, 'object')
                name = ET.SubElement(object_root, 'name')
                name.text = obj_name
                pose = ET.SubElement(object_root, 'pose')
                pose.text = "Unspecified"
                trunc = ET.SubElement(object_root, 'truncated')
                trunc.text = "0"
                diff = ET.SubElement(object_root, 'difficult')
                diff.text = "0"
                bndbox = ET.SubElement(object_root, 'bndbox')
                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(int(box[0]))
                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(int(box[1]))
                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(int(box[2]))
                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(int(box[3]))
        indent(root)
        tree = ET.ElementTree(root)
        tree.write(out_path)

    def __init__(self, xml_dir):
        self.xml_dir = xml_dir
        self.xml_list = self.get_xml_list(xml_dir)
        self.xml_num = len(self.xml_list)
