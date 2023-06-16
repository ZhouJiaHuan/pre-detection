import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from ..dataset.xml_format import XmlFormat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ImageSlice(object):
    ''' slice the image and xml annotations (optional)
    '''
    def __init__(self,
                 img_dir: str,
                 out_dir: str,
                 xml_dir: str=None,
                 slice_size=(640, 640),
                 overlap_ratio=(0.5, 0.5),
                 min_area_ratio=0.2,
                 ext='jpg'
                 ):
        self.img_dir = Path(img_dir)
        self.out_dir = Path(out_dir)
        self.xml_dir = Path(xml_dir) if xml_dir is not None else None
        self.img_list = sorted(list(self.img_dir.glob('*.'+ext)))
        self.slice_w, self.slice_h = slice_size
        self.overlap_w = self.slice_w * overlap_ratio[0]
        self.overlap_h = self.slice_h * overlap_ratio[1]
        self.min_area_ratio = min_area_ratio

        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)
        
        assert self.img_dir.exists(), "image directory not found!"
        if self.xml_dir:
            assert self.xml_dir.exists(), "xml directory not found!"

    def _get_slice_bboxes(self, img_size):
        img_w, img_h = img_size
        slice_bboxes = []
        ymax = ymin = 0
        while ymax < img_h:
            xmin = xmax = 0
            ymax = ymin + self.slice_h
            while xmax < img_w:
                xmax = xmin + self.slice_w
                x2 = min(img_w, xmax)
                y2 = min(img_h, ymax)
                x1 = max(0, x2-self.slice_w)
                y1 = max(0, y2-self.slice_h)
                slice_bboxes.append([x1, y1, x2, y2])
                xmin = xmax - self.overlap_w
            ymin = ymax - self.overlap_h
        return slice_bboxes
    
    def _get_obj_with_bbox(self, obj_info: dict, slice_bbox: list):
        patch_obj = {}
        patch_x1, patch_y1, patch_x2, patch_y2 = slice_bbox
        for obj_name, bboxes in obj_info.items():
            for bbox in bboxes:
                bbox_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                x1 = min(patch_x2, max(patch_x1, bbox[0]))
                y1 = min(patch_y2, max(patch_y1, bbox[1]))
                x2 = min(patch_x2, max(patch_x1, bbox[2]))
                y2 = min(patch_y2, max(patch_y1, bbox[3]))
                area_ratio = (x2-x1)*(y2-y1) / bbox_area
                if area_ratio < self.min_area_ratio:
                    continue
                x1 -= patch_x1
                y1 -= patch_y1
                x2 -= patch_x1
                y2 -= patch_y1
                if obj_name not in patch_obj.keys():
                    patch_obj[obj_name] = []
                patch_obj[obj_name].append([x1, y1, x2, y2])
                # breakpoint()
        return patch_obj
    
    def _slice_single(self, img_path: Path):
        # 1. get patches
        img = Image.open(img_path)
        slice_bboxes = self._get_slice_bboxes(img.size)
        for i, slice_bbox in enumerate(slice_bboxes):
            patch_img = img.crop(slice_bbox)
            save_name = img_path.stem + '_' + str(i)
            save_path = self.out_dir.joinpath(save_name+img_path.suffix)
            patch_img.save(save_path)
            
            if self.xml_dir:
                xml_path = self.xml_dir.joinpath(img_path.stem+'.xml')
                _, obj_info = XmlFormat.parse_xml_info(xml_path)
                patch_img_info = [save_name+img_path.suffix, *patch_img.size, 3]
                patch_bboxes = self._get_obj_with_bbox(obj_info, slice_bbox)
                xml_save_path = self.out_dir.joinpath(save_name+'.xml')
                XmlFormat.dump_xml(patch_img_info, patch_bboxes, str(xml_save_path))

    def run(self):
        for img_path in tqdm(self.img_list):
            try:
                self._slice_single(img_path)
            except Exception as e:
                logger.error(e)

    def run_thread(self, threads=4):
        with ThreadPool(threads) as p:
            return list(tqdm(p.imap(self._slice_single, self.img_list), total=len(self.img_list)))


