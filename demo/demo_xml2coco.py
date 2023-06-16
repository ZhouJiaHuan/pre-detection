import sys
sys.path.append('.')
import time
import argparse
import logging
from predet.dataset.xml2coco import Xml2Coco

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="xml to coco anotations")
    parser.add_argument('xml_dir', type=str, default=None,
                        help='xml directory')
    parser.add_argument('out_json', type=str,
                        help='output coco json path')
    parser.add_argument('cls_txt', type=str,
                        help='class txt file')
    parser.add_argument('--img-ext', type=str, default='jpg',
                        help='image format, default jpg')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    xml_dir = args.xml_dir
    out_json = args.out_json
    cls_txt = args.cls_txt
    img_ext = '.' + args.img_ext

    xml2coco = Xml2Coco(xml_dir, out_json, cls_txt, img_ext)
    t1 = time.time()
    xml2coco.convert()
    t2 = time.time()
    logger.info(f"converted finished in {t2-t1} seconds")