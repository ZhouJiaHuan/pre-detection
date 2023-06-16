import sys
sys.path.append('.')
import time
import argparse
import logging
from predet.dataset.xml2yolo import Xml2Yolo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="xml to coco anotations")
    parser.add_argument('xml_dir', type=str,
                        help='xml directory')
    parser.add_argument('out_dir', type=str,
                        help='output coco json path')
    parser.add_argument('cls_txt', type=str,
                        help='class txt file')
    parser.add_argument('--threads', type=int, default=1,
                        help='threads num for multi-threads')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    xml_dir = args.xml_dir
    out_dir = args.out_dir
    cls_txt = args.cls_txt

    xml2yolo = Xml2Yolo(xml_dir, out_dir, cls_txt)
    t1 = time.time()
    threads = max(1, args.threads)
    if threads == 1:
        xml2yolo.convert()
    else:
        xml2yolo.convert_thread(threads)
    t2 = time.time()
    logger.info(f"converted finished in {t2-t1} seconds")