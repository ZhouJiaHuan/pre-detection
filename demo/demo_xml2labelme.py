import sys
sys.path.append('.')
import time
import argparse
import logging
from predet.dataset.xml2labelme import Xml2labelme

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="xml to coco anotations")
    parser.add_argument('xml_dir', type=str,
                        help='xml directory')
    parser.add_argument('out_dir', type=str,
                        help='output labelme json directory')
    parser.add_argument('cls_txt', type=str,
                        help='class txt file')
    parser.add_argument('--threads', type=int, default=1,
                        help='threads num for multi-threads')
    parser.add_argument('--with_group', action='store_true',
                        help='add group_id info')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    xml_dir = args.xml_dir
    out_dir = args.out_dir
    cls_txt = args.cls_txt
    with_group = args.with_group

    xml2labelme = Xml2labelme(xml_dir, out_dir, cls_txt, with_group)
    t1 = time.time()
    threads = max(1, args.threads)
    if threads == 1:
        xml2labelme.convert()
    else:
        xml2labelme.convert_thread(threads)
    t2 = time.time()
    logger.info(f"converted finished in {t2-t1} seconds")