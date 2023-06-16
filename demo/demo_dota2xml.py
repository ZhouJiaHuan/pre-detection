import sys
sys.path.append('.')
import time
import logging
import argparse
from predet.dataset.dota2xml import Dota2Xml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="convert dota to xml format")
    parser.add_argument('img_dir', type=str,
                        help='dota image directory')
    parser.add_argument('txt_dir', type=str,
                        help='dota rect label directory')
    parser.add_argument('out_dir', type=str,
                        help='output xml label directory')
    parser.add_argument('--difficult', action='store_true',
                        help="with difficult")
    parser.add_argument('--threads', type=int, default=1,
                        help='threads num for multi-threads')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convertor = Dota2Xml(args.img_dir, args.txt_dir, args.out_dir, args.difficult)
    t1 = time.time()
    threads = max(1, args.threads)
    logger.info(f"converting with {threads} processes:")
    if threads == 1:
        convertor.convert()
    else:
        convertor.convert_threads(threads)
    t2 = time.time()
    logger.info(f"converted finished in {t2-t1} seconds")
