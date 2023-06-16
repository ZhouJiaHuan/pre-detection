import sys
sys.path.append('.')
import time
import argparse
import logging
from predet.transform.image_slice import ImageSlice

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="image slice with xml annotations")
    parser.add_argument('img_dir', type=str,
                        help='dota image directory')
    parser.add_argument('xml_dir', type=str, default=None,
                        help='dota rect label directory, default None')
    parser.add_argument('out_dir', type=str,
                        help='output image and xml (optional) directory')
    parser.add_argument('--width', type=int, default=640,
                        help='slice patch width, default 640')
    parser.add_argument('--height', type=int, default=640,
                        help='slice patch height, default 640')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='slice patch overlap, default 0.5')
    parser.add_argument('--threads', type=int, default=1,
                        help='threads num for multi-threads, default 1')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    patch_size = (args.width, args.height)
    overlap = args.overlap
    img_slice = ImageSlice(args.img_dir, args.out_dir, args.xml_dir,
                           slice_size=patch_size, overlap_ratio=(overlap, overlap),
                           min_area_ratio=0.2, ext='png')
    
    t1 = time.time()
    threads = max(1, args.threads)
    logger.info(f"converting with {threads} processes:")
    if threads == 1:
        img_slice.run()
    else:
        img_slice.run_thread(threads)
    t2 = time.time()
    logger.info(f"converted finished in {t2-t1} seconds")