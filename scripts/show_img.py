import cv2 as cv
import os
import sys
import argparse

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory training dataset')
    args = parser.parse_args()

    imgFile = cv.imread(os.path.join(ROOT_DIR, args.image))

    cv.imshow('dst_rt', imgFile)
    cv.waitKey(0)
    cv.destroyAllWindows()