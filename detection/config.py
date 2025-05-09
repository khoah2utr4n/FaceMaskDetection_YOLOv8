import os

DATASET_DIR = 'datasets'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images/')
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, 'annotations/')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels/')
ALL_FILENAMES = [f[:-4] for f in os.listdir(IMAGES_DIR)]
CLASS_INDEXS = {'without_mask': 0, 'with_mask': 1, 'mask_weared_incorrect': 2}
CLASS_NAMES = {0: 'without_mask', 1: 'with_mask', 2: 'mask_weared_incorrect'}