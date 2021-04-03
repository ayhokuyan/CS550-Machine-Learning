

from PIL import Image
import os
import shutil

import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(directory, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(directory), '%s is not a valid directory' % dir
    for root, dirs, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    if len(images) < max_dataset_size:
        return []
    unused_images = np.array(images[max_dataset_size:])
    np.random.shuffle(unused_images)
    return unused_images


def filter_paths(img_paths, end_seq, data_size):
    paths = []
    for path in list(img_paths):
        data_type = path.split("/")[1]
        if data_type.endswith(end_seq):
            paths.append(path)
    return paths[:data_size]


def default_loader(path):
    return Image.open(path).convert("RGB")


def save_dataset(save_path, img_names, reset=True):

    if reset:
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
            os.mkdir(save_path)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    checked = []
    for img_path in img_names:
        img = default_loader(img_path)
        path_to_save = os.path.join(save_path, os.path.split(img_path)[1])
        if not path_to_save.endswith(tuple(IMG_EXTENSIONS)):
            print(img_path)
        if path_to_save in checked:
            path = path_to_save.split(".")
            path_to_save = ".".join(["".join([path[0], "_1"]), path[1]])
        img.save(path_to_save)
        checked.append(path_to_save)






