# @Author: Pieter Blok
# @Date:   2021-03-25 18:48:22
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2022-02-11 10:52:00

## Use a trained model to auto-annotate unlabelled images

## general libraries
import argparse
import numpy as np
import os
import json
from shutil import copyfile
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

## detectron2-libraries 
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

supported_cv2_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")


def check_direxcist(dir):
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)  # make new folder


def check_annotation(filename):
    valid = False

    with open(filename, 'r') as json_file:
        try:
            data = json.load(json_file)
            if 'annotation' in data:
                if len(data['annotation']['objects']) > 0:
                    valid = True
        except:
            pass

    return valid



def matching_images_and_annotations(rootdir):
    print("Checking which images and annotations are valid")
    images = []
    images_basenames = []
    annotations = []
    annotations_basenames = []

    matching_images = []
    matching_annotations = []

    if os.path.isdir(rootdir):
        for root, dirs, files in tqdm(list(os.walk(rootdir))):
            for name in files:
                subdir = root.split(rootdir)
                all('' == s for s in subdir)
                
                if subdir[1].startswith('/'):
                    subdirname = subdir[1][1:]
                else:
                    subdirname = subdir[1]

                if name.lower().endswith(supported_cv2_formats):
                    if all('' == s for s in subdir):
                        images.append(name)
                        images_basenames.append(os.path.splitext(name)[0])
                    else:
                        images.append(os.path.join(subdirname, name))
                        one_folder_up = os.path.join(*(subdirname.split(os.path.sep)[1:]))
                        images_basenames.append(os.path.splitext(os.path.join(one_folder_up, name))[0])

                if name.endswith(".json") or name.endswith(".xml"):
                    if all('' == s for s in subdir):
                        if check_annotation(os.path.join(rootdir, name)):
                            annotations.append(name)
                            annotations_basenames.append(os.path.splitext(name)[0])
                    else:
                        if check_annotation(os.path.join(rootdir, subdirname, name)):
                            annotations.append(os.path.join(subdirname, name))
                            one_folder_up = os.path.join(*(subdirname.split(os.path.sep)[1:]))
                            annotations_basenames.append(os.path.splitext(os.path.join(one_folder_up, name))[0])
    
        images.sort()
        images_basenames.sort()
        annotations.sort()
        annotations_basenames.sort()

        matching_images_annotations = list(set(images_basenames) & set(annotations_basenames))
        matching_images = [img for img in images if os.path.splitext(os.path.join(*(img.split(os.path.sep)[1:])))[0] in matching_images_annotations]
        matching_annotations = [annot for annot in annotations if os.path.splitext(os.path.join(*(annot.split(os.path.sep)[1:])))[0] in matching_images_annotations]

        print("{:d} valid images found!".format(len(matching_images)))
        print("{:d} valid annotations found!".format(len(matching_annotations)))
        print("")

    return matching_images, matching_annotations


def get_class_names(rootdir, annotations):
    unique_class_names = []

    for j in tqdm(range(len(annotations))):
        filename = os.path.join(rootdir, annotations[j])
        with open(filename, 'r') as json_file:
            try:
                data = json.load(json_file)
                for p in range(len(data['annotation']['objects'])):
                    label = data['annotation']['objects'][p]['label']
                    if len(label) > 0:
                        classname = data['annotation']['objects'][p]['label'][0]
                        if classname not in unique_class_names:
                            unique_class_names.append(classname)
            except:
                pass

    return unique_class_names


def get_class_counts(rootdir, annotations, class_names):
    class_counts = [0] * len(class_names)

    for j in tqdm(range(len(annotations))):
        filename = os.path.join(rootdir, annotations[j])
        with open(filename, 'r') as json_file:
            try:
                data = json.load(json_file)
                for p in range(len(data['annotation']['objects'])):
                    label = data['annotation']['objects'][p]['label']
                    if len(label) > 0:
                        classname = data['annotation']['objects'][p]['label'][0]
                        class_id = class_names.index(classname)
                        class_counts[class_id] += 1
            except:
                pass

    return class_counts


def get_minority_classes(class_names, class_counts, percentage):
    minority_classes = []
    percs = np.array([class_counts[c]/sum(class_counts) for c in range(len(class_counts))]).astype(np.float32)
    for c in range(len(class_counts)):
        if percs[c] < (percentage/100):
            minority_classes.append(class_names[c])
    return minority_classes


def organize_images_and_annotations(rootdir, matching_images, matching_annotations):
    print("Copying the valid images and annotations")
    for i in tqdm(range(len(matching_images))):
        imagename = os.path.basename(matching_images[i])
        annotname = os.path.basename(matching_annotations[i])
        
        one_folder_up = os.path.join(*(matching_images[i].split(os.path.sep)[1:]))
        new_dir = os.path.join(rootdir, os.path.dirname(one_folder_up))
        check_direxcist(new_dir)

        copyfile(os.path.join(rootdir, matching_images[i]), os.path.join(new_dir, imagename))
        copyfile(os.path.join(rootdir, matching_annotations[i]), os.path.join(new_dir, annotname))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='datasets/fdf_images', help='the root-folder with fdf images and annotations that need to be organized')
    opt = parser.parse_args()
    print(opt)
    print()

    matching_images, matching_annotations = matching_images_and_annotations(opt.root_dir)

    class_names = get_class_names(opt.root_dir, matching_annotations)
    class_counts = get_class_counts(opt.root_dir, matching_annotations, class_names)
    minority_classes = get_minority_classes(class_names, class_counts, 10)

    print("\r\nThe class-names are: " + ', '.join(class_names) + "\r\n")
    string_ints = [str(int) for int in class_counts]
    print("The class-counts: " + ', '.join(string_ints) + "\r\n")
    print("The minority classes are: " + ', '.join(minority_classes) + "\r\n")
    
    organize_images_and_annotations(opt.root_dir, matching_images, matching_annotations)