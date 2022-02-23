# @Author: Pieter Blok
# @Date:   2021-03-25 18:48:22
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2022-02-23 12:34:01

## Use a trained model to auto-annotate unlabelled images

## general libraries
import argparse
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile
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

## libraries for preparing the datasets
from active_learning.sampling import list_files, visualize_cnn, write_cvat_annotations, write_darwin_annotations, write_labelme_annotations, write_supervisely_annotations, mkdir_supervisely, find_tiff_files, convert_tiffs, create_zipfile


if __name__ == "__main__":
    ## Initialize the args_parser and some variables
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='datasets/train', help='the folder with images that need to be auto-annotated')
    parser.add_argument('--network_config', type=str, default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml', help='specify the backbone of the CNN')
    parser.add_argument('--classes', nargs="+", default=[], help='a list with all the classes')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='confidence threshold for the inference of the trained object detector')
    parser.add_argument('--nms_thres', type=float, default=0.2, help='non-maximum suppression threshold for the inference of the trained object detector')
    parser.add_argument('--weights_file', type=str, default=[], help='weight-file (.pth)')
    parser.add_argument('--export_format', type=str, default='labelme', help='Choose either "labelme", "cvat", "darwin", "supervisely"')
    parser.add_argument('--supervisely_meta_json', type=str, default="", help='the file location of the meta.json for supervisely export')

    ## Load the args_parser and initialize some variables
    opt = parser.parse_args()
    print(opt)
    print()

    if opt.export_format == "supervisely":
        tiff_images, _ = find_tiff_files(opt.img_dir)
        if tiff_images != []:
            print("\n{:d} images found with .tiff or .tif extension: unfortunately Supervisely does not support these extensions".format(len(tiff_images)))
            input("Press Enter to automatically convert the {:d} images to .png extension".format(len(tiff_images)))
            convert_tiffs(tiff_images)
        out_ann_dir = mkdir_supervisely(opt.img_dir, os.path.dirname(opt.img_dir), opt.supervisely_meta_json)

    use_coco = False
    images, annotations = list_files(opt.img_dir)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(opt.network_config))
    cfg.NUM_GPUS = 1

    if opt.weights_file.lower().endswith(".yaml"):
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(opt.weights_file)
        use_coco = True
    elif opt.weights_file.lower().endswith((".pth", ".pkl")):
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(opt.classes)
        cfg.MODEL.WEIGHTS = opt.weights_file

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = opt.conf_thres
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = opt.nms_thres
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.ROI_HEADS.SOFTMAXES = False

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    predictor = DefaultPredictor(cfg)

    for i in tqdm(range(len(images))):
        imgname = images[i]
        basename = os.path.basename(imgname)
        img = cv2.imread(os.path.join(opt.img_dir, imgname))
        height, width, _ = img.shape

        # Do the image inference and extract the outputs from the object detector
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        boxes = instances.pred_boxes.tensor.numpy()

        if use_coco:
            v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            class_labels = v.metadata.get("thing_classes", None)
        else:
            class_labels = opt.classes

        class_names = []
        for h in range(len(classes)):
            class_id = classes[h]
            class_name = class_labels[class_id]
            class_names.append(class_name)

        img_vis = visualize_cnn(img, classes, scores, boxes, class_labels)
        
        if opt.export_format == "cvat":
            write_cvat_annotations(opt.img_dir, basename, class_names, boxes, height, width)

        if opt.export_format == "darwin":
            write_darwin_annotations(opt.img_dir, basename, class_names, boxes, height, width)
        
        if opt.export_format == "labelme":
            write_labelme_annotations(opt.img_dir, basename, class_names, boxes, height, width)

        if opt.export_format == "supervisely":
            write_supervisely_annotations(out_ann_dir, basename, class_names, boxes, height, width, opt.supervisely_meta_json)


    ## zip the cvat xml-files so it's easier to do the export
    if opt.export_format == "cvat":
        create_zipfile(opt.img_dir)