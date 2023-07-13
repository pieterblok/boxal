# Some basic setup:
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, sys, getopt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
#from detectron2.modeling import build_model
#from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import Inference

'''
1- read oct data
2- load a pretrained model
3- infere and evaluate
4- load the saved model in file model_final.pth and save the new prediction in the same output dir

usage:
	python infere_visualize_oct_seg.py -i <data_dir> -o <output_dir> -m <path_to_model> -w <weights_dir>
'''

def main(argv):
    '''
    get input files/dirs
    '''
    # default values
    data_dir = ""
    output_dir = ""
    model_path = ""
    weights_dir = ""

    opts, args = getopt.getopt(argv,"hi:a:o:m:w:",["data_dir=","output_dir=", "path_to_model=", "weights_dir="])
    for opt, arg in opts:
        if opt == '-h':
            print('\npython infere_visualize_oct_seg.py -i <image_dir> -o <output_dir> -m <path_to_model> -w <weights_dir> \n \
                   data_dir: the directory where the train/val/test directories are located.\n \
                   output_dir: the directory where the inference results are saved, for example: ./output\n \
                   path_to_model: the path to the model yaml file that was used for training, for example: ./COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml\n \
                   weight_dir: the directory where the weights of training are saved, for example: ./weights/exp1/uncertainty/\n\n')
            sys.exit()
        elif opt in ("-i", "--data_dir"):
            data_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-m", "--path_to_model"):
            model_path = arg
        elif opt in ("-w", "--weights_dir"):
            weights_dir = arg
    if len(argv) < 5:
        print("WARNING: default inputs will be used: {}, {}, {}, {}".format(data_dir, output_dir, model_path, weights_dir))

    if not os.path.exists(data_dir):
        sys.exit("ERROR: {} does not exist!".format(data_dir))
    if not os.path.exists(weights_dir):
        sys.exit("ERROR: {} does not exist!".format(weights_dir))
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    '''
    1. register the get_oct_dicts function
        note that here the function is not called
    2. load dataset into a dictionary
		here the function is called
    '''
	data = OCTData(data_dir)
    DatasetCatalog.register("test", lambda: data.get_oct_dicts()) # register your function
    MetadataCatalog.get("test").set(thing_classes=["damaged"])
    metadata = MetadataCatalog.get("test")
    test_dicts = data.get_oct_dicts()

    '''
    Inference should use the config with parameters that are used in training
    cfg already contains everything we've set previously.
	We changed it a little bit for inference.
    '''
	n = 40
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # has five classes, one for each layer. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True # to use those data with empty annotation
    cfg.INPUT.FORMAT = "L" # input images are black and white
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01 #suppress boxes with overlap (IoU) >= this threshold
    cfg.MODEL.WEIGHTS = os.path.join(weights_dir, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0001   # set a custom testing threshold
	Inference(cfg, metadata, test_dicts, n).infer_and_visualize()

if __name__ == "__main__":
    main(sys.argv[1:])
