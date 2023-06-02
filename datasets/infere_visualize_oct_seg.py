# Some basic setup:
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, glob
import cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

'''
1- read oct data 
2- load a pretrained model
3- infere and evaluate

it loads the saved model in file model_final.pth in the default output_dir which is "output"
and will save the new prediction in the same output dir
'''

'''
TODO call this from a class
'''
def get_oct_dicts(img_dir, annot_dir):
    from detectron2.structures import BoxMode
    dataset_dicts = []
    for idx, f in enumerate(glob.glob(img_dir+"/*.png")):
        json_file = os.path.join(annot_dir, os.path.basename(f)[:-4]+".json")
        print(json_file)
        img_annot = json.load(open(json_file))
        record={}
        record["file_name"] = f
        record["image_id"] = idx
        record["height"] = img_annot['imageHeight']
        record["width"] = img_annot['imageWidth']
        
        objs = []
        #print(img_annot["shapes"])
        for box in img_annot["shapes"]:
            print(box['points'])
            p1 = box['points'][0]
            p2 = box['points'][1]
            obj = {
                "bbox": [int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        print(objs)
        #if x_box[0]==x_box[1]==0:
        #    record["annotations"] = []
        dataset_dicts.append(record)
    return dataset_dicts

def infer(img_dir, annot_dir, output_dir, model_path, weight_dir):
    '''
    load the input files
    '''
    for d in ["test"]:
        DatasetCatalog.register("oct_" + d, lambda d=d: get_oct_dicts(img_dir, annot_dir))
        MetadataCatalog.get("oct_" + d).set(thing_classes=["damaged"])
    oct_metadata = MetadataCatalog.get("oct_train")

    '''
    load the model
    '''
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_dir))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # has five classes, one for each layer. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False # to use those data with empty annotation
    cfg.INPUT.FORMAT = "L" # input images are black and white
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.01 #suppress boxes with overlap (IoU) >= this threshold
    #cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64, 128 , 256, 512]]
    #cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.002, 0.01, 0.02, 0.05]]

    '''
    inference and evaluation
    Inference should use the config with parameters that are used in training
    cfg now already contains everything we've set previously. We changed it a little bit for inference:
    '''
    cfg.MODEL.WEIGHTS = os.path.join(weights_dir, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0001   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = get_oct_dicts(img_dir, annot_dir)
    for d in random.sample(dataset_dicts, 5):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        from detectron2.data import detection_utils as utils
        groundtruth_instances = utils.annotations_to_instances(d['annotations'], (d['height'],d['height']))
        v_pred = Visualizer(im[:, :, ::-1],
                       metadata=oct_metadata, 
                       scale=1, 
                       instance_mode=ColorMode.IMAGE_BW   # This option is only available for segmentation models
        )
        v_groundtruth = Visualizer(im[:, :, ::-1],
                       metadata=oct_metadata,
                       scale=1, 
                       instance_mode=ColorMode.IMAGE_BW   # This option is only available for segmentation models
        )
        out_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_groundtruth = v_groundtruth.draw_dataset_dict(d)

        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(1, 2, figsize=(20, 10))
        axis[0].imshow(out_pred.get_image()[:, :, ::-1])
        axis[1].imshow(out_groundtruth.get_image()[:, :, ::-1])
        axis[0].set_title('Predicition')
        axis[1].set_title('Ground Truth')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,os.path.basename(d["file_name"])))   
        
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    evaluator = COCOEvaluator("oct_test", output_dir)
    test_loader = build_detection_test_loader(cfg, "oct_test")
    print(inference_on_dataset(predictor.model, test_loader, evaluator))


def main(argv):
    '''
    get input files/dirs
    '''
    # default values
    img_dir = "/projects/parisa/data/test_boxal/faster_rcnn/"
    annot_dir = "/projects/parisa/data/test_boxal/faster_rcnn/"
    output_dir = "./output/"
    model_dir = "./COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    weight_dir = "./weights/exp1/uncertainty/"
    
    opts, args = getopt.getopt(argv,"hi:a:o:m:w:",["img_dir=","annot_dir=","output_dir=", "path_to_model=", "weight_dir="])
    for opt, arg in opts:
        if opt == '-h':
            print('python infere_visualize_oct_seg.py -i <image_dir> -a <annotation_dir> -o <output_dir> -m <path_to_model> -w <weight_dir>')
            sys.exit()
        elif opt in ("-i", "--img_dir"):
            img_dir = arg
        elif opt in ("-a", "--annot_dir"):
            annot_dir = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-m", "--path_to_model"):
            model_path = arg
        elif opt in ("-w", "--weight_dir"):
            weight_dir = arg
    if len(argv) < 6:
        print("WARNING: default inputs will be used: {}, {}, {}, {}, {}".format(img_dir, annot_dir, output_dir, model_path, weight_dir))

    if not os.path.exists(img_dir):
        sys.exit("ERROR: {} does not exist!".format(img_dir))
    if not os.path.exists(annot_dir):
        sys.exit("ERROR: {} does not exist!".format(annot_dir))
    if not os.path.exists(weight_dir):
        sys.exit("ERROR: {} does not exist!".format(weight_dir))
    if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
    infer(img_dir, annot_dir, output_dir, model_path, weight_dir)

if __name__ == "__main__":
    main(sys.argv[1:])
