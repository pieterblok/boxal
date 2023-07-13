# Some basic setup:
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, sys, getopt
import yaml

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import Inference
from detectron2.data.datasets.oct import OCTData

'''
1- read oct data
2- load a pretrained model
3- infere and evaluate
4- load the saved model in file model_final.pth and save the new prediction in the same output dir

usage:
	python infere_visualize.py --config <yaml_file>
'''


def infer_and_visualize(cfg, metadata, data_dicts, n):
    predictor = DefaultPredictor(cfg)
    for d in random.sample(data_dicts, n):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # output format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        groundtruth_instances = utils.annotations_to_instances(d['annotations'], (d['height'], d['width']))
        v_pred = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1#,
                       #instance_mode=ColorMode.IMAGE_BW   # This option is only available for segmentation models
        )
        v_groundtruth = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       scale=1#,
                       #instance_mode=ColorMode.IMAGE_BW   # This option is only available for segmentation models
        )
        out_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_groundtruth = v_groundtruth.draw_dataset_dict(d)

        '''
        plot predictions
        '''
        figure, axis = plt.subplots(1, 2, figsize=(20, 10))
        axis[0].imshow(out_pred.get_image()[:, :, ::-1])
        axis[1].imshow(out_groundtruth.get_image()[:, :, ::-1])
        axis[0].set_title('Predicition')
        axis[1].set_title('Ground Truth')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR,os.path.basename(d["file_name"])))
        plt.close()

    evaluator = COCOEvaluator("test", output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "test")
    print(inference_on_dataset(predictor.model, test_loader, evaluator))

def main(argv):
    opts, args = getopt.getopt(argv,"hc:",["config="])
    for opt, arg in opts:
        if opt == '-h':
            print('python detectron2.py --config <yaml_file>')
            sys.exit()
        elif opt in ("-c", "--config"):
            yaml_filename = arg

    if len(argv) < 1:
        sys.exit("ERROR: input parameters are not provided!\n"
                 "python detectron2.py --config <yaml_file>")

    if not os.path.exists(yaml_filename):
        sys.exit("ERROR: yaml file {} does not exist!".format(yaml_filename))

    # Read the YAML file
    with open(yaml_filename, 'r') as file:
        config = yaml.safe_load(file)

    '''
    get input files/dirs
    '''
    test_dir = config['testdir']
    output_dir = config['outputdir']
    model_path = config['network_config']
    weightsroot = config['weightsroot']
    experiment_name = config['experiment_name']
    strategy = config['strategy']
    weights_dir = os.path.join(weightsroot, experiment_name, strategy)

    if not os.path.exists(test_dir):
        sys.exit("ERROR: {} does not exist!".format(test_dir))
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
    data = OCTData(test_dir)
    DatasetCatalog.register("test", lambda: data.get_oct_dicts()) # register your function
    MetadataCatalog.get("test").set(thing_classes=["damaged"])
    metadata = MetadataCatalog.get("test")
    test_dicts = data.get_oct_dicts()

    '''
    Inference should use the config with parameters that are used in training
    cfg already contains everything we've set previously.
	We changed it a little bit for inference.
    '''
    n = config['inference_num']
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_path))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config['classes'])  # has five classes, one for each layer. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    #cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = config['filter_empty_annotations'] # to use those data with empty annotation
    #cfg.INPUT.FORMAT = "L" # input images are black and white
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold'] #suppress boxes with overlap (IoU) >= this threshold
    cfg.MODEL.WEIGHTS = os.path.join(weights_dir, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['confidence_threshold']   # set a custom testing threshold
    cfg.OUTPUT_DIR = output_dir
    infer_and_visualize(cfg, metadata, test_dicts, n)

if __name__ == "__main__":
    main(sys.argv[1:])
