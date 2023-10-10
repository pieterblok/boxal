# Some basic setup:
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, sys, getopt
import yaml, random, cv2
import matplotlib.pyplot as plt
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.oct import OCTData


'''
1- read oct data
2- load a pretrained model
3- load the saved model in file model_final.pth

usage:
	python convert_torch_model_weights_to_tensorboard_checkpoint.py --config <yaml_file1> <yaml_file2> 
'''


def inference_func(model, image):
    inputs = [{"image": image}]
    return model.inference(inputs, do_postprocess=False)[0]

def main(argv):
    opts, args = getopt.getopt(argv,"hc1:c2:",["config1=", "config2="])
    for opt, arg in opts:
        if opt == '-h':
            print('python convert_torch_model_weights_to_tensorboard_checkpoint.py --config <yaml_file1> <yaml_file2>')
            sys.exit()
        elif opt in ("-c1", "--config1"):
            yaml_filename1 = arg
        elif opt in ("-c2", "--config2"):
            yaml_filename2 = arg


    if len(argv) < 2:
        sys.exit("ERROR: input parameters are not provided!\n"
                 "python convert_torch_model_weights_to_tensorboard_checkpoint.py --config <yaml_file1> <yaml_file2>")

    if not os.path.exists(yaml_filename1):
        sys.exit("ERROR: yaml file {} does not exist!".format(yaml_filename1))
    if not os.path.exists(yaml_filename2):
        sys.exit("ERROR: yaml file {} does not exist!".format(yaml_filename2))

    # Read the YAML file
    with open(yaml_filename1, 'r') as file:
        config1 = yaml.safe_load(file)
    with open(yaml_filename2, 'r') as file:
        config2 = yaml.safe_load(file)


    '''
    get input files/dirs
    '''
    test_dir1 = config1['testdir']
    model_path1 = config1['network_config']
    weightsroot1 = config1['weightsroot']
    resultsroot1 = config1['resultsroot']
    experiment_name1 = config1['experiment_name']
    strategy1 = config1['strategy']
    weights_dir1 = os.path.join(weightsroot1, experiment_name1, strategy1)
    output_dir1 = os.path.join(resultsroot1, experiment_name1, strategy1, 'output')


    if not os.path.exists(test_dir1):
        sys.exit("ERROR: {} does not exist!".format(test_dir1))
    if not os.path.exists(weights_dir1):
        sys.exit("ERROR: {} does not exist!".format(weights_dir1))
    if not os.path.exists(output_dir1):
            os.mkdir(output_dir1)

    test_dir2 = config2['testdir']
    model_path2 = config2['network_config']
    weightsroot2 = config2['weightsroot']
    resultsroot2 = config2['resultsroot']
    experiment_name2 = config2['experiment_name']
    strategy2 = config2['strategy']
    weights_dir2 = os.path.join(weightsroot2, experiment_name2, strategy2)
    output_dir2 = os.path.join(resultsroot2, experiment_name2, strategy2, 'output')

    if not os.path.exists(test_dir2):
        sys.exit("ERROR: {} does not exist!".format(test_dir2))
    if not os.path.exists(weights_dir2):
        sys.exit("ERROR: {} does not exist!".format(weights_dir2))
    if not os.path.exists(output_dir2):
            os.mkdir(output_dir2)


    '''
    Inference should use the config with parameters that are used in training
    cfg already contains everything we've set previously.
	We changed it a little bit for inference.
    '''
    cfg1 = get_cfg()
    cfg1.merge_from_file(model_zoo.get_config_file(model_path1))
    cfg1.MODEL.ROI_HEADS.NUM_CLASSES = len(config1['classes'])  # has five classes, one for each layer. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg1.MODEL.ROI_HEADS.NMS_THRESH_TEST = config1['nms_threshold'] #suppress boxes with overlap (IoU) >= this threshold
    cfg1.MODEL.WEIGHTS = os.path.join(weights_dir1, "model_final.pth")  # path to the model we just trained
    cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config1['confidence_threshold']   # set a custom testing threshold
    cfg1.OUTPUT_DIR = output_dir1


    from detectron2.modeling import build_model
    model1 = build_model(cfg1)
    #from detectron2.checkpoint import DetectionCheckpointer
    state_dict1 = torch.load(cfg1.MODEL.WEIGHTS) # state_dict['model']=model.state_dict()
    model1.load_state_dict(state_dict1['model'])
    model1.eval()

    cfg2 = get_cfg()
    cfg2.merge_from_file(model_zoo.get_config_file(model_path2))
    cfg2.MODEL.ROI_HEADS.NUM_CLASSES = len(config2['classes'])  # has five classes, one for each layer. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg2.MODEL.ROI_HEADS.NMS_THRESH_TEST = config2['nms_threshold'] #suppress boxes with overlap (IoU) >= this threshold
    cfg2.MODEL.WEIGHTS = os.path.join(weights_dir2, "model_final.pth")  # path to the model we just trained
    cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config2['confidence_threshold']   # set a custom testing threshold
    cfg2.OUTPUT_DIR = output_dir2
    
    model2 = build_model(cfg2)
    #from detectron2.checkpoint import DetectionCheckpointer
    state_dict2 = torch.load(cfg2.MODEL.WEIGHTS) # state_dict['model']=model.state_dict()
    model2.load_state_dict(state_dict2['model'])
    model2.eval()
    
    for key in state_dict1['model'].keys():
        if torch.all(torch.eq(model1.state_dict()[key], model2.state_dict()[key]))==True: print("#######----- same weights -----#######", key)
        else: print("#######----- different weights -----#######", key)
    print("#######", state_dict1['model']['backbone.fpn_lateral2.weight'].shape)
 
    '''
    BELOW: Attempt to visualize model weight outputs by converting model.pth to tensorboard_checkpoint. NOT successful.
    '''

    '''
    1. register the get_oct_dicts function
        note that here the function is not called
    2. load dataset into a dictionary
		here the function is called
    '''
    data = OCTData(test_dir1)
    DatasetCatalog.register("test", lambda: data.get_oct_dicts()) # register your function
    MetadataCatalog.get("test").set(thing_classes=["damaged"])
    metadata = MetadataCatalog.get("test")
    test_dicts = data.get_oct_dicts()

    '''
    Detectron2 models expect a dictionary or a list of dictionaries as input by default.
    So you can not directly use torch.jit.trace function.
    But they provide a wrapper, called TracingAdapter, that allows models to take a tensor or a tuple of tensors as input.
    '''
    from detectron2.export.flatten import TracingAdapter, flatten_to_tuple
    height = test_dicts[0]['height']
    width = test_dicts[0]['width']
    im = torch.randn(3,height, width)  # Replace with appropriate input size
    #im = torch.tensor(cv2.imread(test_dicts[0]["file_name"]))
    wrapper = TracingAdapter(model1, im, inference_func)
    wrapper.eval()
    traced_model = torch.jit.trace(wrapper, (im,))
    #traced_model.save(os.path.join(weights_dir, "model_final.pt"))
    import tensorflow as tf
    tf_model = tf.Module()
    tf_model.forward = traced_model
    tf.saved_model.save(tf_model, os.path.join(weights_dir1, "tensorboard_checkpoint"))

    #from tensorboardX import SummaryWriter
    #writer = SummaryWriter()
    #for name, param in state_dict1['model'].items():
    #    writer.add_histogram(name, param, global_step=0)
    #writer.close()

    import onnx
    torch.onnx.export(traced_model, im, 'model.onnx', verbose=True, opset_version=11)


if __name__ == "__main__":
    main(sys.argv[1:])



'''
tf_model = tf.Module()
tf_model.forward = traced_model

# Save the TensorFlow checkpoint
tf.saved_model.save(tf_model, 'tensorboard_checkpoint')

'''
