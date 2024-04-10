# @Author: Pieter Blok
# @Date:   2024-04-10 14:48:22

## Uncertainty estimation on unlabelled images for object detection in Detectron2

## general libraries
import sys
import argparse
import yaml
import numpy as np
import torch
import os
import cv2
import csv
import logging
import pickle
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

## detectron2-libraries 
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm

## libraries that are specific for dropout training
from active_learning.strategies.dropout import FastRCNNConvFCHeadDropout, FastRCNNOutputLayersDropout, Res5ROIHeadsDropout
from active_learning.sampling import observations, prepare_complete_dataset, calculate_repeat_threshold, calculate_iterations, list_files, visualize_uncertainties
from active_learning.sampling.montecarlo_dropout import MonteCarloDropoutHead
from active_learning.heuristics import uncertainty

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 10,10
def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()


supported_cv2_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tiff", ".tif")
supported_annotation_formats = (".json", ".xml")


## initialize the logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s \n'
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
file_handler = logging.StreamHandler()
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)


def check_direxcist(dir):
    if dir is not None:
        if not os.path.exists(dir):
            os.makedirs(dir)  # make new folder


def store_initial_val_value(val_value_init, weightsfolder):
    with open(os.path.join(weightsfolder, 'val_value_init.pkl'), 'wb') as f1:
        pickle.dump(val_value_init, f1)


def load_initial_val_value(weightsfolder):
    with open(os.path.join(weightsfolder, 'val_value_init.pkl'), 'rb') as f1:
        val_value_init = pickle.load(f1)
    return val_value_init
    

def calculate_max_entropy(classes):
    least_confident = np.divide(np.ones(len(classes)), len(classes)).astype(np.float32)
    probs = torch.from_numpy(least_confident)
    max_entropy = torch.distributions.Categorical(probs).entropy()
    return max_entropy
    

def train(config, weightsfolder, gpu_num, iter, val_value, dropout_probability, init=False, skip_training=False):    
    ## Hook to automatically save the best checkpoint
    class BestCheckpointer(HookBase):
        def __init__(self, iter, eval_period, val_value, metric):
            self.iter = iter
            self._period = eval_period
            self.val_value = val_value
            self.metric = metric
            self.logger = setup_logger(name="d2.checkpointer.best")
            
        def store_best_model(self):
            metric = self.trainer.storage._latest_scalars

            try:
                current_value = metric[self.metric][0]
                try:
                    highest_value = metric['highest_value'][0]
                except:
                    highest_value = self.val_value

                self.logger.info("current-value ({:s}): {:.2f}, highest-value ({:s}): {:.2f}".format(self.metric, current_value, self.metric, highest_value))

                if current_value > highest_value:
                    self.logger.info("saving best model...")
                    self.trainer.checkpointer.save("best_model_{:s}".format(str(iter).zfill(3)))
                    self.trainer.storage.put_scalar('highest_value', current_value)
                    comm.synchronize()
            except:
                pass

        def after_step(self):
            next_iter = self.trainer.iter + 1
            is_final = next_iter == self.trainer.max_iter
            if is_final or (self._period > 0 and next_iter % self._period == 0):
                self.store_best_model()
            self.trainer.storage.put_scalars(timetest=12)


    ## CustomTrainer with evaluator and automatic checkpoint-saver
    class CustomTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, {"bbox"}, False, output_folder)

        def build_hooks(self):
            hooks = super().build_hooks()
            hooks.insert(-1, BestCheckpointer(iter, cfg.TEST.EVAL_PERIOD, val_value, 'bbox/AP'))
            return hooks


    if not init:
        DatasetCatalog.remove("train")
        DatasetCatalog.remove("val")

    register_coco_instances("train", {}, os.path.join(config['dataroot'], "train.json"), config['traindir'])
    train_metadata = MetadataCatalog.get("train")
    dataset_dicts_train = DatasetCatalog.get("train")

    register_coco_instances("val", {}, os.path.join(config['dataroot'], "val.json"), config['valdir'])
    val_metadata = MetadataCatalog.get("val")
    dataset_dicts_val = DatasetCatalog.get("val")

    ## add dropout layers to the architecture of Mask R-CNN
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config['network_config']))
    cfg.MODEL.ROI_BOX_HEAD.DROPOUT_PROBABILITY = dropout_probability

    if any(x in config['network_config'] for x in ["FPN", "DC5"]):
        cfg.MODEL.ROI_BOX_HEAD.NAME = 'FastRCNNConvFCHeadDropout'
        cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeadsDropout'
    elif any(x in config['network_config'] for x in ["C4"]):
        cfg.MODEL.ROI_HEADS.NAME = 'Res5ROIHeadsDropout'
        
    cfg.MODEL.ROI_HEADS.SOFTMAXES = False
    cfg.OUTPUT_DIR = weightsfolder


    ## initialize the network weights, with an option to do the transfer-learning on previous models
    if config['pretrained_weights'].lower().endswith(".yaml"):
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config['pretrained_weights'])
    elif config['pretrained_weights'].lower().endswith((".pth", ".pkl")):
        cfg.MODEL.WEIGHTS = config['pretrained_weights']


    ## initialize the train-sampler
    cfg.DATALOADER.SAMPLER_TRAIN = config['train_sampler']
    if cfg.DATALOADER.SAMPLER_TRAIN == 'RepeatFactorTrainingSampler':
        repeat_threshold = calculate_repeat_threshold(config, dataset_dicts_train)
        cfg.DATALOADER.REPEAT_THRESHOLD = repeat_threshold
        
    max_iterations, steps = calculate_iterations(config, dataset_dicts_train)

    ## initialize the training parameters  
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.NUM_GPUS = gpu_num
    cfg.DATALOADER.NUM_WORKERS = config['num_workers']
    cfg.SOLVER.IMS_PER_BATCH = config['train_batch_size']
    cfg.SOLVER.WEIGHT_DECAY = config['weight_decay']
    cfg.SOLVER.LR_POLICY = config['learning_policy']
    cfg.SOLVER.BASE_LR = config['learning_rate']
    cfg.SOLVER.GAMMA = config['gamma']
    cfg.SOLVER.WARMUP_ITERS = config['warmup_iterations']
    cfg.SOLVER.MAX_ITER = max_iterations
    cfg.SOLVER.STEPS = steps

    if config['checkpoint_period'] == -1:
        cfg.SOLVER.CHECKPOINT_PERIOD = (cfg.SOLVER.MAX_ITER+1)
    else:
        cfg.SOLVER.CHECKPOINT_PERIOD = config['checkpoint_period']

    cfg.TEST.EVAL_PERIOD = config['eval_period']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(config['classes'])
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if not skip_training:
        trainer = CustomTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    try:
        val_value_output = trainer.storage._latest_scalars['highest_value'][0]
    except: 
        val_value_output = val_value

    return cfg, dataset_dicts_train, dataset_dicts_val, val_value_output


def evaluate(cfg, config, dataset_dicts_train, dataset_dicts_val, weightsfolder, resultsfolder, csv_name, iter, init=False):      
    if not init:
        DatasetCatalog.remove("test")
    register_coco_instances("test", {}, os.path.join(config['dataroot'], "test.json"), config['testdir'])
    test_metadata = MetadataCatalog.get("test")
    dataset_dicts_test = DatasetCatalog.get("test")

    cfg.OUTPUT_DIR = weightsfolder

    if os.path.isfile(os.path.join(cfg.OUTPUT_DIR, "best_model_{:s}.pth".format(str(iter).zfill(3)))):
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "best_model_{:s}.pth".format(str(iter).zfill(3)))
    else:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config['confidence_threshold']   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config['nms_threshold']
    cfg.DATASETS.TEST = ("test",)

    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("test", {"bbox"}, False, output_dir=resultsfolder)
    val_loader = build_detection_test_loader(cfg, "test")
    eval_results = inference_on_dataset(model, val_loader, evaluator)
    
    bbox_strings = [c.replace(c, 'AP-' + c) for c in config['classes']]

    if len(config['classes']) == 1:
        bbox_values = [round(eval_results['bbox']['AP'], 1) for s in bbox_strings]
    else:
        bbox_values = [round(eval_results['bbox'][s], 1) for s in bbox_strings]

    write_values = [len(dataset_dicts_train), len(dataset_dicts_val), len(dataset_dicts_test), round(eval_results['bbox']['AP'], 1)] + bbox_values

    with open(os.path.join(resultsfolder, csv_name), 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(write_values)

    return cfg, dataset_dicts_test


def uncertainty_sampling(pool_list, cfg, config, max_entropy, mcd_iterations, save_df, write_dir):
    cfg.MODEL.ROI_HEADS.SOFTMAXES = True
    predictor = MonteCarloDropoutHead(cfg, mcd_iterations)
    device = cfg.MODEL.DEVICE
    csv_file = os.path.join(config['resultsroot'], config['experiment_name'], config['csv'])

    if len(pool_list) > 0:
        ## find the images from the pool_list the algorithm is most uncertain about
        for d in tqdm(range(len(pool_list))):
            filename = pool_list[d]
            if os.path.isfile(os.path.join(config['sampledir'], filename)):
                img = cv2.imread(os.path.join(config['sampledir'], filename))
                outputs = predictor(img)

                obs = observations(outputs, device, config['iou_thres'])
                img_uncertainty, u_sem, u_spl, u_n, mean_bboxes = uncertainty(obs, mcd_iterations, max_entropy, device) ## reduce the iterations when facing a "CUDA out of memory" error
                img_vis = visualize_uncertainties(img, mean_bboxes, img_uncertainty, u_sem, u_spl, u_n)
                cv2.imwrite(os.path.join(write_dir, filename), img_vis)

                for box, t_u, sem_u, spl_u, occ_u in zip(mean_bboxes, img_uncertainty, u_sem, u_spl, u_n):
                    x1, y1, x2, y2 = box.detach().cpu().numpy().astype(np.int32)
                    usem = float(sem_u.detach().cpu().numpy())
                    uspl = float(spl_u.detach().cpu().numpy())
                    uocc = float(occ_u.detach().cpu().numpy())
                    utotal = float(t_u.detach().cpu().numpy())

                    cur_data = {
                        'image_name': filename,
                        'bbox_x1': x1,
                        'bbox_y1': y1,
                        'bbox_x2': x2,
                        'bbox_y2': y2,
                        'semantic_uncertainty': round(usem, 2),
                        'spatial_uncertainty': round(uspl, 2),
                        'occurrence_uncertainty': round(uocc, 2),
                        'total_uncertainty': round(utotal, 2),
                        }
                    
                    save_df = pd.concat([save_df, pd.DataFrame([cur_data])], ignore_index=True)
                    save_df.to_csv(csv_file, mode='w+', index=False)


if __name__ == "__main__":
    logger.addHandler(file_handler)
    logger.info("Starting uncertainty sampling")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='uncertainty_sampling.yaml', help='yaml with the training parameters')
    args = parser.parse_args()

    try:
        with open(args.config, 'rb') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        logger.error(f"Could not find configuration-file: {args.config}")
        sys.exit("Closing application")

    print("Configuration:")
    for key, value in config.items():
        print(key, ':', value)

    config["group_val_test_set"] = False
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_visible_devices']
    gpu_num = len(config['cuda_visible_devices'])

    weightsfolder = os.path.join(config['weightsroot'], config['experiment_name'])
    resultsfolder = os.path.join(config['resultsroot'], config['experiment_name'])
    output_image_folder = os.path.join(config['resultsroot'], config['experiment_name'], "images")
    check_direxcist(config['dataroot'])
    check_direxcist(output_image_folder)

    max_entropy = calculate_max_entropy(config['classes'])
    prepare_complete_dataset(config)

    bbox_strings = [c.replace(c, 'mAP-' + c) for c in config['classes']]
    write_strings = ['train_size', 'val_size', 'test_size', 'mAP'] + bbox_strings
    csv_name = config['experiment_name'] + '.csv'
    with open(os.path.join(resultsfolder, csv_name), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(write_strings)

    if config['pretrained_weights'].lower().endswith(".yaml"):
        cfg, dataset_dicts_train, dataset_dicts_val, val_value = train(config, weightsfolder, gpu_num, 0, 0, config['dropout_probability'], init=True)
        store_initial_val_value(val_value, weightsfolder)
    elif config['pretrained_weights'].lower().endswith((".pth", ".pkl")):
        val_value_init = load_initial_val_value(weightsfolder)
        cfg, dataset_dicts_train, dataset_dicts_val, val_value = train(config, weightsfolder, gpu_num, 1, val_value_init, config['dropout_probability'], init=True, skip_training=True)

    cfg, dataset_dicts_test = evaluate(cfg, config, dataset_dicts_train, dataset_dicts_val, weightsfolder, resultsfolder, csv_name, 0, init=True)

    columns = ['image_name', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'semantic_uncertainty', 'spatial_uncertainty', 'occurrence_uncertainty', 'total_uncertainty']
    save_df = pd.DataFrame(columns=columns)

    sample_images, _ = list_files(config['sampledir'])
    uncertainty_sampling(sample_images, cfg, config, max_entropy, config['mcd_iterations'], save_df, output_image_folder)

    logger.info("Uncertainty sampling is finished!")