import random, cv2, os
import matplotlib.pyplot as plt

from detectron2.utils.visualizer import ColorMode
from detectron2.data import detection_utils as utils
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

class Inference:
    '''
    inference class
    '''
    def __init__(self, cfg, metadata, data_dicts, n):
        self.cfg = cfg
        self.metadata = metadata
        self.data_dicts = data_dicts
        self.n = n # number of images to be inferred

    def infer_and_visualize(self):
        predictor = DefaultPredictor(self.cfg)
        for d in random.sample(self.data_dicts, self.n):
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)  # output format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            groundtruth_instances = utils.annotations_to_instances(d['annotations'], (d['height'], d['width']))
            v_pred = Visualizer(im[:, :, ::-1],
                           metadata=self.metadata,
                           scale=1#,
                           #instance_mode=ColorMode.IMAGE_BW   # This option is only available for segmentation models
            )
            v_groundtruth = Visualizer(im[:, :, ::-1],
                           metadata=self.metadata,
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
            plt.savefig(os.path.join(self.cfg.OUTPUT_DIR,os.path.basename(d["file_name"])))
            plt.close()

        evaluator = COCOEvaluator("test", output_dir=self.cfg.OUTPUT_DIR)
        test_loader = build_detection_test_loader(self.cfg, "test")
        print(inference_on_dataset(predictor.model, test_loader, evaluator))
