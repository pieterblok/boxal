# Uncertainty sampling on unlabelled images

## Installation
Linux/Ubuntu: [INSTALL.md](INSTALL.md)<br/><br/>
Windows: [INSTALL_Windows.md](INSTALL_Windows.md)
<br/> <br/>

## How to use this software
Open a terminal (Ctrl+Alt+T):
```console
(base) user@computer:~$ cd boxal
(base) user@computer:~$ git checkout uncertainty
(base) user@computer:~/boxal$ conda activate boxal
(boxal) user@computer:~/boxal$ python uncertainty_sampling.py --config uncertainty_sampling.yaml
```
<br/>
Change the following settings in the uncertainty_sampling.yaml file: <br/>

| Setting        	| Description           														|
| ----------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| weightsroot	        | The file directory where the weight-files are stored											|
| resultsroot		| The file directory where the result-files are stored 											|
| dataroot	 	| The root directory where all image-files are stored											|
| traindir	 	| The file directory where the training images and annotations are stored								|
| valdir	 	| The file directory where the validation images and annotations are stored								|
| testdir	 	| The file directory where the test images and annotations are stored									|
| sampledir	 	| The file directory containing the unlabelled images where you want to estimate the model's uncertainties				|
| experiment_name	| Specify the name of your experiment													|
| csv		 	| CSV file containing the uncertainty metrics per object per image. Stored in 'resultsroot/experiment_name'. **Visual output is stored in 'resultsroot/experiment_name/images'**				|
| network_config	| The configuration-file (.yaml) file for the object detector (choose either Fast R-CNN or Faster R-CNN, Retinanet is not supported) (see the folder './configs')																			|
| pretrained_weights	| The pretrained weights to start the uncertainty sampling. Either specify the network_config (.yaml) or a custom weights-file (.pth or .pkl)|
| cuda_visible_devices 	| The identifiers of the CUDA device(s) you want to use for training and sampling (in string format, for example: '0,1')		|
| classes	 	| The names of the classes in the image annotations											|
| learning_rate	 	| The learning-rate to train the object detector (default value: 0.005)									|
| confidence_threshold 	| Confidence-threshold for the image analysis with the trained object detector (default value: 0.5)					|
| nms_threshold 	| Non-maximum suppression threshold for the image analysis with the trained object detector (default value: 0.3)			|
| dropout_probability	| Specify the dropout probability between 0.1 and 0.9. Typical used values are: **0.25**, **0.50**, **0.75**				|
| mcd_iterations	| The number of Monte-Carlo iterations to calculate the uncertainty of the image. When this number is increased, the uncertainty metric will be more consistent. When this number is decreased, the sampling will be faster. The value **30** is a good compromise between consistency and speed	|
| iou_thres		| Intersection of Union threshold to cluster the different instance segmentations into observations for the uncertainty calculation, for object detection use **0.5**.																			|
<br/>

Description of the other settings in the uncertainty_sampling.yaml file: [MISC_SETTINGS.md](MISC_SETTINGS.md)
<br/>

Please refer to the folder **active_learning/config** for more setting-files. 
<br/> <br/>

## Other software scripts
Use a trained object detector to auto-annotate unlabelled images: **auto_annotate.py** <br/>

| Argument       	| Description           														|
| ----------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| --img_dir	        | The file directory where the unlabelled images are stored										|
| --network_config	| Configuration of the backbone of the network												|
| --classes	 	| The names of the classes on which the CNN was trained											|
| --conf_thres	 	| Confidence threshold of the CNN to do the image analysis										|
| --nms_thres	 	| Non-maximum suppression threshold of the CNN to do the image analysis									|
| --weights_file 	| Weight-file (.pth) of the trained CNN													|
| --export_format	| Specifiy the export-format of the annotations (currently supported formats: **'labelme'**, **'cvat'**, **'darwin'**, **'supervisely'**)|
| --supervisely_meta_json| When the export_format is **'supervisely'**: specifiy the file location of the **meta.json**						|
<br/>

**Example syntax (auto_annotate.py):**
```python
python auto_annotate.py --img_dir datasets/train --network_config COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml --classes healthy damaged matured cateye headrot --conf_thres 0.5 --nms_thres 0.2 --weights_file weights/broccoli/model_final.pth --export_format labelme
```
<br/>

## Troubleshooting
See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
<br/> <br/>

## License
Our software was forked from Detectron2 (https://github.com/facebookresearch/detectron2). As such, the software will be released under the [Apache 2.0 license](LICENSE). <br/> <br/>

## Acknowledgements
Please have a look at our active learning software for Mask R-CNN (which formed the basis of BoxAL): <br/>
https://github.com/pieterblok/maskal<br/><br/>

## Contact
This code is developed and maintained by Pieter Blok. <br/> <br/>
