# BoxAL - Active learning for object detection in Detectron2

<p align="center">
  <img src="./demo/boxal_framework.png?raw=true" alt="boxal_framework"/>
</p>

## Summary
BoxAL is an active learning framework that automatically selects the most-informative images for training an object detector (like Fast R-CNN or Faster R-CNN) in Detectron2. By using BoxAL, it is possible to reduce the number of image annotations, without negatively affecting the performance of the object detector. Generally speaking, BoxAL involves the following steps:
1. Train an object detector on a small initial subset of a bigger dataset
2. Use the trained object detector to make predictions on the unlabelled images of the remaining dataset
3. Select the most-informative images with a sampling algorithm
4. Annotate the most-informative images, and then retrain the object detector on the most informative-images
5. Repeat step 2-4 for a specified number of sampling iterations <br/><br/>

<!---The figure below shows the performance improvement of BoxAL on our dataset. By using BoxAL, the performance of Faster R-CNN improved more quickly and therefore [...] annotations could be saved (see the black dashed line):

![maskAL_graph](./demo/maskAL_vs_random.png?raw=true)

## MaskAL instruction video
[![IMAGE ALT TEXT HERE](./demo/video_screenshot.png?raw=true)](https://www.youtube.com/watch?v=OS2EjuNLTcQ) <br/> <br/>--->

## Installation
Linux/Ubuntu: [INSTALL.md](INSTALL.md)
Windows: [INSTALL_Windows.md](INSTALL_Windows.md)
<br/> <br/>

## Data preparation and training
Split the dataset in a training set, validation set and a test set. It is not required to annotate every image in the training set, because BoxAL will select the most-informative images automatically. <br/> 

1. From the training set, a smaller initial dataset is randomly sampled (the dataset size can be specified in the **boxal.yaml** file). The images that do not have an annotation are placed in the **annotate** subfolder inside the image folder. You first need to annotate these images with LabelMe (json), V7-Darwin (json), Supervisely (json) or CVAT (xml) (when using CVAT, export the annotations to **LabelMe 3.0** format).
2. Step 1 is repeated for the validation set and the test set (the file locations can be specified in the **boxal.yaml** file). 
3. After the first training iteration, the sampling algorithm selects the most-informative images (its size can be specified in the **boxal.yaml** file).
4. The most-informative images that don't have an annotation, are placed in the **annotate** subfolder. Annotate these images with LabelMe (json), V7-Darwin (json), Supervisely (json) or CVAT (xml) (when using CVAT, export the annotations to **LabelMe 3.0** format). 
5. OPTIONAL: it is possible to use the trained model to auto-annotate the unlabelled images to further reduce annotation time. Set **auto_annotate** to **True** in the **boxal.yaml** file, and specify the **export_format** (currently supported formats: **'labelme'**, **'cvat'**, **'darwin'**, **'supervisely'**). 
6. Step 3-5 are repeated for several training iterations. The number of iterations (**loops**) can be specified in the **boxal.yaml** file.

Please note that BoxAL does not work with the default COCO json-files of detectron2. These json-files contain all annotations that are completed before the training starts. Because BoxAL involves an iterative train and annotation procedure, the default COCO json-files lack the desired format.
<br/><br/>

## How to use BoxAL
Open a terminal (Ctrl+Alt+T):
```console
(base) user@computer:~$ cd boxal
(base) user@computer:~/boxal$ conda activate boxal
(boxal) user@computer:~/boxal$ python boxal.py --config boxal.yaml
```
<br/>
Change the following settings in the boxal.yaml file: <br/>

| Setting        	| Description           														|
| ----------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| weightsroot	        | The file directory where the weight-files are stored											|
| resultsroot		| The file directory where the result-files are stored 											|
| dataroot	 	| The root directory where all image-files are stored											|
| initial_train_dir     | When use_initial_train_dir is activated: the file directory where the initial training images and annotations are stored		|
| traindir	 	| The file directory where the training images and annotations are stored								|
| valdir	 	| The file directory where the validation images and annotations are stored								|
| testdir	 	| The file directory where the test images and annotations are stored									|
| use_initial_train_dir | Set this to **True** when you want to start the active-learning from an initial training dataset. When **False**, the initial dataset of size **initial_datasize** is randomly sampled from the **traindir** 																	|
| network_config	| The configuration-file (.yaml) file for the object detector (choose either Fast R-CNN or Faster R-CNN, Retinanet is not supported) (see the folder './configs')																			|
| pretrained_weights	| The pretrained weights to start the active-learning. Either specify the **network_config** (.yaml) or a custom weights-file (.pth or .pkl)|
| cuda_visible_devices 	| The identifiers of the CUDA device(s) you want to use for training and sampling (in string format, for example: '0,1')		|
| classes	 	| The names of the classes in the image annotations											|
| learning_rate	 	| The learning-rate to train the object detector (default value: 0.01)									|
| confidence_threshold 	| Confidence-threshold for the image analysis with the trained object detector (default value: 0.5)					|
| nms_threshold 	| Non-maximum suppression threshold for the image analysis with the trained object detector (default value: 0.3)			|
| initial_datasize 	| The size of the initial dataset to start the active learning (when **use_initial_train_dir** is **False**)				|
| pool_size	 	| The number of most-informative images that are selected from the traindir 								|
| loops		 	| The number of sampling iterations													|
| auto_annotate	 	| Set this to **True** when you want to auto-annotate the unlabelled images						 		|
| export_format	 	| When auto_annotate is activated: specify the export-format of the annotations (currently supported formats: **'labelme'**, **'cvat'**, **'darwin'**, **'supervisely'**)																		|
| supervisely_meta_json| When **export_format** is set to **'supervisely'**: specify the file location of the **meta.json** for supervisely export		|
<br/>

Description of the other settings in the boxal.yaml file: [MISC_SETTINGS.md](MISC_SETTINGS.md)
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
BoxAL is developed and maintained by Pieter Blok. <br/> <br/>
