
# note about this repo:
it is not a forked epo, rather a clone which was pushed to a new remote on iob github. the reason was that the forked repo could be only public while iob does not allow public repo.

---------------------------------
# install on the MAIC workstation
## use a singularity container
1. build the container (only once)
```
cd singularity
singularity build --fakeroot boxal.sif boxal.def
```
---------------------------------
# usage
1. run the container
```
singularity run --nv -B /projects/ -B /usr/local/scratch/ singularity/boxal.sif
```
2. convert excel annotation file to labelme json:
```
python3 datasets/convert_box_annot_excel_to_labelme_json.py
```
3. configure input parameters by modifying the boxal.yaml

4. run the training:
```
python3 boxal.py --config boxal.yaml
```

---------------------------------
# the entire protocol 
The steps for the training process of the box detection model are summarized as follows:
1. Copy  the single_box_annotation_final.csv to the workstation:
```
scp -P 8181 /home/pkhateri/Documents/data/progstar/box_annot/single_box_annotation_final.csv pkhateri@localhost:/projects/parisa/data/boxal/faster_rcnn/
```
2. Copy the annotated files to the initial_training or the annotate dir:
```
python copy_png_files_listed_in_csv.py
```
3. Remove the annotated files which have been copied to the test and val dir, from train dir:
```
python snippets/rm_files_in_train_already_in_test_and_val.py
```
4. Shuffle and divide to 0.8/0.2 for train and validation:
```
shuffle_divide_copy_files.py -i <input_dir> -o <output_dir>
```
5. Convert box annotations in the csv file to labelme json file for the existing png files:
```
python datasets/convert_box_annot_csv_to_labelme_json.py
```
6. Train the active learning model with the current annotation:
```
python3 boxal.py --config boxal.yaml
```
7. The  model introduces new images to be annotated, located at the `/projects/parisa/data/boxal/faster_rcnn/train/annotate` directory.

------------------------------------
# the output metrics/instances:
After training, four text files are generated in the `results` and `weights` folders which include metrics or inference results. No `output` folder is generated. Here are the files:
1. `weights/exp1/uncertainty/metrics.json`
This is similar to the original `metrics.json` file which is generated after running detectron2 without the AL implementation.
2. `weights/exp1/uncertainty/inference/coco_instances_results.json`
This file includes some instances. Part of the file below:
```
[{"image_id": 1, "category_id": 1, "bbox": [314.2623596191406, 95.52009582519531, 341.5835876464844, 76.62232971191406], "score": 0.9237105250358582},
{"image_id": 5, "category_id": 1, "bbox": [146.166259765625, 143.88946533203125, 850.9454956054688, 76.61729431152344], "score": 0.9729232788085938},
{"image_id": 5, "category_id": 1, "bbox": [172.25726318359375, 141.29135131835938, 465.15924072265625, 73.83609008789062], "score": 0.23908992111682892},
{"image_id": 5, "category_id": 1, "bbox": [600.7523193359375, 146.4454803466797, 285.7037353515625, 77.92771911621094], "score": 0.07600747048854828},
{"image_id": 7, "category_id": 1, "bbox": [327.11590576171875, 86.79924011230469, 404.923095703125, 96.70126342773438], "score": 0.207632377743721}, ...
```
3. `results/exp1/uncertainty/coco_instances_results.json`
This file is very similar to the previous file but with different instances.
4. `results/exp1/uncertainty/uncertainty.csv`
The content of this file is: 
```
train_size,val_size,test_size,mAP,mAP-damaged
16,31,29,12.8,12.8
25,31,29,18.2,18.2
33,31,29,17.0,17.0
45,31,29,17.0,17.0
56,31,29,23.2,23.2
60,31,29,20.7,20.7
```
