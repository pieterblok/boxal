
# note about this repo:
it is not a forked epo, rather a clone which was pushed to a new remote on iob github. the reason was that the forked repo could be only public while iob does not allow public repo.

---------------------------------
# install on the MAIC workstation
## use a singularity container
1. build the container (only once)
```
cd singularity
2. singularity build --fakeroot boxal.sif boxal.def
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
