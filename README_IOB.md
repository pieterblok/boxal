# install on the MAIC workstation
## use a singularity container
1. build the container (only once)
```
cd singularity
2. singularity build --fakeroot boxal.sif boxal.def
```
---------------------------------
# use for box annotation
1. run the container
```
singularity run --nv -B /projects/ -B /usr/local/scratch/ singularity/boxal.sif
```
2. convert xml annotation files to labelme json:
```
python3 datasets/convert_box_annot_excel_to_labelme_json.py
```
3. configure input parameters by modifying the boxal.yaml

4. run the training:
```
python3 boxal.py --config boxal.yaml
```

---------------------------------
# note about this repo:
it is not a forked epo, rather a clone which was pushed to a new remote on iob github. the reason was that the forked repo could be only public while iob does not allow public repo.



