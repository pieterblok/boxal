import json, os, glob
from detectron2.structures import BoxMode

__all__ = ["OCTData",]

class OCTData:
    '''
    oct data class:
        read oct data annotation from json file
        write annotations to a dataset_dicts proper for detectron2
    '''
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_oct_dicts(self):
        dataset_dicts = []
        for idx, f in enumerate(glob.glob(self.data_dir+"/*.png")):
            json_file = os.path.join(self.data_dir, os.path.splitext(os.path.basename(f))[0]+".json")
            #print(json_file)
            img_annot = json.load(open(json_file))
            record={}
            record["file_name"] = f
            record["image_id"] = idx
            record["height"] = img_annot['imageHeight']
            record["width"] = img_annot['imageWidth']

            objs = []
            for box in img_annot["shapes"]:
                #print('\tbounding box:', box['points'])
                p1 = box['points'][0]
                p2 = box['points'][1]
                obj = {
                    "bbox": [int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts
