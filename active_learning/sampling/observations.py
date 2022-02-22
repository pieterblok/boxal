# @Author: Pieter Blok
# @Date:   2021-03-25 15:03:44
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2022-01-29 19:46:37

# This function is inspired by the uncertainty_aware_dropout function:
# https://github.com/RovelMan/active-learning-framework/blob/master/al_framework/strategies/dropout.py

import numpy as np
import torch


def observations(outputs, device, iou_thres=0.5):
    """
    To cluster the segmentations for the different Monte-Carlo runs
    """
    observations = {}
    obs_id = 0

    for i in range(len(outputs)):
        sample = outputs[i]
        detections = len(sample['instances'])
        dets = sample['instances'].get_fields()
        
        for det in range(detections):
            if not observations:
                detection = {}
                for key, val in dets.items():
                    detection[key] = val[det]
                observations[obs_id] = [detection]

            else:
                addThis = None
                for group, ds, in observations.items():
                    for d in ds:
                        thisBox = dets['pred_boxes'][det].tensor.squeeze(0)
                        otherBox = d['pred_boxes'].tensor.squeeze(0)

                        xA = torch.max(thisBox[0], otherBox[0])
                        yA = torch.max(thisBox[1], otherBox[1])
                        xB = torch.min(thisBox[2], otherBox[2])
                        yB = torch.min(thisBox[3], otherBox[3])
                        interArea = torch.multiply(torch.max(torch.tensor(0).to(device), xB - xA + 1), torch.max(torch.tensor(0).to(device), yB - yA + 1))
                        boxAArea = torch.multiply((thisBox[2] - thisBox[0] + 1), (thisBox[3] - thisBox[1] + 1))
                        boxBArea = torch.multiply((otherBox[2] - otherBox[0] + 1), (otherBox[3] - otherBox[1] + 1))
                        IOU = torch.divide(interArea, (boxAArea + boxBArea - interArea))

                        if IOU <= iou_thres:
                            break
                        else:
                            detection = {}
                            for key, val in dets.items():
                                detection[key] = val[det]
                            addThis = [group, detection]
                            break
                    if addThis:
                        break
                if addThis:
                    observations[addThis[0]].append(addThis[1])
                else:
                    obs_id += 1
                    detection = {}
                    for key, val in dets.items():
                        detection[key] = val[det]
                    observations[obs_id] = [detection]

    return observations