# @Author: Pieter Blok
# @Date:   2021-03-25 15:06:20
# @Last Modified by:   Pieter Blok
# @Last Modified time: 2022-07-20 21:19:00

# This function is inspired by the uncertainty_aware_dropout function:
# https://github.com/RovelMan/active-learning-framework/blob/master/al_framework/strategies/dropout.py

import numpy as np
import torch

def uncertainty(observations, iterations, max_entropy, device, mode = 'mean'):
    """
    To calculate the uncertainty metrics on the observations
    """
    uncertainty_list = []
    usem_list = []
    uspl_list = []
    un_list = []
    
    for key, val in observations.items():
        softmaxes = [v['softmaxes'] for v in val]
        
        ## check if there is only one class (then use the softmax-values), otherwise do the entropy calculation 
        if len(softmaxes[0]) == 1:
            inv_entropies_norm = torch.stack([softmax for softmax in softmaxes])
        else:
            entropies = torch.stack([torch.distributions.Categorical(softmax).entropy() for softmax in softmaxes])
            entropies_norm = torch.stack([torch.divide(entropy, max_entropy.to(device)) for entropy in entropies]) ## first normalize the entropy-value with the maximum entropy (which is the least confident situation with equal softmaxes for all classes)
            inv_entropies_norm = torch.stack([torch.subtract(torch.ones(1).to(device), entropy_norm) for entropy_norm in entropies_norm]) ## invert the normalized entropy-values so it can be properly used in the uncertainty calculation

        mean_bbox = torch.mean(torch.stack([v['pred_boxes'].tensor for v in val]), axis=0)
        bbox_IOUs = []
        mean_bbox = mean_bbox.squeeze(0)
        boxAArea = torch.multiply((mean_bbox[2] - mean_bbox[0] + 1), (mean_bbox[3] - mean_bbox[1] + 1))
        for v in val:
            current_bbox = v['pred_boxes'].tensor.squeeze(0)
            xA = torch.max(mean_bbox[0], current_bbox[0])
            yA = torch.max(mean_bbox[1], current_bbox[1])
            xB = torch.min(mean_bbox[2], current_bbox[2])
            yB = torch.min(mean_bbox[3], current_bbox[3])
            interArea = torch.multiply(torch.max(torch.tensor(0).to(device), xB - xA + 1), torch.max(torch.tensor(0).to(device), yB - yA + 1))
            boxBArea = torch.multiply((current_bbox[2] - current_bbox[0] + 1), (current_bbox[3] - current_bbox[1] + 1))
            bbox_IOU = torch.divide(interArea, (boxAArea + boxBArea - interArea))
            bbox_IOUs.append(bbox_IOU.unsqueeze(0))

        if len(bbox_IOUs) > 0:
            bbox_IOUs = torch.cat(bbox_IOUs)
        else:
            bbox_IOUs = torch.tensor([float('NaN')]).to(device)

        val_len = torch.tensor(len(val)).to(device)
        outputs_len = torch.tensor(iterations).to(device)

        u_sem = torch.clamp(torch.mean(inv_entropies_norm), min=0, max=1)
        usem_list.append(u_sem.unsqueeze(0))

        u_spl = torch.clamp(torch.divide(bbox_IOUs.sum(), val_len), min=0, max=1)
        uspl_list.append(u_spl.unsqueeze(0))
        u_sem_spl = torch.multiply(u_sem, u_spl)
        
        try:
            u_n = torch.clamp(torch.divide(val_len, outputs_len), min=0, max=1)
        except:
            u_n = 0.0

        un_list.append(u_n.unsqueeze(0))
        u_h = torch.multiply(u_sem_spl, u_n)
        uncertainty_list.append(u_h.unsqueeze(0))

    if uncertainty_list:
        uncertainty_list = torch.cat(uncertainty_list)

        if mode == 'min':
            uncertainty = torch.min(uncertainty_list)
            usem_list = torch.min(torch.cat(usem_list))
            uspl_list = torch.min(torch.cat(uspl_list))
            un_list = torch.min(torch.cat(un_list))
        elif mode == 'mean':
            uncertainty = torch.mean(uncertainty_list)
            usem_list = torch.mean(torch.cat(usem_list))
            uspl_list = torch.mean(torch.cat(uspl_list))
            un_list = torch.mean(torch.cat(un_list))
        elif mode == 'max':
            uncertainty = torch.max(uncertainty_list)
            usem_list = torch.max(torch.cat(usem_list))
            uspl_list = torch.max(torch.cat(uspl_list))
            un_list = torch.max(torch.cat(un_list))
        else:
            uncertainty = torch.mean(uncertainty_list)
            usem_list = torch.mean(torch.cat(usem_list))
            uspl_list = torch.mean(torch.cat(uspl_list))
            un_list = torch.mean(torch.cat(un_list))
            
    else:
        uncertainty = torch.tensor([float('NaN')]).to(device)
        usem_list = torch.tensor([float('NaN')]).to(device)
        uspl_list = torch.tensor([float('NaN')]).to(device)
        un_list = torch.tensor([float('NaN')]).to(device)

    return uncertainty.detach().cpu().numpy().squeeze(0), usem_list.detach().cpu().numpy().squeeze(0), uspl_list.detach().cpu().numpy().squeeze(0), un_list.detach().cpu().numpy().squeeze(0)