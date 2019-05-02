# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import cv2
import datetime
import logging
import time
import glob

import random
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import pandas as pd

from maskrcnn_benchmark.utils.comm import get_world_size, get_rank, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.modanetDrawer import ModaNetDrawer
from tensorboardX import SummaryWriter
from apex import amp

def filter_detections(detections, features) :
    detections_, features_ = [], []
    for detection, feature in zip(detections, features) :

        scores = detection.get_field("scores")
        keep = torch.nonzero(scores > 0.7).squeeze(1)

        detection = detection[keep]
        feature = feature[keep]

        scores = detection.get_field("scores")
        _, idx = scores.sort(0, descending=True)

        detection = detection[idx]
        feature = feature[idx]

        # also filter out redundant classes
        labels = detection.get_field("labels").cpu().numpy().tolist()
        uq_labels = list(set(labels))
        uq_label_idx = [labels.index(l) for l in uq_labels]

        detection = detection[uq_label_idx]
        feature = feature[uq_label_idx]

        assert(len(detection)==feature.shape[0])
        
        detections_.append(detection)
        features_.append(feature)
    return detections_, features_

def split_feats_by_image(detections, features) :
    ndetect_per_image = list(map(len, detections))
    end_ixes = np.cumsum(ndetect_per_image)
    features_ = []
    for ix in range(len(ndetect_per_image)) : 
        if ix == 0 :
            features_.append(features[0:end_ixes[0]])
        else :
            features_.append(features[end_ixes[ix-1]:end_ixes[ix], :])
        assert(features_[ix].shape[0]==ndetect_per_image[ix])
    return features_

def match_detections_with_gt(garment_detections, model_detections, 
                             garment_features, model_features, labels) :
    
    assert(len(garment_detections)==len(model_detections))
    assert(len(garment_detections)==len(labels)) # number of images 
    
    features = torch.Tensor().cuda()
    
    ids = []
    next_id = 0
    # loop over number of images
    for ix in range(len(labels)) :
        label = labels[ix]
        garment_detection, model_detection = garment_detections[ix], model_detections[ix] 
        garment_feature, model_feature = garment_features[ix], model_features[ix]
        
        garment_labels = garment_detection.get_field('labels').cpu().numpy().tolist()
        model_labels = model_detection.get_field('labels').cpu().numpy().tolist()
        
        prediction_match = label in garment_labels and \
                            label in model_labels
        
        if prediction_match : 
            ids += [next_id, next_id]
            next_id += 1
            
            model_idx = model_labels.index(label)
            garment_idx = garment_labels.index(label)
            
            features = torch.cat((features, model_feature[model_idx].unsqueeze(0)))
            features = torch.cat((features, garment_feature[garment_idx].unsqueeze(0)))

        for lbl_ix, lbl in enumerate(model_labels) :
            if prediction_match and lbl == label :
                continue
            features = torch.cat((features, model_feature[lbl_ix].unsqueeze(0)))
            ids.append(next_id)
            next_id += 1

        for lbl_ix, lbl in enumerate(garment_labels) :
            if prediction_match and lbl == label :
                continue
            features = torch.cat((features, garment_feature[lbl_ix].unsqueeze(0)))
            ids.append(next_id)
            next_id += 1
                
    return features, torch.Tensor(ids)

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def get_next(loader_iter, loader) :
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter

def do_train(
    cfg,
    model,
    retrieval_model,
    data_loader,
    dl_val,
    dl_val_coco,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.eval() # do not compute losses 
    start_training_time = time.time()
    end = time.time()

    dl_iter_val = iter(dl_val)
    dl_coco_iter = iter(dl_val_coco)

    writer = SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'logs'))

    for iteration, (garments, models, categoryids) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        garments = garments.to(device)
        models = models.to(device)

        # with torch.no_grad() :
        garment_detections, garment_features = model(garments) # detections is a boxlist, features is a tensor
        model_detections, model_features = model(models)

        garment_features = split_feats_by_image(garment_detections, garment_features) # list of tensors
        model_features = split_feats_by_image(model_detections, model_features)

        garment_detections, garment_features = filter_detections(garment_detections, garment_features)
        model_detections, model_features = filter_detections(model_detections, model_features)

        features, ids = match_detections_with_gt(garment_detections, model_detections, 
                                            garment_features, model_features, categoryids)
        
        # how many garments match? 
        num_pos = int((pd.Series(ids.numpy()).astype(int).value_counts() > 1).sum())
        # how many unique garments 
        num_uq = len(pd.Series(ids.numpy()).astype(int).value_counts())
        meters.update(num_pos=num_pos, num_uq=num_uq)

        try :
            loss, TA_rate = retrieval_model(features, ids)
        except : 
            print('here')
            torch.cuda.empty_cache()
            continue 
        meters.update(loss=loss.item(), TA_rate=TA_rate)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(loss, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # evaluate validation
        # if iteration % cfg.TRAIN.EVAL_VAL_EVERY == 0 :
        #     model.train()
        #     batch, dl_iter_val = get_next(dl_coco_iter, dl_val_coco)
        #     images, targets, _ = batch
        #     images = images.to(device)
        #     targets = [target.to(device) for target in targets]
        #     with torch.no_grad() :
        #         loss_dict = model(images, targets)
        #     loss_dict_reduced = reduce_loss_dict(loss_dict)
        #     losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        #     meters.update(val_loss=losses_reduced, **loss_dict_reduced)
        #     model.eval()

        if iteration % cfg.LOG.PRINT_EVERY == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            
        # if iteration % checkpoint_period == 0:
        #     checkpointer.save("model_{:07d}".format(iteration), **arguments)
        # if iteration == max_iter:
        #     checkpointer.save("model_final", **arguments)

        torch.cuda.empty_cache()

        synchronize()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
