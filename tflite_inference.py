
"""
Usage:
VIDEO:
    1. Inference Video source from File
        * python tflite_int8_infer.py -t video -s c:/workspace/out.mp4 --weights ''
        * Captured frame image will be stored at c:/workspace/capture_video_images
        * with "--store_video_result", the result videos will be store at c:/workspace/infer_video/{model_name}_infer-video
        * with "--show_cls_conf", the result videos will show all class confidence
    
    2. Inference Video source from Video Folder
        * python tflite_int8_infer.py -t video -s c:/workspace/videos --weights ''

    2. Inference Video source from Device
        * python ftflte int8_infer.py -t video -s 0 --weights ''
        * Captured frame image will be stored at {Current_Directory}/capture_video_images

IMAGE:
    1. Inference Video source from Image Folder
        * python tflite_int8_infer.py -s C:/workspace/dataset/data_STD_1_005_001/images --weights ''
        * with "--store_image_result", the result images will be store at C:/workspace/dataset/data_STD_1_005_001/images/{model_name}_infer-result
        * with "--show_cls_conf", the result images will show all class confidence



Video playe:
s       key     : PLAY/ PAUSE
Space   key     : Single Frame Play
c       key     : Capture Frame Image (Original frame)
"""

from cProfile import label
import os
import sys
import copy
import time
import argparse

sys.path.append('..')
from pathlib import Path
import pickle
import numpy as np
import cv2
import torch
import torchvision

import yaml
import math
import pandas as pd

# import tensorflow as tf

try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    import tensorflow as tf
    Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,


# Common parameters
MEDIA_TYPE                  = -1
MEDIA_SOURCE                = ""
WEIGHT_FILE                 = ""


# IMAGE SOURCE PARAMETERS
ENABLE_DISPLAY              = False
ENABLE_STORE_IMAGE_RESULT   = False
ENABLE_STORE_VIDEO_RESULT   = False
ENABLE_SHOW_CLASS_CONFIDENCE = False
ENABLE_GENERATE_CONF_EXCEL = True
ENABLE_GENERATE_SA_TXT = True
IMAGE_SCALE                 = 4

IS_GET_MISMATCH = False
APPEND_MISMATCH_STRING = False
IS_GET_YOLO_LABEL = False
IS_DRAW_LABEL = False


# VIDEO SORUCE PARAMETERS
DEFAULT_WAIT_TIME = 30  # ms


# cleaner robot project class name
classNameIndex = {
              "Socks"       : 0
            , "PetStool"    : 1
            , "Bottle"      : 2
            , "PowerCable"  : 3
            , "Slippers"    : 4
            , "Scale"       : 5
            , "ChairLeg"    : 6
            , "Cup"         : 7
            , "Fan"         : 8
            , "Shoes"       : 9
            , "Feet"        : 10
        }
classIndexName = {v: k for k, v in classNameIndex.items()}

# hand project class name
# classNameIndex = {
#               "single_close_front"       : 0
#             , "single_close_back"        : 1
#             , "single_open"              : 2
#             , "fist"                     : 3
#             , "both_overlap"             : 4
#         }
# classIndexName = {v: k for k, v in classNameIndex.items()}


#copy from utils.metrics
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

# copy from yolov5.utils.general
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# copy from yolov5.utils.general
def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300,
                        final_confThres = 0.0):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    upd_tabel = np.zeros([11,12], dtype = int)

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates


    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.1 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    output_class_conf = [torch.zeros((0, nc), device=prediction.device)] * bs
    output_max_objness = float(0)

    output_candidate = [torch.zeros((0, 6), device=prediction.device)] * bs
    output_candidate_class_conf = [torch.zeros((0, nc), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)


        # print("---------start-----------")
        # print("class_PRs:", x[:, 5:])
        # print("obj_PRs:", x[:, 4:5])
        # if ENABLE_GENERATE_CONF_EXCEL:
        #     class_PRs = x[:, 5:]
        #     obj_PRs = x[:, 4:5]
        #     for class_PR in class_PRs:
        #         for i in range(len(class_PR)):
        #             class_PR_value = float(class_PR[i])
        #             class_PR_value = (math.floor(class_PR_value*10)/10.0)
        #             if str(class_PR_value).split('.')[0] == '0':
        #                 conf_id = int((str(class_PR_value).split('.')[-1]))
        #                 # print(i, conf_id)
        #                 upd_tabel[i,conf_id] += 1
        #             elif str(class_PR_value).split('.')[0] == '1':
        #                 upd_tabel[i,10] += 1
        #     if class_PRs.shape[0] == 0:
        #         upd_tabel[0,11] += 1

        # If none remain process next image
        if not x.shape[0]:
            continue
        
        max_boxes = []
        max_boxes_class_conf = []

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = class_PRs * obj_PRs
        print("conf:", x[:, 5:])
        print("-----------end------------")

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        class_conf = x

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)

            class_conf = torch.cat((conf, x[:, 5:]),1)[conf.view(-1) > conf_thres] # (obj_conf, cls_conf)
            class_conf = class_conf[:, 1:] # (cls_conf)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres] # (xyxy, conf, cls)
            

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            if ENABLE_GENERATE_CONF_EXCEL:
                upd_tabel[0,11] += 1
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        ## output candidates
        output_candidate[xi] = x[i]
        output_candidate_class_conf[xi] = class_conf[i]

        ## pick the max conf output_candidate to be the output
        if len(output_candidate[0].detach().numpy())>=1:
            candidates = output_candidate[0].detach().numpy()
            candidates_class_conf = output_candidate_class_conf[0].detach().numpy()
            for c in range(len(candidates)):
                if candidates[c][4] > final_confThres:
                    if candidates[c][4] > output_max_objness:
                        max_boxes = candidates[c].tolist()
                        max_boxes_class_conf = candidates_class_conf[c].tolist()
                        output_max_objness = candidates[c][4]
        if len(max_boxes) != 0:
            output = [torch.Tensor([max_boxes])]
            output_class_conf = [torch.Tensor([max_boxes_class_conf])]

        if ENABLE_GENERATE_CONF_EXCEL:
            for output_conf in output_class_conf[0].detach().numpy():
                for j in range(len(output_conf)):
                    output_conf_value = float(output_conf[j])
                    output_conf_value = (math.floor(output_conf_value*10)/10.0)
                    if str(output_conf_value).split('.')[0] == '0':
                        conf_id = int((str(output_conf_value).split('.')[-1]))
                        # print(i, conf_id)
                        upd_tabel[j,conf_id] += 1
                    elif str(output_conf_value).split('.')[0] == '1':
                        upd_tabel[j,10] += 1
            if output_class_conf[0].detach().numpy().shape[0] == 0:
                upd_tabel[0,11] += 1
        print("output:", output)
        print("output_class_conf: ", output_class_conf)
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    
    return output, output_class_conf, upd_tabel

# copy from yolov5.utils.general
def non_max_suppression_objcls(prediction,
                               class_idx_gt=0,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        classes=None,
                        agnostic=False,
                        multi_label=False,
                        labels=(),
                        max_det=300,
                        final_confThres = 0.0):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes

    upd_tabel = np.zeros([nc,12], dtype = int)

    xc = (torch.max(prediction[...,4:5]*prediction[...,5:],dim=2).values) > conf_thres  # candidates


    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.1 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    output_class_conf = [torch.zeros((0, nc), device=prediction.device)] * bs
    output_max_objness = float(0)

    output_candidate = [torch.zeros((0, 6), device=prediction.device)] * bs
    output_candidate_class_conf = [torch.zeros((0, nc), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)


        # print("---------start-----------")
        # print("class_PRs:", x[:, 5:])
        # print("obj_PRs:", x[:, 4:5])
        # if ENABLE_GENERATE_CONF_EXCEL:
        #     class_PRs = x[:, 5:]
        #     obj_PRs = x[:, 4:5]
        #     for class_PR in class_PRs:
        #         for i in range(len(class_PR)):
        #             class_PR_value = float(class_PR[i])
        #             class_PR_value = (math.floor(class_PR_value*10)/10.0)
        #             if str(class_PR_value).split('.')[0] == '0':
        #                 conf_id = int((str(class_PR_value).split('.')[-1]))
        #                 # print(i, conf_id)
        #                 upd_tabel[i,conf_id] += 1
        #             elif str(class_PR_value).split('.')[0] == '1':
        #                 upd_tabel[i,10] += 1
        #     if class_PRs.shape[0] == 0:
        #         upd_tabel[0,11] += 1

        # If none remain process next image
        if not x.shape[0]:
            if ENABLE_GENERATE_CONF_EXCEL:
                upd_tabel[class_idx_gt,11] += 1
            continue
        
        max_boxes = []
        max_boxes_class_conf = []

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = class_PRs * obj_PRs
        # print("conf:", x[:, 5:])
        # print("-----------end------------")

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        class_conf = x

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)

            class_conf = torch.cat((conf, x[:, 5:]),1)[conf.view(-1) > conf_thres] # (obj_conf, cls_conf)
            class_conf = class_conf[:, 1:] # (cls_conf)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres] # (xyxy, conf, cls)
            

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]


        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            if ENABLE_GENERATE_CONF_EXCEL:
                upd_tabel[class_idx_gt,11] += 1
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy
        ## output candidates
        output_candidate[xi] = x[i]
        output_candidate_class_conf[xi] = class_conf[i]

        ## pick the max conf output_candidate to be the output
        if len(output_candidate[0].cpu().numpy())>=1:
            candidates = output_candidate[0].cpu().numpy()
            candidates_class_conf = output_candidate_class_conf[0].cpu().numpy()
            for c in range(len(candidates)):
                if candidates[c][4] > final_confThres:
                    if candidates[c][4] > output_max_objness:
                        max_boxes = candidates[c].tolist()
                        max_boxes_class_conf = candidates_class_conf[c].tolist()
                        output_max_objness = candidates[c][4]
        if len(max_boxes) != 0:
            output = [torch.Tensor([max_boxes])]
            output_class_conf = [torch.Tensor([max_boxes_class_conf])]

        if ENABLE_GENERATE_CONF_EXCEL:
            for output_conf in output[0].cpu().numpy():
                    output_class_id = int(output_conf[5])
                    output_conf_value = float(output_conf[4])
                    output_conf_value = (math.floor(output_conf_value*10)/10.0)
                    if str(output_conf_value).split('.')[0] == '0':
                        conf_id = int((str(output_conf_value).split('.')[-1]))
                        # print(i, conf_id)
                        upd_tabel[output_class_id,conf_id] += 1
                    elif str(output_conf_value).split('.')[0] == '1':
                        upd_tabel[output_class_id,10] += 1
            if output[0].cpu().numpy().shape[0] == 0:
                upd_tabel[class_idx_gt,11] += 1
        # print("upd_tabel", upd_tabel)
        # print("output:", output)
        # print("output_class_conf: ", output_class_conf)
        if (time.time() - t) > time_limit:
            # LOGGER.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            print(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    
    return output, output_class_conf, upd_tabel

def mt_NMS(pred, conf_thres=0.25, iou_thres=0.45, obj_thres=0.5):
    upd_tabel = np.zeros([5,12], dtype = int)

    bs = pred.shape[0]  # batch size
    nc = pred.shape[2] - 5  # number of classes
    xc_indices = pred[..., 4] > conf_thres  # candidates

    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    output = [torch.zeros((0, 10), device=pred.device)] * bs
    output_class_conf = [torch.zeros((0, nc), device=pred.device)] * bs
    for xi, x in enumerate(pred):  # image index, image inference
        x = x[xc_indices[xi]]
        boxes = xywh2xyxy(x[:, :4])

        nms_boxes = []
        merged_index = []
        # print("**********************************")
        conf = x[:, 4].clone() # object probability
        # print("conf:", conf)
        
        conf[conf.view(-1) < obj_thres] = 0
        # print("conf filter with obj_thres:", conf)

        if ENABLE_GENERATE_CONF_EXCEL:
            class_PRs = x[:, 5:]
            obj_PRs = x[:, 4].clone()
            for l, class_PR in enumerate(class_PRs):
                # print(l, class_PR)
                if conf[l] != 0:
                    for i in range(len(class_PR)):
                        class_PR_value = float(class_PR[i])
                        class_PR_value = (math.floor(class_PR_value*10)/10.0)
                        if str(class_PR_value).split('.')[0] == '0':
                            conf_id = int((str(class_PR_value).split('.')[-1]))
                            # print(i, conf_id)
                            upd_tabel[i,conf_id] += 1
                        elif str(class_PR_value).split('.')[0] == '1':
                            upd_tabel[i,10] += 1
            if class_PRs.shape[0] == 0:
                upd_tabel[0,11] += 1
        # If none remain process next image
        # print("conf.shape[0]: ",conf.shape[0])
        if not x.shape[0]:
            continue
        
        if not conf.shape[0]:
            continue
        for i, box in enumerate(boxes):
            if i in merged_index:
                continue
            box = torch.unsqueeze(box, dim=0)
            
            iou = box_iou(box, boxes)
            

            conf[iou.view(-1) < iou_thres] = 0
            # print("conf filter with iou:", conf)

            matched_indices = torch.nonzero(conf).view(-1).tolist()
            merged_index.extend(matched_indices)
            score, max_idx = conf.max(0)
            # print("score:", score)
            print("max_idx:", max_idx)
            test = torch.cat( (boxes[max_idx], x[max_idx, 4:]), 0)
            # print("test",test)
            # test = test.cpu().detach().numpy()
            class_conf  = x[max_idx, 5:]
            nms_boxes.append( test.unsqueeze(0) )
            output_class_conf[xi]= class_conf.unsqueeze(0)
        # print("nms_boxes: ",nms_boxes)
        output = nms_boxes
        # print("output_class_conf: ", output_class_conf)

    return output, output_class_conf, upd_tabel




class TFLiteModel:
    def __init__(self, weight_file: str) -> None:
        self.interpreter = Interpreter(model_path=weight_file)  # load TFLite model
        self.interpreter.allocate_tensors()  # allocate
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs

    def getInputSize(self):
        [n, inputH, inputW, c] = self.input_details[0]["shape"]
        return (inputH, inputW)

    def infer(self, im):
        input, output = self.input_details[0], self.output_details[0]

        scale, zero_point = input['quantization']
        if input['dtype'] == np.int8:
            im = (im / scale + zero_point).astype(np.int8)  # de-scale
        elif input['dtype'] == np.uint8:
            im = (im / scale + zero_point).astype(np.uint8)  # de-scale

        self.interpreter.set_tensor(input['index'], im)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(output['index'])
        if input['dtype'] == np.int8 or input['dtype'] == np.uint8:
            scale, zero_point = output['quantization']
            y = (y.astype(np.float32) - zero_point) * scale  # re-scale
        
        return y



# def get_image_path_dir(dir_path: str):
#     img_paths = list()
#     for dir_path, _, file_names in os.walk(dir_path):
#         # print(f'{dir_path} ')
#         for file_name in file_names:
#             ext_type = file_name.split('.')[-1]
#             if ext_type in extensions:
#                 img_paths.append(os.path.join(dir_path, file_name))
#     return img_paths

    # return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(extensions)]

def get_img_paths(folder_path_list):
    img_extensions = ('.jpg', 'jpeg', 'png', 'bmp')
    img_paths = list()
    if isinstance(folder_path_list, str):
        folder_path_list = [folder_path_list]

    file_path_list = list()
    for folder_path in folder_path_list:
        if os.path.isfile(folder_path):
            file_path_list.append(folder_path)
            continue
        for dir_path, _, file_names in os.walk(folder_path):
            for file_name in file_names:
                file_path = os.path.join(dir_path, file_name)
                if os.path.isfile(file_path):
                    file_path_list.append(file_path)

    for file_path in file_path_list:
        ext_type = file_path.split('.')[-1]
        if ext_type in img_extensions:
            img_paths.append(file_path)
        elif ext_type == 'txt':
            with open(file_path, 'r') as fp:
                lines = fp.readlines()
                for line in lines:
                    line = line.rstrip()
                    img_paths.append(line)
    return img_paths

def get_video_path_dir(dir_path: str):
    extensions = ('.mp4','.MP4')
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith(extensions)]

def infer_video(model:TFLiteModel, media_source:str, media_type, confThr: float=0.4):

    wait_time = 0
    FRAME_COUNT = 5
    SPACE_KEY_ORD = 32
    capture_serial_number = 0
    if media_type == DatasetType.VIDEO_DEVICE:
        video_file_list = [int(media_source)]
        capture_name = f"device_{media_source}"
        store_path = Path(__file__).parent.resolve() / "capture_device_images"
    elif media_type == DatasetType.VIDEO_FILE:
        capture_name = str(Path(media_source).stem)
        video_file_list = [media_source]
        if ENABLE_STORE_VIDEO_RESULT:
            model_name = str(Path(WEIGHT_FILE).stem) + "_infer-video"
            store_path = Path(media_source).parents[0] / "infer_video" / model_name
        else:
            store_path = Path(media_source).parents[0] / "capture_video_images"
    elif media_type == DatasetType.VIDEO_FOLDER:
        video_file_list = get_video_path_dir(media_source)
        model_name = str(Path(WEIGHT_FILE).stem) + "_infer-video"
        store_path = Path(media_source) / "infer_video" / model_name
    total_videos = len(video_file_list)
    if total_videos == 0:
        raise Exception(f"can't get video file: {media_source}")
    print(f"Total Videos: {total_videos}")
    

    

    for video_file in video_file_list:
        if ENABLE_STORE_VIDEO_RESULT:
            store_path.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            if media_type == DatasetType.VIDEO_FOLDER:
                capture_name = str(Path(video_file).stem)

            video_name = f"{capture_name}_infer.mp4"
            if ENABLE_SHOW_CLASS_CONFIDENCE:
                video_name = f"{capture_name}_infer_all_cls.mp4"
            print("video_name: ", video_name)
            video_path = store_path / video_name
            vout = cv2.VideoWriter(str(video_path), fourcc, 20.0, (320, 240), True)

        if media_type == DatasetType.VIDEO_DEVICE:
            vcap = cv2.VideoCapture( video_file )
        else:
            test_path = Path(video_file)
            vcap = cv2.VideoCapture( str(test_path) )
        if vcap is None:
            print(f"vcap is None")
        if vcap.isOpened() == False:
            print(f"vcap is not open")
        frame_time = (1.0 / vcap.get(cv2.CAP_PROP_FPS)) * 1000 # millisecond

        while True:
            ret, frame = vcap.read()
            frame = cv2.resize(frame, (320, 240))
            if ret:
                cur_pos_time = vcap.get(cv2.CAP_PROP_POS_MSEC)
                print(f"cur_pos_msec: {cur_pos_time}")
                frame_RGB = frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
                frame_bak = copy.deepcopy(frame_RGB)
                frameH, frameW = frame.shape
                
                (inputH, inputW) = model.getInputSize()
                input_image = cv2.resize(frame, (inputW, inputH))
                im = input_image.astype(np.float32) / 255
                im = im[None, None, ...]
                n, c, h, w = im.shape  # batch, channel, height, width
                im = im.transpose( (0,2,3,1) )
                
                y = model.infer(im)

                if isinstance(y, np.ndarray):
                    y = torch.tensor(y, device=device)
                bndboxes, class_confs, upd_tabel = mt_NMS(pred=y, conf_thres=confThr)
                print("bndboxes:",bndboxes)
                # bndboxes, class_confs = non_max_suppression(y, conf_thres=confThr)
                bndboxes = bndboxes[0].detach().numpy()
                class_confs = class_confs[0].detach().numpy()

                # ==== all class confidence
                all_conf = []
                for i in range(class_confs.shape[0]):
                    one_conf = []
                    for idx in range(len(class_confs[i])):
                        cls_name = classIndexName[idx]
                        cls_conf = class_confs[i][idx]
                        conf_str = str(cls_name) + ":"+ str(cls_conf) + " "
                        one_conf.append(conf_str)
                    all_conf.append(one_conf)



                # ===== convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # print("bndboxes len:",len(bndboxes))
                # print(bndboxes)
                for i in range(bndboxes.shape[0]):
                    cls_idx = int(bndboxes[i][-1])
                    x0 = int(bndboxes[i][0] * frameW)
                    y0 = int(bndboxes[i][1] * frameH)
                    x1 = int(bndboxes[i][2] * frameW)
                    y1 = int(bndboxes[i][3] * frameH)
                    x0 = 0 if x0 < 0 else x0
                    y0 = 0 if y0 < 0 else y0
                    x1 = frameW if x1 > frameW else x1
                    y1 = frameW if y1 > frameW else y1
                    
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.rectangle(frame, (x0, y0), (x0+10, y0+10), (0,0,0), -1, cv2.LINE_AA)  # filled
                    cv2.putText(frame, str(cls_idx), (x0, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    if ENABLE_SHOW_CLASS_CONFIDENCE:
                        for j in range(len(all_conf[i])):
                            cv2.putText(frame, str(all_conf[i][j]), (int((x0+x1)/2), y0+(j+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                if ENABLE_STORE_VIDEO_RESULT:

                    vout.write(frame)
                if ENABLE_DISPLAY:
                    cv2.imshow('frame', frame)
                    key = cv2.waitKey(wait_time)
                    if key == 27:
                        break
                    elif key == SPACE_KEY_ORD:  # Single frame play
                        wait_time = 0
                    elif key == ord('s'):       # toogle Play / Pause
                        wait_time = DEFAULT_WAIT_TIME if wait_time == 0 else 0
                    elif key == ord('a'):       # backward
                        new_pos_time = cur_pos_time - (frame_time * FRAME_COUNT) 
                        vcap.set(cv2.CAP_PROP_POS_MSEC, new_pos_time)
                        wait_time = 0
                    elif key == ord('d'):       # fast forward
                        new_pos_time = cur_pos_time + (frame_time * FRAME_COUNT) 
                        vcap.set(cv2.CAP_PROP_POS_MSEC, new_pos_time)
                        wait_time = 0
                    elif key == ord('c'):       # Capture frame image
                        store_path.mkdir(parents=True, exist_ok=True)
                        file_name = f"{capture_name}_{int(cur_pos_time)}_{capture_serial_number:04d}.jpeg"
                        print("file_name: ",file_name)
                        file_path = store_path / file_name
                        print("file_path: ",file_path)
                        cv2.imwrite(str(file_path), frame_bak)
                        capture_serial_number += 1

                # # ===== auto capture images
                # if len(bndboxes) == 0:
                #     store_path.mkdir(parents=True, exist_ok=True)
                #     print(store_path)
                #     file_name = f"{capture_name}_{int(cur_pos_time)}_{capture_serial_number:04d}.jpeg"
                #     print("file_name: ",file_name)
                #     file_path = store_path / file_name
                #     print("file_path: ",file_path)
                #     # cv2.imwrite(str(file_path), frame_bak)
                #     capture_serial_number += 1
            else:
                break

        vcap.release()
        vout.release()


def get_yolo_label(image_file:str):
    """
    return list of [c, xc, yc, w, h]
    """
    label_file = image_file.rsplit('.', 1)[0]
    label_file += '.txt'
    labels = []
    if os.path.exists(label_file):
        with open(label_file, 'r') as fp:
            for line in fp.readlines():
                line = line.rstrip()
                e = line.split(' ')
                label = [int(e[0]), float(e[1]), float(e[2]), float(e[3]), float(e[4])]
                labels.append(label)
                # print(f"labels: {labels[-1]}")
    else:
        print(f"{label_file} does not exist")
    return labels

def xcycwh2xyxy(labels, imw, imh):
    """
    labels: list of [c, xc, yc, w, h]
    return: list of [c, x0, y0, x1, y1]
    """
    new_labels = []
    for l in labels:
        xc = int(l[1] * imw)
        yc = int(l[2] * imh)
        bw = int(l[3] * imw)
        bh = int(l[4] * imh)
        x0 = xc - int(bw/2)
        y0 = yc - int(bh/2)
        x1 = xc + int(bw/2)
        y1 = yc + int(bh/2)

        x0 = 0 if x0 < 0 else x0
        y0 = 0 if y0 < 0 else y0
        x1 = imw if x1 > imw else x1
        y1 = imh if y1 > imh else y1
        new_labels.append( [l[0], x0, y0, x1, y1] )
    return new_labels

    

def draw_label(target_image, labels, imw, imh, draw_class_name=False):
    labels = xcycwh2xyxy(labels, imw, imh)
    for label in labels:
        # elems = label.split(' ')
        cls_idx = label[0]
        cls_name = classIndexName[cls_idx]
        x0 = label[1]
        y0 = label[2]
        x1 = label[3]
        y1 = label[4]

        cv2.rectangle(target_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        # cv2.rectangle(target_image, (x0, y0), (x0+10, y0+10), (0,0,0), -1, cv2.LINE_AA)  # filled
        if draw_class_name:
            cv2.putText(target_image, cls_name, (x0, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(target_image, str(cls_idx), (x0, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

def box_iou_1(b1, b2):
    """
    b1, b2: [x0, y0, x1, y1]
    """
    def box_area(b):
        if b[2] < b[0] or b[3] < b[1]:
            return 0
        return (b[2] - b[0]) * (b[3] - b[1])
    area1 = box_area(b1)
    area2 = box_area(b2)
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    inter = box_area([x0, y0, x1, y1])
    return inter / (area1 + area2 - inter)

def mismatch_iou(labels, predictions, iou_thres=0.4):
    iou_mismatch_count = 0
    cls_mismatch_count = 0
    false_pred = False
    for l in labels:
        iou_match = False
        cls_match = False
        # print(f"label: {l}")
        for p in predictions:
            # print(f"pred: {p}")
            iou = box_iou_1(l[1:], p[1:])
            # print(f"iou: {iou}")
            if iou >= iou_thres:
                iou_match = True
                if l[0] == p[0]:
                    cls_match = True
                    predictions.remove(p)
                    break
        if iou_match == False:
            # print(f"iou not match: {l}")
            iou_mismatch_count += 1 
        else:
            if cls_match == False:
                # print(f"cls not match: {l}")
                cls_mismatch_count += 1
    if len(predictions) > 0:
        # print(f"False Pred: {len(predictions)}")
        # print(f"{predictions}")
        false_pred = True
    
    result = ''
    if iou_mismatch_count > 0:
        result += "iou_not_match_"
    if cls_mismatch_count > 0:
            result += "cls_not_match_"
    if false_pred:
        result += "false_pred_"
    
    return result[:-1]


def infer_images(tflite_model_path, 
                 data, 
                 conf_thres: float=0.4, 
                 iou_thres:float=0.4, 
                #  obj_thr: float=0.5,
                 image_scale=4,
                 num_classes=4,
                 device = 'cpu',
                 save_dir='results/',
                 generate_conf_excel=True,
                 generate_sa_txt=True,
                 **kwargs):
    with open(data, 'r') as f:
        data_path = yaml.safe_load(f)['test']
    print(f'data_path: {data_path}')
    img_paths = get_img_paths(data_path)
    num_images = len(img_paths)

    if num_images == 0:
        raise Exception(f"can't get image file: {img_paths}")
    print(f"Total Images: {num_images}")

    assert os.path.isfile(tflite_model_path), f'tflite model file: {tflite_model_path} is not exist.'
    model = TFLiteModel(tflite_model_path)    

    if device == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device('cuda')

    if kwargs.get('conf_thres_test') != None:
        conf_thres = kwargs['conf_thres_test']

    if generate_conf_excel:
        table = None

        obj_0 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]           # Socks
        obj_1 = [14,15,16,17,18,19,20,21,22]                # PetStool
        obj_2 = [37,38,39,40,41,42,43,44,45,46,47,48]       # Bottle
        obj_3 = [23,24,25,26,27,28,29,30,31,32,33,34,35,36] # PowerCable
        obj_4 = []                                          # Slippers
        obj_5 = []                                          # Scale
        obj_6 = []                                          # ChairLeg
        obj_7 = []                                          # Cup
        obj_8 = []                                          # Fan
        obj_9 = []                                          # Shoes
        obj_10 = []                                         # Feet
        obj_all_case = [obj_0, obj_1, obj_2, obj_3, obj_4, obj_5, obj_6, obj_7, obj_8, obj_9, obj_10]
    
    if generate_sa_txt:
        sa_txt = []

    
    cur_idx = 0
    for cur_idx in range(num_images):
        if((cur_idx+1) % 1000 == 0):
            print(f'Finish Inference Count: {cur_idx+1}')
        img_path = img_paths[cur_idx]

        dirname = os.path.dirname(img_path)
        # distance = 1
        ### TMP FOR TEST
        if 'Socks' in dirname:
            obj_case = 0
        elif 'PetStool' in dirname:
            obj_case = 14
        elif 'Bottle' in dirname:
            obj_case = 37
        elif 'PowerCable' in dirname:
            obj_case = 23
        else:
            img_name = Path(img_path).name
            d = img_name.split('_')
            floor_type = int(d[0])
            light_type = int(d[1])
            obj_case = int(d[2])
            # distance = int(d[3])
            # raise Exception(f"{img_path} is not in PetStool or PowerCable or Socks")

        ### TMP FOR TEST

        class_idx_gt = None
        for i in range(len(obj_all_case)):
            if obj_case in obj_all_case[i]:
                class_idx_gt = i

            # if distance == 1: # 10 cm
            #     upd_tab_id = class_idx_gt
            # elif distance == 2: # 20 cm
            #     upd_tab_id = class_idx_gt + 11
            # elif distance == 3: # 30 cm
            #     upd_tab_id = class_idx_gt + 22
            

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception(f"{img_path} is not an image file")

        img_h, img_w = img.shape
        enlarge_image = cv2.resize(img, (img_w*image_scale, img_h*image_scale))

        (inputH, inputW) = model.getInputSize()
        # print(f"input hw: [{inputH}, {inputW}]")
        img = cv2.resize(img, (inputW, inputH))
        img = img.astype(np.float32) / 255
        img = img[None, None, ...]
        # n, c, h, w = img.shape  # batch, channel, height, width
        img = img.transpose( (0,2,3,1) )
        y = model.infer(img)

        if isinstance(y, np.ndarray):
            y = torch.tensor(y, device=device)

        # bndboxes, class_confs, upd_tabel = mt_NMS(pred=y, conf_thres=confThr, iou_thres=iou_thres, obj_thres= objThr)
        bndboxes, class_confs, upd_tabel = non_max_suppression_objcls(y, class_idx_gt, conf_thres=conf_thres, iou_thres=iou_thres)
        # print("bndboxes:", bndboxes)
        bndboxes = bndboxes[0].cpu().numpy()
        class_confs = class_confs[0].cpu().numpy()
        # print(upd_tabel)
        if generate_conf_excel:
            if not isinstance(table, np.ndarray) and table == None:            
                num_classes = y.shape[2] - 5  # number of classes
                table = np.zeros([num_classes,12,num_classes], dtype = int)
            table[:,:,class_idx_gt] += upd_tabel

        # ==== all class confidence
        all_conf = []
        for i in range(class_confs.shape[0]):
            one_conf = []
            for idx in range(len(class_confs[i])):
                cls_name = classIndexName[idx]
                cls_conf = class_confs[i][idx]
                conf_str = str(cls_name) + ":"+ str(cls_conf) + " "
                one_conf.append(conf_str)
            all_conf.append(one_conf)


        # ===== convert image to RGB
        image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        enlarge_image = cv2.cvtColor(enlarge_image, cv2.COLOR_GRAY2BGR)

        # ===== Ground Truth Label
        
        # labels = []
        # labels = get_yolo_label(image_file)
        
        # if is_draw_label and is_get_yolo_label and len(labels) > 0:
        #     draw_label(image, labels, imageW, imageH)
        #     draw_label(enlarge_image, labels, imageW*image_scale, imageH*image_scale, draw_class_name=False)

        # ===== Predition
        predictions = []
        if generate_sa_txt:
            if bndboxes.shape[0] == 0:
                inference_detail = f"{int(d[0])}_{int(d[1])}_{int(d[2])}_{int(d[3])}_{int(d[4])}_{d[5]}_msg_255_0_0_0_0_0"
                # inference_detail = f"msg_255_0_0_0_0_0"
                sa_txt.append(inference_detail)


        for i in range(bndboxes.shape[0]):
            cls_idx = int(bndboxes[i][-1])
            cls_name = classIndexName[cls_idx]
            x0 = int(bndboxes[i][0] * img_w)
            y0 = int(bndboxes[i][1] * img_h)
            x1 = int(bndboxes[i][2] * img_w)
            y1 = int(bndboxes[i][3] * img_h)
            x0 = 0 if x0 < 0 else x0
            y0 = 0 if y0 < 0 else y0
            x1 = img_w if x1 > img_w else x1
            y1 = img_h if y1 > img_h else y1

            predictions.append( [cls_idx, x0, y0, x1, y1] )

            cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.rectangle(image, (x0, y0), (x0+10, y0+10), (0,0,0), -1, cv2.LINE_AA)  # filled
            cv2.putText(image, str(cls_idx), (x0, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if generate_sa_txt:
                score = float(bndboxes[i][4])*100
                inference_detail = f"{int(d[0])}_{int(d[1])}_{int(d[2])}_{int(d[3])}_{int(d[4])}_{d[5]}_msg_{cls_idx}_{x0}_{y0}_{x1}_{y1}_{score}"
                # inference_detail = f"msg_{cls_idx}_{x0}_{y0}_{x1}_{y1}_{score}"
                sa_txt.append(inference_detail)

            x0 = int(bndboxes[i][0] * img_w * image_scale)
            y0 = int(bndboxes[i][1] * img_h * image_scale)
            x1 = int(bndboxes[i][2] * img_w * image_scale)
            y1 = int(bndboxes[i][3] * img_h * image_scale)
            x0 = 0 if x0 < 0 else x0
            y0 = 0 if y0 < 0 else y0
            x1 = img_w*image_scale if x1 > img_w*image_scale else x1
            y1 = img_h*image_scale if y1 > img_h*image_scale else y1
            cv2.rectangle(enlarge_image, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.rectangle(enlarge_image, (x0, y0), (x0+10, y0+10), (0,0,0), -1, cv2.LINE_AA)  # filled
            cv2.putText(enlarge_image, str(cls_idx), (x0, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # if show_class_confidence:
            #     for j in range(len(all_conf[i])):
            #         cv2.putText(enlarge_image, str(all_conf[i][j]), (int((x0+x1)/2), y0+(j+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


        
        # if is_get_mismatch and store_path != "":
        #     # new_labels = xcycwh2xyxy(labels, imageW, imageH)
        #     result = mismatch_iou(new_labels, predictions, iou_thres=0.4)
        #     # print(f"result: {result}")
        #     if result != "":
        #         if "iou_not_match" in result:
        #             folder_name = "iou_not_match"
        #         elif "cls_not_match" in result:
        #             folder_name = "cls_not_match"
        #         elif "false_pred" in result:
        #             folder_name = "false_pred"
        #         else:
        #             folder_name = "mismatch"

        #         if isinstance(result_folder_path, Path) == False:
        #             result_folder_path = Path(result_folder_path)
        #         mismatch_path = result_folder_path / folder_name
        #         mismatch_path.mkdir(parents=True, exist_ok=True)
        #         file_name = Path(image_file).stem
        #         if append_mismatch_string:
        #             file_path = str(mismatch_path.absolute() / file_name) + f"_{result}.jpg"
        #         else:
        #             file_path = str(mismatch_path.absolute() / Path(image_file).name )
        #         print(f"mismatch: {file_path}")
        #         cv2.imwrite(file_path, enlarge_image)

        # if store_image_result:
        #     if isinstance(store_path, Path) == False:
        #         store_path = Path(store_path)
        #     file_name = Path(image_file).stem

        #     if show_class_confidence:
        #         file_path = str(store_path / file_name) + "_l.jpg"
        #         cv2.imwrite(file_path, enlarge_image)
        #     else:
        #         file_path = str(store_path / file_name) + ".jpg"
        #         cv2.imwrite(file_path, image)

        # if display:
        #     cv2.imshow('image', image)
        #     cv2.imshow('enlarge image', enlarge_image)
            
        #     key = cv2.waitKey(0)
        #     if key == 27:
        #         break
        #     elif key == ord('d') or key == 32: # next image
        #         cur_idx = total_images if cur_idx >= total_images else (cur_idx + 1)
        #     elif key == ord('a'): # next image
        #         cur_idx = 0 if (cur_idx - 1) < 0 else (cur_idx - 1)
        # else:
    # generate dataframe
    if generate_conf_excel:
        excel_path = os.path.join(save_dir, f'robot_Photo_inference_obj_conf{conf_thres}_{str(Path(tflite_model_path).stem)}.xlsx')
        f_excel = pd.ExcelWriter(excel_path)

        # [num_classes+1(all-accuracy), 11(conf_interval)+2(class_name+val_name)]
        num_measurement = 4
        summary_table = np.zeros([num_classes*num_measurement+1,11]) 
        confusion_matrix = np.zeros([num_classes, num_classes])
        confusion_matrix_sheets = list()

        num_data_per_class = [np.sum(table[..., i]) for i in range(num_classes)]

        class_detect_cnt = [0] * num_classes
        correct_cnt = [0] * num_classes
        wrong_cnt = [0] * num_classes
        for conf_idx in range(10, -1, -1):
            for class_idx in range(num_classes):
                confusion_matrix[class_idx] += table[:, conf_idx, class_idx]
                class_detect_cnt += table[:, conf_idx, class_idx]
                correct_cnt[class_idx] += table[class_idx, conf_idx, class_idx]
                wrong_cnt[class_idx] += np.sum(table[:, conf_idx, class_idx]) - table[class_idx, conf_idx, class_idx]

                if num_data_per_class[class_idx]:
                    recall = correct_cnt[class_idx] / num_data_per_class[class_idx]
                    object_view = (correct_cnt[class_idx] + wrong_cnt[class_idx]) / num_data_per_class[class_idx]
                else:
                    recall = 0
                    object_view = 0

                # summary_table[class_idx*3, 0] = classIndexName[class_idx]
                # summary_table[class_idx*3, 1] = 'Recall'
                summary_table[class_idx*num_measurement, conf_idx] =  recall

                # summary_table[class_idx*3+2, 0] = classIndexName[class_idx]
                # summary_table[class_idx*3+2, 1] = 'Object View'
                summary_table[class_idx*num_measurement+2, conf_idx] = object_view

            for class_idx in range(num_classes):
                if class_detect_cnt[class_idx]:
                    precision = correct_cnt[class_idx] / class_detect_cnt[class_idx]
                else:
                    precision = 0

                recall = summary_table[class_idx*num_measurement, conf_idx]
                if precision+recall:
                    f1_score = float(2) * precision * recall / (precision + recall)
                else:
                    f1_score = 0
                
                # summary_table[class_idx*3+1, 0] = classIndexName[class_idx]
                # summary_table[class_idx*3+1, 1] = 'Precision'
                summary_table[class_idx*num_measurement+1, conf_idx] =  precision
                summary_table[class_idx*num_measurement+3, conf_idx] =  f1_score

            # summary_table[-1, 0] = 'All'
            # summary_table[-1, 1] = 'Accuracy'
            summary_table[-1, conf_idx] =  np.sum(correct_cnt) / np.sum(num_data_per_class)

            confusion_matrix_sheet = pd.DataFrame(confusion_matrix.copy())
            multi_index = [[classIndexName[idx] for idx in range(num_classes)]]
            multi_index = pd.MultiIndex.from_product(multi_index, names=['Predicted'])
            confusion_matrix_sheet.columns = multi_index
            multi_index = [[], []]
            multi_index[0].append(np.round(conf_idx*0.1, 1))
            for class_idx in range(num_classes):
                multi_index[1].append(classIndexName[class_idx])
            multi_index = pd.MultiIndex.from_product(multi_index, names=['Confidence', 'Acutal'])
            confusion_matrix_sheet.index = multi_index
            confusion_matrix_sheets.append(confusion_matrix_sheet)



        summary_table = np.round(summary_table, 3)
        summary_sheet = pd.DataFrame(summary_table)
        multi_index_columns = [[str(np.round(i*0.1, 1)) for i in range(11)]]
        summary_sheet.columns = pd.MultiIndex.from_product(multi_index_columns, names=['Confidence'])
        multi_index = [[], []]
        for class_idx in range(num_classes):
            multi_index[0].append(classIndexName[class_idx])
        multi_index[1].append('Recall')
        multi_index[1].append('Precision')
        multi_index[1].append('Object View')    
        multi_index[1].append('F1 Score')    
        multi_index = pd.MultiIndex.from_product(multi_index, names=["Class Name", "Measurement"])
        multi_index = multi_index.append(pd.MultiIndex.from_arrays([['All'], ['Accuracy']]))
        summary_sheet.index = multi_index
        summary_sheet.to_excel(f_excel, sheet_name = 'Summary')

        #auto-adjust column width
        for column in summary_sheet:
            column_length = max(summary_sheet[column].astype(str).map(len).max(), len(column))
            col_idx = summary_sheet.columns.get_loc(column)
            f_excel.sheets['Summary'].set_column(col_idx, col_idx, column_length+10)

        confusion_matrix_sheets = pd.concat([confusion_matrix_sheet for confusion_matrix_sheet in confusion_matrix_sheets], axis=0)
        confusion_matrix_sheets.to_excel(f_excel, sheet_name = 'Confusion Matrix')
        #auto-adjust column width
        for column in confusion_matrix_sheets:
            column_length = max(confusion_matrix_sheets[column].astype(str).map(len).max(), len(column))
            col_idx = confusion_matrix_sheets.columns.get_loc(column)
            f_excel.sheets['Confusion Matrix'].set_column(col_idx, col_idx, column_length+10)

        for class_idx in range(num_classes):
            table_pd = pd.DataFrame(table[:,:,class_idx])
            table_pd.index = [classIndexName[idx] for idx in range(num_classes)]
            multi_index_columns = [[str(np.round(i*0.1, 1)) for i in range(11)]]
            multi_index_columns[0].append('None')
            table_pd.columns = pd.MultiIndex.from_product(multi_index_columns, names=['Confidence'])
            table_pd.to_excel(f_excel, sheet_name = classIndexName[class_idx], na_rep='NaN')
        
            #auto-adjust column width
            for column in table_pd:
                column_length = max(table_pd[column].astype(str).map(len).max(), len(column))
                col_idx = table_pd.columns.get_loc(column)
                f_excel.sheets[classIndexName[class_idx]].set_column(col_idx, col_idx, column_length+10)

        # f_excel.save()
        f_excel.close()        
        print(f'saved EXCEL at {excel_path}')

        # table_pd_0_1 = 
        # table_pd_1_1 = pd.DataFrame(table[:,:,1])
        # table_pd_2_1 = pd.DataFrame(table[:,:,2])
        # table_pd_3_1 = pd.DataFrame(table[:,:,3])

        # table_pd_0_1.index = [classIndexName[0], classIndexName[1], classIndexName[2], classIndexName[3], classIndexName[4], classIndexName[5], classIndexName[6], classIndexName[7], classIndexName[8], classIndexName[9], classIndexName[10]]
        # table_pd_1_1.index = table_pd_0_1.index
        # table_pd_2_1.index = table_pd_0_1.index
        # table_pd_3_1.index = table_pd_0_1.index

        # table_pd_0_1.columns = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0', 'None']
        # table_pd_1_1.columns = table_pd_0_1.columns
        # table_pd_2_1.columns = table_pd_0_1.columns
        # table_pd_3_1.columns = table_pd_0_1.columns
        
        # table_pd_0 = pd.concat([table_pd_0_1, table_pd_0_2, table_pd_0_3], axis=0)
        # table_pd_1 = pd.concat([table_pd_1_1, table_pd_1_2, table_pd_1_3], axis=0)
        # table_pd_2 = pd.concat([table_pd_2_1, table_pd_2_2, table_pd_2_3], axis=0)
        # table_pd_3 = pd.concat([table_pd_3_1, table_pd_3_2, table_pd_3_3], axis=0)
        # table_pd_4 = pd.concat([table_pd_4_1, table_pd_4_2, table_pd_4_3], axis=0)
        # table_pd_5 = pd.concat([table_pd_5_1, table_pd_5_2, table_pd_5_3], axis=0)
        # table_pd_6 = pd.concat([table_pd_6_1, table_pd_6_2, table_pd_6_3], axis=0)
        # table_pd_7 = pd.concat([table_pd_7_1, table_pd_7_2, table_pd_7_3], axis=0)
        # table_pd_8 = pd.concat([table_pd_8_1, table_pd_8_2, table_pd_8_3], axis=0)
        # table_pd_9 = pd.concat([table_pd_9_1, table_pd_9_2, table_pd_9_3], axis=0)
        # table_pd_10 = pd.concat([table_pd_10_1, table_pd_10_2, table_pd_10_3], axis=0)

        # with pd.ExcelWriter(f'{save_dir}/robot_Photo_inference_obj_conf{conf_thres}.xlsx') as writer:
            # table_pd_0.to_excel(writer, sheet_name = 'single_close_front')
            # table_pd_1.to_excel(writer, sheet_name = 'single_close_back')
            # table_pd_2.to_excel(writer, sheet_name = 'single_open')
            # table_pd_3.to_excel(writer, sheet_name = 'fist_front')
            # table_pd_4.to_excel(writer, sheet_name = 'both_overlap')
            # table_pd_5.to_excel(writer, sheet_name = 'fist_back')
    if generate_sa_txt:
        sa_txt_save_name = os.path.join(save_dir, f'ROT_infer_{str(Path(tflite_model_path).stem)}.txt')
        with open(sa_txt_save_name, 'w') as txtf:
            for txt_line in sa_txt:
                print(txt_line, file = txtf)
        print(f'saved SA(TXT) at {sa_txt_save_name}')

def evaluate_procedure(opt):
    infer_images(**vars(opt))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device',   default=0, help='0 for cpu, 1 for gpu')
    parser.add_argument('-t', '--type',     type=str, default='image', help='image or video')
    parser.add_argument('-s', '--source',   type=str, help='source for image file or folder path of image or video file path or video device id')
    parser.add_argument('-w', '--weight',   type=str, help='path of tflite weight file')
    parser.add_argument('--store_image_result',     action='store_true', default=False , help="Enable Store image inference result")
    parser.add_argument('--store_video_result',     action='store_true', help="Enable Store video inference result")
    parser.add_argument('--show_cls_conf',          action='store_true', help="Enalbe Show all class confidence")
    parser.add_argument('--display',                action='store_true', help="Enable Display")
    parser.add_argument('--store_path', type=str, help='path for storing result')

    parser.add_argument('--get_mismatch',   action='store_true', help="get prediction and ground truth mismatch, only for image")
    parser.add_argument('--append_mismatch_str',   action='store_true', help="append mismatch string when get mismatch image")
    # parser.add_argument('--draw_label',   action='store_true', help="draw label on image for get mismatch image")
    parser.add_argument('--gen_conf_excel', action='store_true', default=True,help="generate object confidence statistics excel")
    parser.add_argument('--conf_thres', type=float, default = 0.4, help="confidence threshold")
    parser.add_argument('--gen_sa_txt', action='store_true', default=True, help="generate SA images inference imformation txt")



    args = parser.parse_args()

    if args.device == 1:
        device = torch.cuda.device(0)
    else:
        device = torch.device("cpu")

    if args.get_mismatch:
        IS_GET_MISMATCH = True
    if args.append_mismatch_str:
        APPEND_MISMATCH_STRING = True
    if IS_GET_MISMATCH:
        IS_GET_YOLO_LABEL = True
        IS_DRAW_LABEL = True

    
    
    MEDIA_TYPE = ""
    MEDIA_SOURCE = args.source
    if args.display:
        ENABLE_DISPLAY = True
    if args.store_image_result:
        store_image_result = True
    if args.store_video_result:
        ENABLE_STORE_VIDEO_RESULT = True
    store_path = ""
    if args.store_path:
        store_path = args.store_path
    
    if args.gen_conf_excel:
        ENABLE_GENERATE_CONF_EXCEL = True

    if args.gen_sa_txt:
        ENABLE_GENERATE_SA_TXT = True

    if args.type == "video":
        if (MEDIA_SOURCE).isdecimal():
            print(f"video device: {MEDIA_SOURCE}")
            MEDIA_TYPE = "VIDEO_DEVICE"
        else:
            if os.path.isfile(MEDIA_SOURCE):
                print(f"video file: {MEDIA_SOURCE}")
                MEDIA_TYPE = "VIDEO_FILE"
            elif os.path.isdir(MEDIA_SOURCE):
                MEDIA_TYPE = "VIDEO_FOLDER"
    elif args.type == "image":
        if os.path.isdir(MEDIA_SOURCE):
            MEDIA_TYPE = "IMAGE_FOLDER"
        elif os.path.isfile(MEDIA_SOURCE):
            if MEDIA_SOURCE.endswith("jpeg") or MEDIA_SOURCE.endswith("jpg") or MEDIA_SOURCE.endswith("png") or MEDIA_SOURCE.endswith("bmp"):
                MEDIA_TYPE = "IMAGE_FILE"
            elif MEDIA_SOURCE.endswith("txt"):
                MEDIA_TYPE = "IMAGE_FILE_LIST"
    

    if MEDIA_TYPE == "":
        raise Exception(f"can not locate your source: {args.source}")

    print(f"DEVICE          : {device}")
    print(f"MEDIA_TYPE      : {MEDIA_TYPE}")
    print(f"MEDIA_SOURCE    : {MEDIA_SOURCE}")
    print(f"WEIGHT_FILE     : {WEIGHT_FILE}")
    if (MEDIA_TYPE == "IMAGE_FOLDER"
        or MEDIA_TYPE == "IMAGE_FILE"
        or MEDIA_TYPE == "IMAGE_FILE_LIST"):
        print( "ENABLE DISPLAY   : {}".format("ON" if ENABLE_DISPLAY else "OFF") )
        print( "IMAGE INFER STORE: {}".format("ON" if store_image_result else "OFF"))
        print( "VIDEO INFER STORE: {}".format("ON" if ENABLE_STORE_VIDEO_RESULT else "OFF"))
        print( "GENERATE CONFIDENCE EXCEL: {}".format("ON" if ENABLE_GENERATE_CONF_EXCEL else "OFF"))
        print( "GENERATE SA TXT: {}".format("ON" if ENABLE_GENERATE_SA_TXT else "OFF"))
        print( "==================================")
        print( "GET MISMATCH              : {}".format("ON" if IS_GET_MISMATCH else "OFF"))
        print( "APPEND MISMATCH STRING    : {}".format("ON" if APPEND_MISMATCH_STRING else "OFF"))
        print( "GET YOLO LABEL            : {}".format("ON" if IS_GET_YOLO_LABEL else "OFF"))
        print( "GET DRAW LABEL            : {}".format("ON" if IS_DRAW_LABEL else "OFF"))


    if args.weight is not None and os.path.isfile(args.weight):
        WEIGHT_FILE = args.weight
    

    # IS_GET_YOLO_LABEL = True
    # IS_DRAW_LABEL = True
    
    model = TFLiteModel(WEIGHT_FILE)

    if (MEDIA_TYPE == "VIDEO_DEVICE" 
        or MEDIA_TYPE == "VIDEO_FILE"
        or MEDIA_TYPE == "VIDEO_FOLDER"):
        infer_video(model, MEDIA_SOURCE, MEDIA_TYPE)
    elif (MEDIA_TYPE == "IMAGE_FOLDER"
        or MEDIA_TYPE == "IMAGE_FILE"
        or MEDIA_TYPE == "IMAGE_FILE_LIST"):

        infer_images(model, MEDIA_SOURCE, MEDIA_TYPE, confThr=args.conf_thres, iou_thres=0.4, store_path=store_path, objThr=0.5)
    else:
        print(f"Undefine MEDIA_SOURCE")
