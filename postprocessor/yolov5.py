import numpy as np
import torch
import torchvision

def yolov5_postprocess(output, confThr=0.7, iou_thres=0.45):
    def xywh2xyxy(x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def calculatIoU(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        area1 = (box1[2]-box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2]-box2[0]) * (box2[3] - box2[1])
        unionArea = area1 + area2 - intersection

        return intersection / unionArea


    def nms(bbox, iou_threshold):
        sorted_idx = np.argsort(bbox[:, 4])
        bbox = bbox[sorted_idx]

        keep = []
        for i in bbox:
            keepBox = True
            for k in keep:
                iou = calculatIoU(i, k)
                if iou > iou_threshold:
                    keepBox = False
                    break
            if keepBox:
                keep.append(i)

        return np.array(keep)

    score_thres = 0.5

    x = output[0]
    xc = x[..., 4] > confThr
    x = x[xc]
    print("number of candidates",len(x))

    #box = xywh2xyxy(x[:, :4])
    #class_conf = x
    #conf, j = x[:, 5:].max(1)
    #class_conf = torch.cat((conf, x[:, 5:]),1)[conf.view(-1) >= score_thres]
    #class_conf = class_conf[:, 1:]
    #x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) >= score_thres]
    #print("before nms",x)
    x[..., :4] = xywh2xyxy(x[..., :4])
    x[:, 5:] *= x[:, 4:5]
    x = nms(x, iou_thres)

    return x