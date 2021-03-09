import argparse
import os
import pickle
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import xgboost as xgb

from models.experimental import attempt_load
from flaskblog.utils.datasets import LoadStreams, LoadImages, letterbox
from flaskblog.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from flaskblog.utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
import pandas as pd


yolo_model = None
boxes_classifier = None
bags_classifier = None
device = "cpu"

boxes_sizes = [{'l': '514', 'w': '344', 'h': '300'}, {'l': '550', 'w': '300', 'h': '300'}, {'l': '410', 'w': '310', 'h': '310'}, {'l': '450', 'w': '300', 'h': '225'}, {'l': '357', 'w': '300', 'h': '300'}, {'l': '450', 'w': '300', 'h': '150'}, {'l': '360', 'w': '280', 'h': '310'}, {'l': '610', 'w': '195', 'h': '80'}, {'l': '330', 'w': '250', 'h': '120'}, {'l': '410', 'w': '310', 'h': '310'}, {'l': '357', 'w': '300', 'h': '240'}]

bags_sizes = [{'l': '320', 'w': '110', 'h': '360'}, {'l': '350', 'w': '80', 'h': '275'}, {'l': '405', 'w': '165', 'h': '330'}]

def load_models(device):
    device = "cpu"
    device = select_device(device)
    weights = "models/yolo/best.pt"
    global yolo_model
    yolo_model = attempt_load(weights, map_location=device)  # load FP32 model

    global boxes_classifier
    boxes_classifier = pickle.load(open("models/xgboost/model_CardBoard.pkl", "rb"))

    global bags_classifier
    bags_classifier = pickle.load(open("models/xgboost/model_PaperBag.pkl", "rb"))

def detect(horizontal_img_path, vertical_img_path, device):

    device = "cpu"

    # Load model
    device = select_device(device)
    model = yolo_model

    half = device.type != 'cpu'

    if half:
        model.half()  # to FP16

    imgsz = 640
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[255, 0, 0] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    img_paths = [horizontal_img_path, vertical_img_path]

    imgs0 = [cv2.imread(img_path) for img_path in img_paths]

    imgs = [letterbox(img0, imgsz)[0] for img0 in imgs0]

    # print (imgs[0].shape)
    # exit()

    imgs = [img[:, :, ::-1].transpose(2, 0, 1) for img in imgs]
    imgs = np.asarray([np.ascontiguousarray(img) for img in imgs])

    imgs = torch.from_numpy(imgs).to(device)
    imgs = imgs.half() if half else imgs.float()
    imgs /= 255

    preds = model(imgs)[0]
    preds = non_max_suppression(preds, 0.8, 0.45)

    # Process detections
    results = []
    detected_boxes_bags = []
    selected_dets = []
    for i, (det, img0, img, p) in enumerate(zip(preds, imgs0, imgs, img_paths)):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            # print (img0.shape)
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh

            det[:, :4] = scale_coords(img.shape[1:], det[:, :4], img0.shape).round()

            # Only select one box/bag with the highest conf
            selected_det = []
            selected_box_bag = None
            for each_det in det:
                if int(each_det[-1]) in [0, 1]:
                    if not selected_box_bag or selected_box_bag[1] < each_det[1]:
                        selected_box_bag = each_det
                else:
                    selected_det.append(each_det)

            if selected_box_bag is not None:
                selected_det.append(selected_box_bag)
                detected_box_bag_xywh = (xyxy2xywh(torch.tensor(selected_box_bag[:-2]).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                detected_boxes_bags.append((selected_box_bag[-1], detected_box_bag_xywh))

            selected_dets.append(selected_det)
        else:
            selected_dets.append([])
            selected_box_bag.append([])

        # Stream results
        results.append(img0)

    detected_size = ""
    detected_box_bag_cls = ""
    if detected_boxes_bags[0] and detected_boxes_bags[1]:
        detected_box_bag_cls = detected_boxes_bags[0][0]

        if detected_box_bag_cls == 0:
            classifier = boxes_classifier
            size_texts = boxes_sizes
        else:
            classifier = bags_classifier
            size_texts = bags_sizes

        columns = ['x0','y0', 'w0', "h0", 'x1','y1', 'w1', "h1"]
        features = np.asarray([detected_boxes_bags[0][1] + detected_boxes_bags[1][1]])
        features = xgb.DMatrix(pd.DataFrame(features, columns=columns))

        classifier_preds = classifier.predict(features)
        best_classifiers_preds = np.asarray([np.argmax(line) for line in classifier_preds])
        detected_size = size_texts[best_classifiers_preds[0]]

    # infor = detected_size
    # infor['type'] = int(detected_box_bag_cls)
    # infor['care_mark'] = False
    # infor['broken'] = False
    # print(selected_dets)
    # # Write results
    results = []
    for img0, selected_det in zip(imgs0, selected_dets):
        infor = {}
        infor['type'] = int(detected_box_bag_cls)
        infor['corner'] = []
        infor['care_mark'] = []
        infor['broken'] = []
        infor['bdbox'] = []
        infor['size'] = detected_size
        for *xyxy, conf, cls in selected_det:
            label = '%s %.2f' % (names[int(cls)], conf)
            if int(cls) == 0 or int(cls) == 1:
                infor['bdbox'].append([int(x) for x in xyxy])
            if int(cls) == 2:
                infor['care_mark'].append([int(x) for x in xyxy])
            if int(cls) == 3:
                infor['broken'].append([int(x) for x in xyxy])
            if int(cls) == 4:
                infor['corner'].append([int(x) for x in xyxy])
            # plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
        # print(infor['bdbox'][0])
        # x, y, w, h = infor['bdbox'][0]
        # print(x, y, w, h)
        results.append(infor)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for img0, selected_dets in zip(imgs0, selected_dets):
            x, y, w, h = infor['bdbox'][0]
            cv2.line(img0, (x, y), (x, h), (255,0,0), 2)
            cv2.line(img0, (x, y), (w, y), (255,0,0), 2)
            cv2.line(img0, (w, h), (x, h), (255,0,0), 2)
            cv2.line(img0, (w, h), (w, y), (255,0,0), 2)

            cv2.putText(img0, 'w = 310', (int((w+x)/2), y), font, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.putText(img0, 'h = 310', (w, int((h+y)/2)), font, 1, (255,0,0), 2, cv2.LINE_AA)
       

        cv2.imshow('test.jpg', img0)
        cv2.waitKey()
    
    return results
load_models(device)

img_paths = [
    "/home/tuyen/Desktop/Exas/exas/datatest/Broken02/frame0.jpg",
    "/home/tuyen/Desktop/Exas/exas/datatest/Broken02/frame10.jpg"
]

detection_results = detect(img_paths[0], img_paths[1], device)

# for detection_result, p in zip(detection_results, img_paths):
#     print(detection_result)
#     save_path = str(Path("test") / Path(p).name)
#     print (save_path)
#     cv2.imwrite(save_path, detection_result)

