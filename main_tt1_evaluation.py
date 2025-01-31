# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import pdb
import matplotlib
import argparse
import datasets
import json
import torch
import torch.nn as nn
import torchvision
import numpy as np

from tqdm import tqdm

import pickle
from datasets import Dataset, bbox_iou, DOTAv2Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Self-Attention maps")
    parser.add_argument(
        "--type_pred",
        default="detectron",
        choices=["detectron"],
        type=str,
        help="Type of predictions will inform on how to load",
    )
    parser.add_argument(
        "--pred_file", default="", type=str, help="File location of predictions."
    )
    parser.add_argument(
        "--dataset",
        default="VOC07",
        type=str,
        choices=[None, "VOC07", "VOC12", "COCO20k", "DOTA", "DOTA04", "DOTA055", "DOTA08", "DOTA10", "DOTA_SS", "DOTA_MS_NORMALIZED"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--set",
        default="test",
        type=str,
        choices=["val", "train", "trainval", "test"],
        help="Path of the image to load.",
    )
    parser.add_argument(
        "--no_hard",
        action="store_true",
        help="Only used in the case of the VOC_all setup (see the paper).",
    )
    parser.add_argument(
        "--scores_cache", default="./score_cache_dota055.pkl", type=str, help="score cache location"
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------------------------------------
    # Dataset
    dataset = Dataset(args.dataset, args.set, args.no_hard)
    dataset_classes = DOTAv2Dataset.METAINFO["classes"]

    # -------------------------------------------------------------------------------------------------------
    # Load predictions
    if not os.path.exists(args.pred_file):
        raise ValueError(f"File {args.pred_file} does not exists.")

    if args.type_pred == "detectron":
        with open(args.pred_file, "r") as f:
            predictions = json.load(f)

    cnt = 0
    if not os.path.exists(args.scores_cache):
        all_scores = []
        all_labels = []
        pbar = tqdm(dataset.dataloader)
        for im_id, inp in enumerate(pbar):

            # ------------ IMAGE PROCESSING -------------------------------------------
            img = inp[0]
            init_image_size = img.shape

            # Get the name of the image
            im_name = dataset.get_image_name(inp[1])

            # Pass in case of no gt boxes in the image
            if im_name is None:
                continue

            gt_bbxs, gt_cls = dataset.extract_gt(inp[1], im_name)
            if gt_bbxs is not None:
                # Discard images with no gt annotations
                # Happens only in the case of VOC07 and VOC12
                if gt_bbxs.shape[0] == 0 and args.no_hard:
                    continue

            if args.type_pred == "detectron":
                name_ind = im_name
                if "VOC" in args.dataset:
                    name_ind = im_name[:-4]

                pred_ids = [
                    id_i
                    for id_i, pred in enumerate(predictions)
                    if int(pred["image_id"]) == int(name_ind)
                ]

                # No predictions made
                if len(pred_ids) == 0:
                    print("No prediction made")
                    continue

                # Select the most confident prediction
                confidence = [
                    pred["score"]
                    for id_i, pred in enumerate(predictions)
                    if id_i in pred_ids
                ]

                all_scores+=confidence
                boxes = [predictions[ind]["bbox"] for ind in pred_ids]

                curr_labels = []
                # From xywh to x1y1x2y2
                for box in boxes:
                    x1, x2 = box[0], box[0] + box[2]
                    y1, y2 = box[1], box[1] + box[3]
                    pred = np.asarray([x1, y1, x2, y2])
                    ious = datasets.bbox_iou(
                        torch.from_numpy(pred), torch.from_numpy(gt_bbxs.astype(np.float32))
                    )

                    if torch.any(ious >= 0.5):
                        curr_class = gt_cls[torch.argmax(ious).item()]
                    else:
                        curr_class = 0  # zero value stands for background
                    curr_labels.append(curr_class)
                all_labels+=curr_labels
        with open(args.scores_cache, "wb") as f:
            pickle.dump((all_scores, all_labels), f)
    else:
        with open(args.scores_cache, "rb") as f:
            all_scores, all_labels = pickle.load(f)
    all_labels = torch.tensor(all_labels)
    all_scores = torch.tensor(all_scores)
    unique_labels = np.unique(all_labels)
    unique_labels = unique_labels[unique_labels!=0]
    sorted_anomaly_scores, sorted_anomaly_scores_indices = torch.sort(all_scores)
    ranks_dict = {}
    for curr_label in unique_labels:
        # reducing one for the background class
        curr_class = dataset_classes[curr_label-1]
        curr_label_anomaly_scores = all_scores[all_labels == int(curr_label)]
        curr_label_sorted_anomaly_scores, curr_label_sorted_anomaly_scores_indices = torch.sort(
            curr_label_anomaly_scores)
        curr_label_ranks_in_sorted_anomaly_scores = (len(sorted_anomaly_scores) - 1 -
                                                     torch.searchsorted(sorted_anomaly_scores, curr_label_sorted_anomaly_scores))
        curr_label_ranks_in_sorted_anomaly_scores, curr_label_ranks_in_sorted_anomaly_scores_indices = torch.sort(
            curr_label_ranks_in_sorted_anomaly_scores)
        print(f"Class {curr_class} first rank in LOST scores: "
              f"{curr_label_ranks_in_sorted_anomaly_scores[0]}")
        ranks_dict[curr_class] = list(curr_label_ranks_in_sorted_anomaly_scores.numpy().astype(np.float64))

    with open(f"./ranks_dict_{args.dataset}_100.json", 'w') as f:
        json.dump(ranks_dict, f, indent=4)
