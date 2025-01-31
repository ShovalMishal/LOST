# Copyright 2021 - Valeo Comfort and Driving Assistance
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

import json
import os
import pathlib
import argparse
import detectron2.data
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepares the CAD gt for DOTAV2"
                    "dataset in the data format expected from detectron2.")
    parser.add_argument("--dota_dir", type=str, default='/home/shoval/Documents/Repositories/data/multiscale_normalized_dataset_rotated/test',
                        help="Path to where the DOTA dataset is.")
    args = parser.parse_args()

    print('Prepare Class-Agnostic DOTAV2 in the data format expected from detectron2.')

    # Load annotations
    annotation_file = pathlib.Path(args.dota_dir) / "annotations" / "instances.json"
    with open(annotation_file) as json_file:
        annot = json.load(json_file)

    dota_data_gt = detectron2.data.DatasetCatalog.get("dota_ms_normalized_test")
    ann_to_img_ids = [x['id'] for ind, x in enumerate(annot['images'])]
    map_id_to_annot = [x['image_id'] for x in dota_data_gt]

    data = []
    for sample in tqdm(dota_data_gt):
        image_id = sample['image_id']
        image_id_int = int(image_id)
        
        full_img_path = pathlib.Path(sample["file_name"])
        ann_id = ann_to_img_ids.index(image_id_int)
        assert ann_id+1 == image_id_int
        assert full_img_path.is_file()
        annotations = sample["annotations"]
        ca_annotations = [{'iscrowd':v['iscrowd'], 'bbox':v['bbox'], 'category_id': 0, 'bbox_mode':v['bbox_mode']} for v in annotations]

        data.append({
            "file_name": str(full_img_path),
            "image_id": image_id,
            "height": annot['images'][ann_id]['height'],
            "width": annot['images'][ann_id]['width'],
            "annotations": ca_annotations,
        })

    print("Dataset DOTAV2 CAD-gt has been saved.")

    json_data = {"dataset": data, }
    with open(f'./datasets/dota_multiscale_normalized_test_CAD_gt.json', 'w') as outfile:
        json.dump(json_data, outfile)
