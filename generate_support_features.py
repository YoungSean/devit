import torch
from collections import defaultdict
import random
# cur_patchs = "weights/initial/few-shot/prototypes/fs_coco14_base_train.vitl14.pkl"
# cur_patchs = "fs_coco17_support_novel_5shot.vitl14.bbox.pkl"
# cur_patchs = "fs_coco_trainval_base.vitl14.bbox.pkl"
# cur_patchs = "fs_coco_trainval_base.vitb14_5000_samples.bbox.pkl"
# coco60 = torch.load(cur_patchs)
# print(len(coco60['patch_tokens']))

fs_coco_2014_seen_classes = ['truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# not sure about the following list of unseen classes
# fs_coco_2014_unseen_classes = ['airplane', 'bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 'couch', 'train', 'tv']
# print(len(coco60['label_names']))
# print(coco60['label_names'] == fs_coco_2014_seen_classes)

#
cur_patchs = "fs_coco_trainval_base.vitb14_5000_samples.bbox.pkl"
coco60 = torch.load(cur_patchs)
labels = coco60['labels']
label_idx = defaultdict(list)
for idx, label in enumerate(labels):
    label_idx[label].append(idx)

RoIs_list = coco60['RoI_feature_tokens']
class_RoIs = {}
for label, idxs in label_idx.items():
    # print(label, len(idxs))
    # Select 8 elements without replacement
    selected_elements = random.sample(idxs, 8)
    selected_RoIs = [RoIs_list[i].flatten(1) for i in selected_elements]
    RoIs = torch.cat(selected_RoIs, dim=1).permute(1, 0)
    class_RoIs[label] = RoIs

# print(class_RoIs)
RoI_prototypes = []
for i in range(len(class_RoIs)):
    # print(class_RoIs[i].shape)
    RoI_prototypes.append(class_RoIs[i])

RoI_prototypes = torch.stack(RoI_prototypes, dim=0)
print(RoI_prototypes.shape)
# print(label_idx)
# name = cur_patchs[:-4] + ".label_idx.pkl"
# print(f'Saving to {name}')
# torch.save(label_idx, name)





