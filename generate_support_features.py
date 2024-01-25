import torch
from collections import defaultdict
import random
# cur_patchs = "weights/initial/few-shot/prototypes/fs_coco14_base_train.vitl14.pkl"
# cur_patchs = "fs_coco17_support_novel_5shot.vitl14.bbox.pkl"
# cur_patchs = "fs_coco_trainval_base.vitl14.bbox.pkl"
# cur_patchs = "fs_coco_trainval_novel_5shot.vitb14_100_samples.bbox.pkl"
cur_patchs = 'weights/initial/few-shot/prototypes/fs_coco_trainval_novel_5shot.vitb14.pkl'
coco60 = torch.load(cur_patchs)
# print(len(coco60['patch_tokens']))

fs_coco_2014_seen_classes = ['truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
fs_coco_2014_unseen_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 'couch', 'potted plant', 'dining table', 'tv']
# not sure about the following list of unseen classes
# fs_coco_2014_unseen_classes = ['airplane', 'bicycle', 'bird', 'boat', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorcycle', 'person', 'potted plant', 'sheep', 'couch', 'train', 'tv']
# print(len(coco60['label_names']))
# print(coco60['label_names'])
# print(coco60['label_names'] == fs_coco_2014_unseen_classes)

# all_combined_classes = fs_coco_2014_seen_classes + fs_coco_2014_unseen_classes
# check_all_class = ['truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'bottle', 'chair', 'couch', 'potted plant', 'dining table', 'tv']
# print("checking all classes", all_combined_classes == check_all_class)
# coco60['labels'] = [label + len(fs_coco_2014_seen_classes) for label in coco60['labels']]
# print(max(coco60['labels']))

# name = "fs_coco_trainval_novel_5shot.vitb14_100_samples_label60_79.bbox.pkl"
# print(f'Saving to {name}')
# torch.save(coco60, name)

#
# cur_patchs = "fs_coco_trainval_base.vitb14_5000_samples.bbox.pkl"
# coco60 = torch.load(cur_patchs)
#
# labels = coco60['labels']
# label_idx = defaultdict(list)
# for idx, label in enumerate(labels):
#     label_idx[label].append(idx)
#
# RoIs_list = coco60['RoI_feature_tokens']
# class_RoIs = {}
# for label, idxs in label_idx.items():
#     print(label, len(idxs))
#     # Select 30 elements with replacement
#     if len(idxs) < 30:
#         selected_elements = [ i for i in idxs]
#         print(selected_elements)
#     else:
#         selected_elements = random.sample(idxs, 30)
#     # print(selected_elements)
#     selected_RoIs = [RoIs_list[i] for i in selected_elements]
#     RoIs = torch.stack(selected_RoIs, dim=0)
#     if RoIs.shape[0] < 30:
#         RoIs = torch.cat([RoIs, torch.zeros(30 - RoIs.shape[0], *RoIs.shape[1:]).to(RoIs)], dim=0)
#     class_RoIs[label] = RoIs
#
# print(class_RoIs[0].shape)
# # Sort the dictionary by keys and extract tensors
# sorted_tensors = [class_RoIs[k] for k in sorted(class_RoIs)]
# #
# # # Concatenate the tensors
# combined_tensor = torch.stack(sorted_tensors, dim=0)
# print(combined_tensor.shape)  # class_num, sample_num, channel, height, width
# RoI_prototypes = []
# for i in range(len(class_RoIs)):
#     # print(class_RoIs[i].shape)
#     RoI_prototypes.append(class_RoIs[i])
#
# RoI_prototypes = torch.stack(RoI_prototypes, dim=0)
# print(RoI_prototypes.shape)
# print(label_idx)
# name = cur_patchs[:-4] + ".classNum.pkl"
# print(f'Saving to {name}')
# torch.save(combined_tensor, name)





