import copy
import random
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torchvision.ops import box_area
from detectron2.structures import Boxes, Instances
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from demo.demo import filter_boxes, list_replace, assign_colors
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, DatasetMapper
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer
from matplotlib.pyplot import imshow as cv2_imshow
import matplotlib.pyplot as plt
import sys
import os
import os.path as osp
import detectron2.data.transforms as T
import detectron2.data.detection_utils as utils
from detectron2.config import get_cfg
from tools.train_net import Trainer
from detectron2.data.datasets import register_coco_instances
import torchvision.ops as ops
from torchvision.ops import box_area, box_iou

register_coco_instances("d2s_dataset_train", {}, "demo/d2s/annotations/D2S_training.json", "demo/d2s/images")
register_coco_instances("d2s_dataset_val", {}, "demo/d2s/annotations/D2S_validation.json", "demo/d2s/images")
register_coco_instances("d2s_dataset_val_clutter", {}, "demo/d2s/annotations/D2S_validation_clutter.json", "demo/d2s/images")

dataset_name = "d2s_dataset_train"
d2s_metadata = MetadataCatalog.get(dataset_name)
dataset_dicts = DatasetCatalog.get(dataset_name)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# Path to the COCO formatted ground truth annotations
gt_annotations_path = 'demo/d2s/annotations/D2S_training.json'
coco_gt = COCO(gt_annotations_path)

classes = coco_gt.loadCats(coco_gt.getCatIds())
# get the id according to the class name
classname2id = {}
for i in range(len(classes)):
    classname2id[classes[i]['name']] = classes[i]['id']

category_space = "demo/d2s_all_templates_each_object_prototypes.pth"
category_space = torch.load(category_space)
label_names = category_space['label_names']
def from_pred_id_to_classid(pred_id):
    return classname2id[label_names[pred_id]]


print(d2s_metadata)
# print(dataset_dicts[0])
# print(len(dataset_dicts))
# print(dataset_dicts[0].keys())
# print(dataset_dicts[0]["annotations"])
# print(dataset_dicts[0]["annotations"][0].keys())
# print(dataset_dicts[0]["annotations"][0]["bbox"])
# print(dataset_dicts[0]["annotations"][0]["bbox_mode"])
# print(dataset_dicts[0]["annotations"][0]["segmentation"])
# print(dataset_dicts[0]["annotations"][0]["category_id"])
# print(dataset_dicts[0]["annotations"][0]["iscrowd"])

for d in dataset_dicts[400:800:100]:#random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    print(d["image_id"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=d2s_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    plt.imshow(out.get_image())
    plt.show()

def main(
        config_file="configs/open-vocabulary/lvis/vitl.yaml",
        rpn_config_file="configs/RPN/mask_rcnn_R_50_FPN_1x.yaml",
        model_path="weights/trained/open-vocabulary/lvis/vitl_0069999.pth",

        image_dir='demo/d2s_input',
        output_dir='demo/d2s_output',
        category_space="demo/d2s_all_templates_each_object_prototypes.pth", #"demo/ycb_prototypes.pth",
        device='cuda', #'cpu
        overlapping_mode=True,
        topk=1,
        output_pth=False,
        threshold=0.45
    ):
    assert osp.abspath(image_dir) != osp.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    config = get_cfg()
    config.merge_from_file(config_file)
    config.DE.OFFLINE_RPN_CONFIG = rpn_config_file
    config.DE.TOPK = topk
    config.MODEL.MASK_ON = True
    config.INPUT.MASK_FORMAT = "bitmask"
    # config.MODEL.DEVICE = device
    config.DATASETS.TEST = ("d2s_dataset_val_clutter",)
    # config.DATASETS.TRAIN = ("d2s_dataset_train",)

    config.freeze()
    augs = utils.build_augmentation(config, False)
    augmentations = T.AugmentationList(augs)

    # building models
    model = Trainer.build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    model.eval()
    model = model.to(device)

    if category_space is not None:
        category_space = torch.load(category_space)
        model.label_names = category_space['label_names']
        model.test_class_weight = category_space['prototypes'].to(device)

    label_names = model.label_names
    if 'mini soccer' in label_names:  # for YCB
        label_names = list_replace(label_names, old='mini soccer', new='ball')

    # for img_file in glob(osp.join(image_dir, '*')):
    #     base_filename = osp.splitext(osp.basename(img_file))[0]
    #
    #     dataset_dict = {}
    #     image = utils.read_image(img_file, format="RGB")
    evaluator = COCOEvaluator(dataset_name, output_dir="./output/")
    evaluator.reset()
    print('dataset_dicts', len(dataset_dicts))
    for dataset_dict in tqdm(dataset_dicts[400:800:100]):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        img_file = dataset_dict["file_name"]
        image = utils.read_image(img_file, format="RGB")
        dataset_dict["height"], dataset_dict["width"] = image.shape[0], image.shape[1]

        aug_input = T.AugInput(image)
        augmentations(aug_input)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))).to(device)

        batched_inputs = [dataset_dict]

        output = model(batched_inputs)[0]
        output['label_names'] = model.label_names
        # if output_pth:
        #     torch.save(output, osp.join(output_dir, base_filename + '.pth'))

        # visualize output
        instances = output['instances']
        boxes, pred_classes, scores = filter_boxes(instances, threshold=threshold)

        if overlapping_mode:
            # remove some highly overlapped predictions
            mask = box_area(boxes) >= 400
            boxes = boxes[mask]
            pred_classes = pred_classes[mask]
            scores = scores[mask]
            mask = ops.nms(boxes, scores, 0.3)
            boxes = boxes[mask]
            pred_classes = pred_classes[mask]
            areas = box_area(boxes)
            indexes = list(range(len(pred_classes)))
            for c in torch.unique(pred_classes).tolist():
                box_id_indexes = (pred_classes == c).nonzero().flatten().tolist()
                for i in range(len(box_id_indexes)):
                    for j in range(i + 1, len(box_id_indexes)):
                        bid1 = box_id_indexes[i]
                        bid2 = box_id_indexes[j]
                        arr1 = boxes[bid1].cpu().numpy()
                        arr2 = boxes[bid2].cpu().numpy()
                        a1 = np.prod(arr1[2:] - arr1[:2])
                        a2 = np.prod(arr2[2:] - arr2[:2])
                        top_left = np.maximum(arr1[:2], arr2[:2])  # [[x, y]]
                        bottom_right = np.minimum(arr1[2:], arr2[2:])  # [[x, y]]
                        wh = bottom_right - top_left
                        ia = wh[0].clip(0) * wh[1].clip(0)
                        if ia >= 0.9 * min(a1,
                                           a2):  # same class overlapping case, and larger one is much larger than small
                            if a1 >= a2:
                                if bid2 in indexes:
                                    indexes.remove(bid2)
                            else:
                                if bid1 in indexes:
                                    indexes.remove(bid1)

            boxes = boxes[indexes]
            pred_classes = pred_classes[indexes]
            scores = scores[indexes]
            instances = Instances((dataset_dict["height"], dataset_dict["width"]))
            instances.pred_boxes = Boxes(boxes)
            instances.scores = scores
            # # visualize output
            colors = assign_colors(pred_classes, label_names, seed=4)
            output = to_pil_image(draw_bounding_boxes(torch.as_tensor(image).permute(2, 0, 1), boxes,
                                                      labels=[label_names[cid] for cid in pred_classes.tolist()],
                                                      colors=colors, width=5, font="DejaVuSans-Bold.ttf", font_size=30))
            # # show pil image
            output.show()
            updated_pred_classes = []
            # update pred_classes
            # print('before updating class', pred_classes)
            for cid in pred_classes.tolist():
                label_name = label_names[cid]
                actual_cid = classname2id[label_name] - 1
                updated_pred_classes.append(actual_cid)
            pred_classes = np.array(updated_pred_classes)
            # print('after updating class', pred_classes)

            instances.pred_classes = pred_classes

            # print('after overlapping removal', boxes.shape[0])
            # print('after overlapping removal', pred_classes.shape[0])
            # print('after overlapping removal, scores: ', scores)
            filter_output = {'instances': instances}
            evaluator.process([dataset_dict], [filter_output])

    eval_results = evaluator.evaluate(img_ids=[910, 911, 912])
    print(eval_results)


    # predictor = DefaultPredictor(config)
    #

    # val_loader = build_detection_test_loader(config, mapper=DatasetMapper(config, is_train=False,
    #                                                                       augmentations=[augmentations]))
    # inference_on_dataset(predictor.model, val_loader, evaluator)

main()