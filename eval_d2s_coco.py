import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# Path to the COCO formatted ground truth annotations
gt_annotations_path = 'demo/d2s/annotations/D2S_training.json'
coco_gt = COCO(gt_annotations_path)

coco_dt = coco_gt.loadRes('output/coco_instances_results.json')
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# print(coco_eval.stats)
# get all classes from the groundtruth
# classes = coco_gt.loadCats(coco_gt.getCatIds())
# get the id according to the class name
# print(classes)
# classname2id = {}
# for i in range(len(classes)):
#     classname2id[classes[i]['name']] = classes[i]['id']
# print(classname2id)
# category_space = "demo/d2s_all_templates_each_object_prototypes.pth"
# category_space = torch.load(category_space)
# label_names = category_space['label_names']
# def from_pred_id_to_classid(pred_id):
#     return classname2id[label_names[pred_id]]
# # classname = label_names[2]
# # print(classname)
# # print(classname2id[classname])
# print(from_pred_id_to_classid(2))






