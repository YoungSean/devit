import torch
cur_patchs = "weights/initial/few-shot/prototypes/fs_coco14_base_train.vitl14.pkl"
# cur_patchs = "fs_coco17_support_novel_5shot.vitl14.bbox.pkl"
# cur_patchs = "fs_coco_trainval_base.vitl14.bbox.pkl"
coco60 = torch.load(cur_patchs)
print(len(coco60['patch_tokens']))