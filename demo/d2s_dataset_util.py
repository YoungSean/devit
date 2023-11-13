import sys
import time

sys.path.append("..")
from pycocotools.coco import COCO
from PIL import Image, ImageDraw
import random
# from test_util import *

import os
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

import pickle



# json_path = "./dataset/d2s/d2s_annotations_v1.1/annotations/D2S_training.json"
img_path = "./d2s/images"

clutter_json_path = "./d2s/annotations/D2S_validation_clutter.json"
random_json_path = "./d2s/annotations/D2S_validation_random_background.json"
test_json_path = "./d2s/annotations/D2S_training.json"
json_path = "./d2s/annotations/D2S_training.json"
light_json_path = "./d2s/annotations/D2S_training_light0.json"

# load coco format data
D2S_training = COCO(annotation_file=json_path)
clutter_coco = COCO(annotation_file=clutter_json_path)
random_coco = COCO(annotation_file=random_json_path)
light_coco = COCO(annotation_file=light_json_path)

test_coco = COCO(annotation_file=test_json_path)

# get all class labels
coco_classes = dict([(v['id'], v['name']) for k,v in test_coco.cats.items()])
print(coco_classes)
# display COCO categories and supercategories
cats = D2S_training.loadCats(D2S_training.getCatIds())
nms=[cat['name'] for cat in cats]

def display_image(coco, img_id):
    I = Image.open(os.path.join(img_path, coco.imgs[img_id]['file_name']))
    plt.axis('off')
    plt.imshow(I)
    plt.show()

def cat_to_images(coco_file, cat_ids):
    """Get image ids that have all the specified categories"""
    return coco_file.getImgIds(catIds=cat_ids)

def vis_color_and_mask(image, mask):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Display the first image in the first subplot
    axs[0].imshow(image)
    axs[0].set_title('Image')
    # Display the second image in the second subplot
    axs[1].imshow(mask) # , cmap='gray'
    axs[1].set_title('Its mask')
    plt.show()

def split_image_with_mask(color, mask, ignore=None):
    # plt.imshow(color)
    # plt.show()
    mask_ids = np.sort(np.unique(mask))
    color = np.array(color).astype(int)
    # print("sorted mask ids: ", mask_ids)
    sub_images = []
    sub_masks = []
    for i in mask_ids:
        if i in ignore:
            continue
        temp_mask = (mask==i).astype(int)
        sub_image = color.copy()
        sub_mask = mask.copy()
        # set all pixels not in the mask to 0
        sub_image[temp_mask==0] = 0
        sub_mask[temp_mask==0] = 0
        # set all labels to 1
        sub_mask[sub_mask>0] = 1
        sub_images.append(sub_image)
        sub_masks.append(sub_mask)
        # print("unique values in sub mask: ", np.unique(sub_mask))
        # visualize the image and its mask
    # vis_color_and_mask(sub_images[0], sub_masks[0])
    return sub_images, sub_masks

def split_image_with_mask_all(color, mask, ignore=None):
    # plt.imshow(color)
    # plt.show()
    mask_ids = np.sort(np.unique(mask))
    color = np.array(color).astype(int)
    # print("sorted mask ids: ", mask_ids)
    sub_images = []
    sub_masks = []
    temp_mask = (mask>0).astype(int)
    sub_image = color.copy()
    sub_mask = mask.copy()
    # set all pixels not in the mask to 0
    sub_image[temp_mask==0] = 0
    sub_mask[temp_mask==0] = 0
    # set all labels to 1
    sub_mask[sub_mask>0] = 1
    sub_images.append(sub_image)
    sub_masks.append(sub_mask)
    # visualize the image and its mask
    # vis_color_and_mask(sub_images[0], sub_masks[0])
    return sub_images, sub_masks

def imgID_to_sample(coco, img_id, target_cat_id=None):
    """
    according to img id, get the image file name, image and the corresponding mask
    for template images, only keep the mask with the target category id
    """
    ann_ids = coco.getAnnIds(imgIds=img_id)

    # according to anno idx, get the annos
    targets = coco.loadAnns(ann_ids)

    for t in targets:
        obj_id = t['category_id']
        if obj_id != target_cat_id:
            targets.remove(t)

    # get image file name
    path = coco.loadImgs(img_id)[0]["file_name"]
    img_file_path = os.path.join(img_path, path)
    # filenames.append(os.path.join(img_path, path))
    # frame = np.array(Image.open(os.path.join(img_path, path)), dtype=float)
    # frames.append(frame)
    # print(frame)
    # read image
    img = Image.open(img_file_path).convert('RGB')
    raw_img = img.copy()
    draw = ImageDraw.Draw(img)
    # # draw box to image
    for target in targets:
        x, y, w, h = target['bbox']
        x1, y1, x2, y2 = x, y, int(x + w), int(y + h)
        draw.rectangle((x1, y1, x2, y2))
        draw.text((x1, y1), coco_classes[target["category_id"]])
    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.axis("off")
    # coco.showAnns(targets)

    # fig.add_subplot(1, 2, 2)
    mask = coco.annToMask(targets[0])
    for i in range(1, len(targets)):
        mask += coco.annToMask(targets[i]) * (i + 1)

    # masks.append(mask)
    # print(mask)
    # print(type(mask))
    #
    # plt.imshow(mask)
    # plt.axis("off")
    # plt.show()

    return img_file_path, raw_img, mask

def image_to_sub_images(coco, img_id, cat_id=None):
    # display_image(coco, i)
    img_file_path, color, mask = imgID_to_sample(coco, img_id, cat_id)
    # sub_imgs, sub_masks = split_image_with_mask(color, mask, ignore=[0])
    sub_imgs, sub_masks = split_image_with_mask_all(color, mask)
    return sub_imgs, sub_masks

def image_to_sub_images_each_object(coco, img_id, cat_id=None):
    # display_image(coco, i)
    img_file_path, color, mask = imgID_to_sample(coco, img_id, cat_id)
    sub_imgs, sub_masks = split_image_with_mask(color, mask, ignore=[0])
    # sub_imgs, sub_masks = split_image_with_mask_all(color, mask)
    return sub_imgs, sub_masks

def catID_to_support_set(coco, cat_id, object_bank):
    res_imgs = []
    res_masks = []
    for i in object_bank[cat_id]:
        sub_imgs, sub_masks = image_to_sub_images(coco, i, cat_id)
        res_imgs += sub_imgs
        res_masks += sub_masks
    return res_imgs, res_masks

def catID_to_support_set_each_object(coco, cat_id, object_bank):
    res_imgs = []
    res_masks = []
    for i in object_bank[cat_id]:
        sub_imgs, sub_masks = image_to_sub_images_each_object(coco, i, cat_id)
        res_imgs += sub_imgs
        res_masks += sub_masks
    return res_imgs, res_masks

def catID_to_clutter_query(clutter_coco, cat_ids, scene_img_id):
    clutter_imgIds = cat_to_images(clutter_coco, cat_ids)
    print("number of clutter images: ", len(clutter_imgIds))
    query_img_file_path, query_img, query_mask = imgID_to_sample(clutter_coco, clutter_imgIds[scene_img_id])
    query_img = np.array(query_img).astype(int)
    return query_img_file_path, query_img, query_mask

def sample_sup_images(sup_imgs, sup_masks, sample_num=9):
    num_support_imgs = min(9,len(sup_imgs))
    num_list = list(range(0, len(sup_imgs)))
    # Randomly pick distinct numbers from the list
    random_numbers = random.sample(num_list, num_support_imgs)
    # # random_numbers = [0, 1, 2, 3, 4]
    sup_imgs = [sup_imgs[i] for i in random_numbers]
    sup_masks = [sup_masks[i] for i in random_numbers]
    return sup_imgs, sup_masks

def from_point_to_token_number(point, patch_size=16):
    # point: [x, y]
    # Assuming you have the coordinates of the point (x, y)
    x = point[0]  # X-coordinate of the point
    y = point[1]  # Y-coordinate of the point

    # Calculate the row and column indices of the patch containing the point
    patch_row = y // patch_size
    patch_col = x // patch_size
    return patch_row, patch_col

def transform_point(point, transform, img_np):
    coords = point
    coords = torch.as_tensor(coords, dtype=torch.float32)
    coords = transform.apply_coords_torch(coords, img_np.shape[:2])
    number_coords = coords.numpy().astype(np.int32)
    first_row, first_col = from_point_to_token_number(number_coords)
    target_point = torch.tensor([first_row, first_col])
    return target_point

def get_object_image(original_image, mask_image):
    # Find contours in the mask image
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) > 0:
        # Get the largest contour (assuming it corresponds to the main object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box coordinates
        x, y, width, height = cv2.boundingRect(largest_contour)

        # Extract the object part from the original image
        object_part = original_image[y:y + height, x:x + width]
        object_part = object_part.astype(np.uint8)
        # Resize the object part to the size of the original image
        # resized_object = cv2.resize(object_part, (original_image.shape[1], original_image.shape[0]))

        # Save or display the resized object
        plt.imshow(object_part)
        plt.axis("off")
        plt.show()
        # cv2.imwrite('resized_object.jpg', resized_object)
        # cv2.imshow('Resized Object', resized_object)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("No contours found in the mask image.")

    return object_part
