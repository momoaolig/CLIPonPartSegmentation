import argparse
import json
import os
from spin import SPIN, InitialRequestError
import cv2
import numpy as np
from tqdm import tqdm
import pycocotools.mask as maskUtils

# from refer import REFER

parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--data_root', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--dataset',
                    type=str,
                    # choices=['refcoco', 'refcoco+', 'refcocog', 'refclef'],
                    default='refcoco')
parser.add_argument('--split', type=str, default='umd')
parser.add_argument('--generate_mask', action='store_true')
args = parser.parse_args()
img_path = os.path.join(args.data_root, 'images', 'train2014')

h, w = (416, 416)

# refer = REFER(args.data_root, args.dataset, args.split)

print('dataset [%s_%s] contains: ' % (args.dataset, args.split))
# ref_ids = refer.getRefIds()
# image_ids = refer.getImgIds()
# print('%s expressions for %s refs in %s images.' %
#       (len(refer.Sents), len(ref_ids), len(image_ids)))

# print('\nAmong them:')
# if args.dataset == 'refclef':
#     if args.split == 'unc':
#         splits = ['train', 'val', 'testA', 'testB', 'testC']
#     else:
#         splits = ['train', 'val', 'test']
# elif args.dataset == 'refcoco':
#     splits = ['train', 'val', 'testA', 'testB']
# elif args.dataset == 'refcoco+':
splits = ['train', 'val', 'testA', 'testB']
# elif args.dataset == 'refcocog':
#     splits = ['train', 'val',
#               'test']  # we don't have test split for refcocog right now.

# for split in splits:
#     ref_ids = refer.getRefIds(split=split)
#     print('%s refs are in split [%s].' % (len(ref_ids), split))


def cat_process(cat):
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return cat


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))

def get_annotations(coco, img_id):

    dict = {}
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    masks = []
    bboxes = []
    labels = []
    for ann in anns:
        mask = maskUtils.decode(ann['segmentation'])
        bbox = ann['bbox']
        label = ann['category_id']
        masks.append(mask)
        bboxes.append(bbox)
        labels.append(label)
    # print(f"Total masks: {len(masks)}")
    dict['masks'], dict['bboxes'], dict['label_ids'] = masks, bboxes, labels
    return dict

def prepare_dataset(dataset, splits, output_dir, generate_mask=False):
    ann_path = os.path.join(output_dir, 'anns', dataset)
    mask_path = os.path.join(output_dir, 'masks', dataset)
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    for split in splits:
        dataset_array = []
        ref_ids = refer.getRefIds(split=split)
        print('Processing split:{} - Len: {}'.format(split, len(ref_ids)))
        for i in tqdm(ref_ids):
            ref_dict = {}

            refs = refer.Refs[i]
            bboxs = refer.getRefBox(i)
            sentences = refs['sentences']
            image_urls = refer.loadImgs(image_ids=refs['image_id'])[0]
            cat = cat_process(refs['category_id'])
            image_urls = image_urls['file_name']
            if dataset == 'refclef' and image_urls in [
                    '19579.jpg', '17975.jpg', '19575.jpg'
            ]:
                continue
            box_info = bbox_process(bboxs)

            ref_dict['bbox'] = box_info
            ref_dict['cat'] = cat
            ref_dict['segment_id'] = i
            ref_dict['img_name'] = image_urls

            if generate_mask:
                cv2.imwrite(os.path.join(mask_path,
                                         str(i) + '.png'),
                            refer.getMask(refs)['mask'] * 255)

            sent_dict = []
            for i, sent in enumerate(sentences):
                sent_dict.append({
                    'idx': i,
                    'sent_id': sent['sent_id'],
                    'sent': sent['sent'].strip()
                })

            ref_dict['sentences'] = sent_dict
            ref_dict['sentences_num'] = len(sent_dict)

            dataset_array.append(ref_dict)
        print('Dumping json file...')
        with open(os.path.join(output_dir, 'anns', dataset, split + '.json'),
                  'w') as f:
            json.dump(dataset_array, f)


def prepare_part_dataset(dataset, splits, output_dir, generate_mask=True):
    mask_path = os.path.join(output_dir, 'masks')
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    # for split in splits:
    dataset_array = []

    annotation_dir = "../../PartImageNet/jsons"
    image_dir = "../../PartImageNet/images"

    split = "train"
    spin_api = SPIN(
        annotation_dir=annotation_dir, image_dir=image_dir, split=split, download=True
    )
    try:
        spin_api.download_spin(save_directory=annotation_dir)
    except InitialRequestError as e:
        print(f"An error occurred: {e}")

    img_ids = spin_api.getImgIds()
    print(f"Total number of images in split {spin_api.split}: {len(img_ids)}")
    print('Processing split:{} - Len: {}'.format(split, len(img_ids)))
    for id in tqdm(img_ids):
        
        wholes, parts = get_annotations(spin_api.wholes, id), get_annotations(spin_api.parts, id)

        part_categories = spin_api.get_categories(granularity="part")
        parts['labels'] = [part_categories[label_id]['name'].lower() for label_id in parts['label_ids']]
        parts['labels'] = [label.replace("tier", "tire") for label in parts['labels']]
        whole_categories = spin_api.get_categories(granularity="whole")
        wholes['labels'] = [whole_categories[label_id]['name'] for label_id in wholes['label_ids']]
        
        # print(class_hier)
        whole = spin_api.get_object_name_for_file(id)

        masks = parts['masks'] 
        bboxes = parts['bboxes'] 
        labels = parts['labels'] 
        sentences = labels
        image_urls = spin_api.subparts.loadImgs(id)[0]
        # cat = cat_process(refs['category_id'])
        image_urls = image_urls['file_name']
        # if dataset == 'refclef' and image_urls in [
        #         '19579.jpg', '17975.jpg', '19575.jpg'
        # ]:
        #     continue
        for i, (mask, box, label) in enumerate(zip(masks, bboxes, labels)): 
            ref_dict = {}
            box_info = bbox_process(box)
            segment_id = str(id) + '_' + str(i)

            ref_dict['bbox'] = box_info
            ref_dict['cat'] = whole
            ref_dict['segment_id'] = segment_id
            ref_dict['img_name'] = image_urls

            # if generate_mask:
            #     print(1)
            cv2.imwrite(os.path.join(mask_path, segment_id + '.png'), mask * 255)

            sent_dict = []
            # for i, sent in enumerate(sentences):
                # sent_dict.append({
                #     'idx': i,
                #     'sent_id': sent['sent_id'],
                #     'sent': sent['sent'].strip()
                # }
            sent_dict.append({
                'idx': id,
                'sent_id': 0,
                'sent': label
            })

            ref_dict['sentences'] = sent_dict
            ref_dict['sentences_num'] = len(sent_dict)
            dataset_array.append(ref_dict)
        if len(dataset_array) >= 5000:
            break
        # print(dataset_array)
    print('total images:', id)
    with open(os.path.join(output_dir, 'anns', dataset + '.json'),
                'w') as f:
        json.dump(dataset_array, f)


prepare_part_dataset(args.dataset, splits, args.output_dir, args.generate_mask)
