import os
import numpy as np
import json
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from lavis.models import load_model_and_preprocess
from transformers import CLIPProcessor, CLIPModel
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from spin import SPIN
import pycocotools.mask as maskUtils
import nltk
from nltk.corpus import wordnet as wn
import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Access environment variables
hf_api_key = os.getenv("HF_API_KEY")

# Ensure WordNet data is available
nltk.download('wordnet')
print(torch.cuda.is_available())

def get_args_parser():
    parser = argparse.ArgumentParser('Caption Generation script', add_help=False)
    parser.add_argument("--output_dir", type=str, default='train-data', help="path to the json file with image path and image caption")
    parser.add_argument("--split", type=str, default='train', help="split of dataset")
    parser.add_argument("--caption_refine", type=lambda x:x.lower() == "true", default=False, help="if to use llama 2 to refine generated captions")
    parser.add_argument("--hier_level", type=str, default=None, help="The hierarchy level of parts in the caption")
    parser.add_argument("--gran", type=str, default="full", help="The granularity of annotations for training")
    parser.add_argument("--coco", type=lambda x:x.lower()== "true", default=False, help="if using coco dataset")

    return parser

def encode_tensor_as_string(arr):
    if type(arr) != np.ndarray:
        arr = arr.data.cpu().numpy()
    return base64.b64encode(arr.tobytes()).decode('utf-8')

# load BLIP and CLIP model
def load_clip():
    # CLIP model: get processer for text
    text_version = "openai/clip-vit-large-patch14"
    clip_text_model = CLIPModel.from_pretrained(text_version).cuda().eval()
    clip_text_processor = CLIPProcessor.from_pretrained(text_version)
    print("CLIP loaded  successfully!")
    return clip_text_model, clip_text_processor


# preprocess text using CLIP
def preprocess_text(processor, input):
    if input == None:
        return None
    inputs = processor(text=input,  return_tensors="pt", padding=True)
    inputs['input_ids'] = inputs['input_ids'].cuda()
    inputs['pixel_values'] = torch.ones(1,3,224,224).cuda() # placeholder 
    inputs['attention_mask'] = inputs['attention_mask'].cuda()
    return inputs

# get CLIP embeddings
def get_clip_feature_text(model, processor, input):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using {device}")
    model.to(device)
    
    which_layer_text = 'before'
    inputs = preprocess_text(processor, input)
    if inputs == None:
        return None
    for key in inputs:
         if torch.is_tensor(inputs[key]):
             inputs[key] = inputs[key].to(device)
    
    outputs = model(**inputs)
    if which_layer_text == 'before':
        feature = outputs.text_model_output.pooler_output
        return feature

def apply_mask(image, mask):
    """Apply the mask to the image and return the masked image."""
    image_np = np.array(image)
    mask_np = np.array(mask)

    mask_binary = mask.astype(bool)
    area = np.sum(mask_binary)
    
    masked_image_np = image_np * mask_np[:, :, np.newaxis]
    masked_image = Image.fromarray(masked_image_np)
    return masked_image, area
    
def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

# decode base64 to pillow image
def decode_base64_to_pillow(image_b64):
    return Image.open(BytesIO(base64.b64decode(image_b64))).convert('RGB')

# read images
def read_image_from_json(filename):
    # read RGB image with Image
    with open(filename, 'r') as f:
        data = json.load(f)
    img = decode_base64_to_pillow(data['image'])
    return img, data

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)

# convert binary mask to RLE
def mask_2_rle(binary_mask):
    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def save_mask_data(output_dir, mask_list, box_list, label_list, file_name, output, clip_text_model, clip_text_processor):
    value = 0  # 0 for background

    for label, box, mask in zip(label_list, box_list, mask_list):
        value += 1
        box_xywh = [int(x) for x in box]
        x1y1x2y2 = [int(x) for x in box]
        x1y1x2y2[2] += x1y1x2y2[0]
        x1y1x2y2[3] += x1y1x2y2[1]
        rle=mask_2_rle(mask)
        anno = get_base_anno_dict(is_stuff=0, is_thing=1, bbox=box_xywh, mask_value=value, rle=rle, category_name=label, area=box_xywh[-1]*box_xywh[-2])

        instance_caption = label
        text_embedding_before = encode_tensor_as_string(get_clip_feature_text(clip_text_model, clip_text_processor, label))
        anno['text_embedding_before'] = text_embedding_before

        anno['caption'] = instance_caption
        output['annos'].append(anno)

    with open(os.path.join(output_dir, 'label_{}.json'.format(file_name)), 'w') as f:
        json.dump(output, f)
        # print("Saved {}/label_{}.json".format(output_dir, file_name))

# convert PIL image to base64
def encode_pillow_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def get_base_output_dict(image, dataset_name, image_path, data=None):
    output = {}
    if data != None:
        if 'similarity' in data:
            output['similarity'] = data['similarity']
        if 'AESTHETIC_SCORE' in data:
            output['AESTHETIC_SCORE'] = data['AESTHETIC_SCORE']
        if 'caption' in data:
            output['caption'] = data['caption']
        if 'width' in data:
            output['width'] = data['width']
        if 'height' in data:
            output['height'] = data['height']
        if 'file_name' in data:
            output['file_name'] = data['file_name']
        if 'is_det' in data:
            output['is_det'] = data['is_det']
        else:
            output['is_det'] = 0
        if 'image' in data:
            output['image'] = data['image']
    else:
        output['file_name'] = image_path # image_paths[i]
        output['is_det'] = 1
        output['image'] = encode_pillow_to_base64(image.convert('RGB'))
    output['dataset_name'] = dataset_name
    output['data_id'] = 1
    # annos for all instances
    output['annos'] = []
    return output

def get_base_anno_dict(is_stuff, is_thing, bbox, mask_value, rle, category_name, area):
    anno = {
        "bbox": bbox,
        "mask_value": mask_value,
        "mask": rle,
        "category_name": category_name,
        "text_embedding_before": "",
        "caption": "",
        "is_stuff": is_stuff,
        "is_thing": is_thing,
        "area": area
    }
    return anno

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

def get_subpart_by_part(parts, subparts):
    subparts = [[' '.join(label.split('-')[1:-1]), label.split('-')[-1]] for label in subparts['labels']]
    parts = [' '.join(label.split(' ')[1:]) for label in parts['labels']]
    class_hierarchy = {part: [] for part in parts}
    for subpart in subparts:
        for part in parts:
            if subpart[0] == part:
                class_hierarchy[part].append(subpart[-1])
    return class_hierarchy


# Function to refine caption
instruction = """
Refine this caption for clarity and style, directly give your answer:
"""
def refine_caption(caption, instruction=instruction):
    prompt = f"{instruction} {caption}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate a refined caption
    output = model.generate(inputs.input_ids, max_new_tokens=200, temperature=0.7, num_beams=5, early_stopping=True)
    refined_caption = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return refined_caption

def generate_caption(whole, class_hier, refine=False, hier_level=None):
    # Start with the object-level description
    caption = f"There is "
    if whole[0] in 'aeiou':
        caption += 'an '
    else:
        caption += 'a '
    caption += f"{whole} in the image composed of "
    if hier_level:
        parts = [whole + ' ' + key for key in class_hier.keys()]
    else:
        parts = [key for key in class_hier.keys()]
    if len(parts) > 1:
        part_dscp = ", ".join(parts[:-1]) + " and " + parts[-1] + '.'
    else:
        part_dscp = parts[0] + '.'
    caption += part_dscp
    
    # Add part-level annotations
    part_descriptions = []
    for part, subparts in class_hier.items():
        if subparts:
            if hier_level == 'object':
                subparts = [whole + ' ' + subpart for subpart in subparts]
            elif hier_level == 'full':
                subparts = [whole + ' ' + part + ' ' + subpart for subpart in subparts]
            # Start the part description
            part_text = f" The {part}"
            # Add subpart descriptions, if they exist
            if len(subparts) > 1:
                subpart_text = ", ".join(subparts[:-1]) + " and " + subparts[-1]
            else:
                subpart_text = subparts[0]

            part_text += f" has finer structures including {subpart_text}"
            
            # Append part description
            part_descriptions.append(part_text)
    
    # Join part descriptions into the final caption
    caption += ";".join(part_descriptions) + "."
    
    if refine:
        caption = refine_caption(caption)
    return caption

def generate_prompts(image_pil, masks, bboxes, labels, caption, output_dir, file_name, use_bbox=True, use_mask=False):
    output = {
        "caption":"",
        "width":512,
        "height":512,
        "annos":[],
        "image":image_pil
    }
    output['image'] = encode_pillow_to_base64(output['image'])
    output["caption"]=caption
    for mask, bbox, label in zip(masks, bboxes, labels):
        rle = mask_2_rle(mask)
        anno = {}
        if use_mask:
            anno["mask"]=rle
        if use_bbox:
            anno["bbox"]=bbox
        anno["caption"]=label
        output["annos"].append(anno)
    with open(os.path.join(output_dir, "{}.json".format(file_name)), 'w') as f:
        json.dump(output, f)

def main_coco(args):
    dataDir='../..'
    dataType='train2017'
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco_caps=COCO(annFile)

    print("Initialize CLIP model")
    clip_text_model, clip_text_processor = load_clip()

    # make dir
    # output_dir = args.output_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # load images for model inference
    dataset_name = 'COCO'

    ## Get image IDs
    # Get all image IDs in the dataset
    img_ids = coco.getImgIds()
    print(f"Total number of images in COCO train: {len(img_ids)}")

    for id in tqdm.tqdm(img_ids):
        img_meta_data = {} # store image meta data
        image_info = coco.loadImgs(id)[0]
        file_name = image_info["file_name"]

        image_path = f"{dataDir}/{dataType}/{file_name}"

        # read image and convert to RGB image using PIL
        image_pil = Image.open(image_path).convert("RGB") # load image

        # save raw image
        img_meta_data['image'] = encode_pillow_to_base64(image_pil.convert('RGB'))

        # save file name
        img_meta_data['file_name'] = image_path

        annIds = coco.getAnnIds(imgIds=id)
        anns = coco.loadAnns(annIds)
        if not anns:
            continue
        
        masks = []
        bboxes = []
        labels = []
        for ann in anns:
            mask = coco.annToMask(ann)
            masks.append(mask)
            bboxes.append(ann['bbox'])
            category_info = coco.loadCats(ann['category_id'])
            labels.append(category_info[0]['name'])

        # generate and save image caption
        annIds = coco_caps.getAnnIds(imgIds=id)
        caps = coco_caps.loadAnns(annIds)
        img_meta_data['caption'] = caps[0]['caption']

        # get base output dictionary
        output = get_base_output_dict(image_pil, dataset_name, file_name, data=img_meta_data)

        # save mask data
        save_mask_data(output_dir, masks, bboxes, labels, file_name, output, clip_text_model, clip_text_processor)

def main(args):

    annotation_dir = "../../PartImageNet/jsons"
    image_dir = "../../PartImageNet/images"
    split = args.split
    spin_api = SPIN(
        annotation_dir=annotation_dir, image_dir=image_dir, split=split, download=True
    )
    print("Initialize CLIP model")
    clip_text_model, clip_text_processor = load_clip()

    # make dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # load images for model inference
    dataset_name = 'SPIN'

    ## Get image IDs
    # Get all image IDs in the dataset
    img_ids = spin_api.getImgIds()
    print(f"Total number of images in split {spin_api.split}: {len(img_ids)}")

    for id in tqdm.tqdm(img_ids):
        img_meta_data = {} # store image meta data
        image_info = spin_api.subparts.loadImgs(id)[0]
        file_name = image_info["file_name"]

        image_path = f"{spin_api.image_dir}/{spin_api.split}/{file_name}.JPEG"
        
        # read image and convert to RGB image using PIL
        image_pil = Image.open(image_path).convert("RGB") # load image
        
        # save raw image
        img_meta_data['image'] = encode_pillow_to_base64(image_pil.convert('RGB'))
        
        # save file name
        img_meta_data['file_name'] = image_path

        wholes, parts, subparts = get_annotations(spin_api.wholes, id), get_annotations(spin_api.parts, id), get_annotations(spin_api.subparts, id)
        
        subpart_categories = spin_api.get_categories(granularity="subpart")
        subparts['labels'] = [subpart_categories[label_id]['name'].lower() for label_id in subparts['label_ids']]
        
        part_categories = spin_api.get_categories(granularity="part")
        parts['labels'] = [part_categories[label_id]['name'].lower() for label_id in parts['label_ids']]
        parts['labels'] = [label.replace("tier", "tire") for label in parts['labels']]

        whole_categories = spin_api.get_categories(granularity="whole")
        wholes['labels'] = [whole_categories[label_id]['name'] for label_id in wholes['label_ids']]
        
        # print(subparts['labels'], parts['labels'], wholes['labels'])
        class_hier = get_subpart_by_part(parts, subparts)
        # print(class_hier)
        whole = spin_api.get_object_name_for_file(id)

        # extract exact name
        try:
            synset = wn.synset_from_pos_and_offset('n', int(file_name.split('_')[0][1:]))
            label = synset.name().split('.')[0].replace('_', ' ')
            # print(label)
        except Exception as e:
            print(f"Could not find label for {id}: {e}")
            label = whole

        # generate and save image caption
        img_meta_data['caption'] = generate_caption(whole, class_hier, args.caption_refine, args.hier_level)
        
        # # get base output dictionary
        output = get_base_output_dict(image_pil, dataset_name, file_name, data=img_meta_data)
        
        if args.gran == 'full':
            masks = subparts['masks'] + parts['masks'] + wholes['masks']
            bboxes = subparts['bboxes'] + parts['bboxes'] + wholes['bboxes']
            labels = subparts['labels'] + parts['labels'] + [whole]
        elif args.gran == 'part':
            masks = parts['masks'] + wholes['masks']
            bboxes = parts['bboxes'] + wholes['bboxes']
            labels = parts['labels'] + [whole]
        elif args.gran == 'object':
            masks = wholes['masks']
            bboxes = wholes['bboxes']
            labels = [whole]
        else: 
            raise ValueError("Invalid input provided.")       

        # # save mask data
        save_mask_data(output_dir, masks, bboxes, labels, file_name, output, clip_text_model, clip_text_processor)
        # # print("Processed {} image; {}".format(num_images, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser('InstDiff training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.caption_refine:
        # Initialize the model and tokenizer
        model_name = "meta-llama/Llama-2-7b-chat-hf"  # replace with the model size you need
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_api_key)
        model = AutoModelForCausalLM.from_pretrained(model_name,token = hf_api_key, torch_dtype=torch.float16, device_map="auto")
    if args.coco:
        main_coco(args)
    else:
        main(args)