import os
import io
import json
import random
from typing import List, Tuple, Optional

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------
# Constants
# -----------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SFT_QUESTION_TEMPLATE = "{Question}"
SFT_TYPE_TEMPLATE = {
    "multiple choice": " Do not respond with anything other than a single letter!",
    "regression": " Do not respond with anything other than a single number!",
}

# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------
def save_image_bytes(image_bytes, save_path="output.jpg"):
    """
    Save image bytes or numpy array to a file.
    """
    if isinstance(image_bytes, np.ndarray):
        if image_bytes.dtype.kind == 'O':
            image_bytes = image_bytes[0]
        elif image_bytes.dtype == np.uint8:
            image_bytes = image_bytes.tobytes()
        else:
            raise ValueError(f"Unsupported ndarray dtype: {image_bytes.dtype}")

    if isinstance(image_bytes, bytes):
        image = Image.open(io.BytesIO(image_bytes))
        image.save(save_path)
    else:
        raise ValueError(f"Unsupported type: {type(image_bytes)}")


def build_transform(input_size):
    """
    Build a torchvision transform for image preprocessing.
    """
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    Find the target ratio closest to the input aspect ratio.
    """
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Preprocess an image dynamically by splitting into blocks and optionally adding a thumbnail.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    """
    Load an image file and return a tensor of pixel values.
    """
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# -----------------------------------------------------------
# Video functions
# -----------------------------------------------------------
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    """
    Compute frame indices for video segment extraction.
    """
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    Load video frames and return pixel values tensor and patch counts.
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


# -----------------------------------------------------------
# Dataset classes
# -----------------------------------------------------------
class SAT:
    def __init__(self, data_path: str, file_name: str):
        self.data_path = data_path
        self.file_name = file_name
        file_path = os.path.join(self.data_path, self.file_name)
        self.dataset = pd.read_parquet(file_path)
        self.prompts = []
        self.process()
        print(f"Constructed input_prompts with {len(self.prompts)} items.")
    
    def format_prompts(self, images, problem, answer_choices, answer):
        """
        Format SAT prompts with images, problem text, and answer choices.
        """
        correct_answer = answer
        if len(answer_choices)>1:   
            ans_choice_order = answer_choices
            ans_choice_order = ['"'+ans+'"' for ans in ans_choice_order]
            random.shuffle(ans_choice_order)
            answer_choices_format = " or ".join(ans_choice_order)
        else:
            answer_choices_format = ""

        image_prompt_format = "<image>"*len(images)
        prefix = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            
        prompt = f"{prefix}###Human: <im_start>{image_prompt_format}<im_end> \nHuman: Answer in natural language. {problem} Answer the question using a single word or phrase. "

        if answer_choices_format != "":
            prompt += f"Choose between the following options: {answer_choices_format}. ###Assistant: \n"
        else:
            prompt += f"###Assistant: \n"

        text_label = prompt + correct_answer + " \n###"    
        return prompt, text_label

    def process(self):
        """
        Process dataset to create prompts with pixel_values and ground truths.
        """
        start_idx = 0
        end_idx = 49
        for idx, row in self.dataset.iterrows():
            if idx < start_idx:
                continue
            if idx > end_idx:
                break
            images = row['image_bytes']
            all_pixel_values = []   
            num_patches_list = []   
            for img in images:
                save_image_bytes(img, "./image.png")
                pixel_values = load_image("./image.png")
                num_patches_list.append(pixel_values.size(0)) 
                all_pixel_values.append(pixel_values)

            pixel_values = torch.cat(all_pixel_values, dim=0).to(device="cuda", dtype=torch.bfloat16)    
            problem = row['question']
            answer_choices = row['answers']
            corrected_answer_choices = []
            for answer in answer_choices:
                if "in the first frame" in answer:
                    answer = answer.replace("in the first frame", "")
                corrected_answer_choices.append(answer)
            answer_choices = corrected_answer_choices

            ans_choice_order = answer_choices
            ans_choice_order = ['"'+ans+'"' for ans in ans_choice_order]
            random.shuffle(ans_choice_order)
            final_prompt, final_label = self.format_prompts(images, problem, answer_choices, answer)

            item = [
                {
                "role": "user", 
                "question": final_prompt,
                "pixel_values": pixel_values,
                "ground_truth": row['correct_answer'],
                "question_type": row['question_type'],
                "num_patches_list": num_patches_list
                }
            ]
            self.prompts.append(item)


class VSI_Bench:
    def __init__(self, data_path: str, file_name: str, question_type: str):
        self.data_path = data_path
        self.file_name = file_name 
        self.question_type = question_type
        self.dataset = self.load_vsi_evalset()
        self.prompts = []
        self.process()
    
    def process(self):
        for idx, row in enumerate(tqdm(self.dataset, desc="Processing vsibench batches")):
            if row['original_question_type'] != self.question_type:
                continue
            video_path = self.data_path + row['path']
            pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
            pixel_values = pixel_values.to(device="cuda", dtype=torch.bfloat16)
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            problem = video_prefix + row['problem']
            problem_type = row['problem_type']
            options = row['options']
            original_problem_type = row['original_question_type']
            if problem_type == "multiple choice":
                problem += "\nOptions:\n"
                for op in options:
                    problem += op + "\n"
            item = [
                {
                "role": "user", 
                "question": SFT_QUESTION_TEMPLATE.format(Question=problem) + SFT_TYPE_TEMPLATE[problem_type],
                "pixel_values": pixel_values,  
                "question_type": problem_type, 
                "num_patches_list": num_patches_list,
                "ground_truth": row['solution'],
                "original_question_type": original_problem_type,
                "options": options,
                }
            ]
            self.prompts.append(item)
        print(f"Constructed imput_prompts with {len(self.prompts)} items.")
    
    def load_vsi_evalset(self):
        file_path = os.path.join(self.data_path, self.file_name)
        print("file_path: ", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data