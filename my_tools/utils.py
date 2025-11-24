import re
import torch
from typing import Optional, Dict, Any
from Levenshtein import ratio


# -----------------------------------------------------------
# Answer extraction functions
# -----------------------------------------------------------
def extract_answer_letter(text):
    """
    Extract a single choice letter (A/B/C/D) from model output.
    Priority: match beginning letter first, then the first standalone letter in the text.
    """
    if not text:
        return None

    text = text.strip()

    # Match letters at the beginning: "A.", "B -", "C", etc.    match = re.match(r"^\s*([A-Da-d])[\.\s\-:]", text)
    if match:
        return match.group(1).upper()

    # Match the first standalone letter A/B/C/D in the text
    match = re.search(r"\b([A-Da-d])\b", text)
    if match:
        return match.group(1).upper()

    # Fallback: take the first occurrence of A/B/C/D
    letters = re.findall(r"[A-Da-d]", text)
    if letters:
        return letters[0].upper()

    return None

def extract_answer_number(text):
    """
    Extract a number (integer or float) from text.
    Returns the last number found, usually the answer.
    """
    # Match integer or float
    pattern = r'<*([0-9]+(?:\.[0-9]+)?)>*'
    matches = re.findall(pattern, text)
    if matches:
        return float(matches[-1])
    
    # Attempt extraction from phrases like "approximately X" or "the answer is X"
    approx_pattern = r'(?:approximately|the answer is|answer:)\s*([0-9]+(?:\.[0-9]+)?)'
    approx_matches = re.findall(approx_pattern, text, flags=re.IGNORECASE)
    if approx_matches:
        return float(approx_matches[-1])
    return None


def extract_answer(text: str) -> str:
    """Extract answer enclosed by <answer> tags."""
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# -----------------------------------------------------------
# Text cleaning utilities
# -----------------------------------------------------------
def clean_choice_text(text, exclue_chars=["\n", "\r"]):
    """Base text cleaning: remove specified characters and extract <answer> content."""
    # Extract <answer> content if present
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        text = answer_matches[-1]

    for char in exclue_chars:
        if char in ["\n", "\r"]:
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")

    text = extract_answer_letter(text)
    if text == None:
        return None
    return text.strip().rstrip(".").lower()


def clean_number_text(text, exclue_chars=["\n", "\r"]):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        text = answer_matches[-1]

    for char in exclue_chars:
        if char in ["\n", "\r"]:
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")

    text = extract_answer_number(text)
    return text


def clean_text(text, exclue_chars=["\n", "\r"]):
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_matches:
        # Use the last match
        text = answer_matches[-1]

    for char in exclue_chars:
        if char in ["\n", "\r"]:
            # If there is a space before the newline, remove the newline
            text = re.sub(r"(?<=\s)" + re.escape(char), "", text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r"(?<!\s)" + re.escape(char), " ", text)
        else:
            text = text.replace(char, " ")

    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip(".").lower()

def normalize_number(num_str: str) -> Optional[float]:
    """Convert string number to float, handling commas."""
    try:
        num_str = num_str.replace(",", "")
        return float(num_str)
    except Exception:
        return None


def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    """Calculate mean relative accuracy for regression tasks."""
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not torch.is_tensor(target):
        target = torch.tensor(target, dtype=torch.float32)

    epsilon = 1e-8
    rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)

    thresholds = torch.arange(start, end + interval / 2, interval, dtype=torch.float32)
    conditions = rel_error < (1 - thresholds)
    mra = conditions.float().mean()
    return mra.item()


def vsi_reward(clean_ans_gt: str, clean_ans_pred: str, question_type: str) -> float:
    """Calculate reward based on question type and model output."""
    if question_type == "multiple choice":
        return 1.0 if clean_ans_pred.strip() == clean_ans_gt.strip() else 0.0
    elif question_type == "regression" or question_type == "numerical":
        gt_number = normalize_number(clean_ans_gt)
        pred_number = normalize_number(clean_ans_pred)
        if gt_number is None or pred_number is None:
            return 0.0
        return mean_relative_accuracy(pred_number, gt_number)
    else:
        raise ValueError(f"Unsupported question type: {question_type}")


# -----------------------------------------------------------
# Postprocessing
# -----------------------------------------------------------
def postprocess(prompts, model_outputs, data_name):
    """
    Postprocess model outputs: clean answers, extract standardized results.
    """
    results = []
    
    for idx, prompt in enumerate(prompts):
        model_output = model_outputs[idx]
        result_sample = {}
        result_sample['question'] = prompt[0]['question']
        result_sample['model_output'] = model_output
        result_sample['ground_truth'] = prompt[0]['ground_truth']
        if data_name == 'SAT': 
            result_sample['question_type'] = prompt[0]['question_type']
        elif data_name == 'VSI_Bench':
            result_sample['question_type'] = prompt[0]['original_question_type']
        
        if data_name == 'VSI_Bench':
            if prompt[0]['question_type'] == "multiple choice":
                clean_ans = clean_choice_text(model_output)
                clean_ans_gt = clean_choice_text(prompt[0].get("ground_truth", ""))
                result_sample['metric'] = 'acc'
            else:
                clean_ans = clean_number_text(model_output)
                clean_ans_gt = clean_number_text(prompt[0].get("ground_truth", ""))
                result_sample['metric'] = 'mra'
        else:
            result_sample['metric'] = 'acc'
            clean_ans = clean_text(model_output)
            clean_ans_gt = clean_text(prompt[0].get("ground_truth", ""))
        
        result_sample["cleaned_model_output"] = clean_ans
        result_sample["cleaned_gt_answer"] = clean_ans_gt
        
        results.append(result_sample)
    return results