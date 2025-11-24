import os
import sys
import re
from typing import Optional, Dict, Any

import torch
import numpy as np
from Levenshtein import ratio


def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    """
    Calculate mean relative accuracy (MRA) for regression tasks.
    
    Args:
        pred: Predicted value.
        target: Ground truth value.
        start: Start threshold for relative accuracy.
        end: End threshold for relative accuracy.
        interval: Interval between thresholds.
    
    Returns:
        mra: Mean relative accuracy.
    """    
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


def compute_accuracy_reward(response: str, ground_truth: str):
    """
    Compute exact match accuracy reward.
    
    Args:
        response: Model output string.
        ground_truth: Ground truth string.
    
    Returns:
        reward: 1.0 if exact match, else 0.0
    """
    if response is None:
        return 0.0
    return 1.0 if response.strip() == ground_truth.strip() else 0.0

def compute_mra_reward(response: str, ground_truth: str):
    """
    Compute regression reward based on mean relative accuracy (MRA).
    
    Args:
        response: Model output string representing a number.
        ground_truth: Ground truth string representing a number.
    
    Returns:
        reward: MRA value or 0.0 if inputs are invalid.
    """
    if response is None or ground_truth is None:
        return 0.0
    return mean_relative_accuracy(float(response), float(ground_truth))


# -----------------------------------------------------------
# VSI Benchmark score computation
# -----------------------------------------------------------
def vsi_compute_score(metric: str, response: str, ground_truth: str) -> float:
    """
    Compute VSI benchmark score based on the specified metric.
    
    Args:
        metric: Metric type, e.g., 'acc' or 'mra'.
        response: Model output.
        ground_truth: Ground truth value.
    
    Returns:
        score: Computed score.
    """ 
    if metric == 'acc':
        return compute_accuracy_reward(response, ground_truth)
    elif metric == 'mra':
        return compute_mra_reward(response, ground_truth)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


# -----------------------------------------------------------
# SAT Benchmark score computation
# -----------------------------------------------------------
def sat_compute_score(metric: str, response: str, ground_truth: str) -> float:
    """
    Compute SAT benchmark score for multiple choice / textual answers.
    
    Args:
        metric: Metric type (currently ignored, always exact match).
        response: Model output string.
        ground_truth: Ground truth string.
    
    Returns:
        score: 1.0 if correct, 0.0 otherwise.
    """
    pred_answers = response.strip()

    format_answer = pred_answers.split("\n")[0].split("###")[-1].strip().lower()
    gt_answer = ground_truth.lower().strip()

    correct = gt_answer in format_answer or format_answer in gt_answer
        
    if "rotated left and rotated right" in gt_answer:
        if "rotated left and rotated right" in format_answer or "rotated right and rotated left" in format_answer or "did not move" in format_answer:
            correct = True
    
    return 1.0 if correct else 0.0