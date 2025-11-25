<div align="center">

# ✨InternVL Evaluation with VERL: Benchmarking InternVL2 on VSI-Bench & InternVL3 on SAT✨

<p align="center">
    <a href="https://github.com/">Duan Sangni</a><sup>1</sup>
    <br>
    <sup>1</sup>The University of Hong Kong
</p>

<img src='https://img.shields.io/badge/Framework-VERL-blue'> &nbsp;&nbsp;&nbsp;&nbsp;
<img src='https://img.shields.io/badge/Model-InternVL2/InternVL3-green'> &nbsp;&nbsp;&nbsp;&nbsp;
<img src='https://img.shields.io/badge/Dataset-VSI_Bench/SAT-orange'>


</div>

---

## Overview

This repository provides a **complete evaluation pipeline built on the VERL multimodal framework**, enabling reproducible benchmarking of **InternVL2 on VSI-Bench** and **InternVL3 on SAT**.  
It includes unified prompt templates, dataset handling, model interfaces, and distributed inference scripts fully compatible with VERL.

---

## 🌟 Key Features

- **InternVL2 + VSI-Bench**  
  Video-based spatial reasoning benchmark covering object counting, spatial relations, distance estimation, and geometry reasoning.

- **InternVL3 + SAT**  
  Image-based Spatial Ability Test (SAT) including mental rotation, symmetry, shape reasoning, and spatial logic tasks.

**Features:**  
✔ VERL-compatible evaluator design  
✔ Unified preprocessing for video and image MLLMs  
✔ Distributed batch inference  
✔ Reproducible metric computation

---

## 💻 Quick Start Diagram

```mermaid
flowchart LR
    A[Dataset] --> B[Preprocessing]
    B --> C[Model Loading]
    C --> D[Distributed Evaluation]
    D --> E[Metric Computation]
```

**Workflow Explanation:**

1. Dataset: Load VSI-Bench videos or SAT images.

2. Preprocessing: Resize, normalize, and tile images or frames; extract patches.

3. Model Loading: Load InternVL2/InternVL3 with Hugging Face transformers or FSDP for distributed inference.

4. Distributed Evaluation: Run batched forward passes and handle multi-frame or multi-image inputs.

5. Metric Computation: Compute Accuracy (acc) or Mean Relative Accuracy (MRA) depending on the task.

---

## 🎉 Performance

### InternVL2 on VSI-Bench

Metrics:

- **Classification tasks (accuracy)**:  
  Object Relative Direction, Object Relative Distance, Route Planning, Object Appearance Order

- **Regression tasks (Mean Relative Accuracy - MRA)**:  
  Object Absolute Distance, Object Counting, Object Size Estimation, Room Size Estimation

Higher values indicate better performance.


#### 📊 Evaluation Results on VSI-Bench 

| Task                          | InternVL2-8B     |
|------------------------------|--------|
| **object_rel_direction_easy**      | 50.23 |
| **object_rel_direction_medium**    | 29.63 |
| **object_rel_direction_hard**      | 25.47 |
| **object_rel_distance**            | 31.13 |
| **route_planning**                 | 31.44 |
| **obj_appearance_order**           | 33.01 |
| **object_abs_distance**            | 24.44 |
| **object_counting**                | 50.23 |
| **object_size_estimation**         | 42.93 |
| **room_size_estimation**           | 32.05 |
| **average**           | 32.85 |

### InternVL3 on SAT

SAT evaluates spatial-temporal reasoning ability from video inputs.  
We report **accuracy (acc)** for all tasks:

- **ego_movement** – Predict the ego agent's movement direction  
- **obj_movement** – Identify moving objects and their movement  
- **action_conseq** – Predict consequences of physical actions  
- **perspective** – Reasoning about viewpoints  
- **goal_aim** – Inferring intentions or goals  

Higher is better.

#### 📊 Evaluation Results on VSI-Bench

| Task               | InternVL3-2B | InternVL3-8B |
|--------------------|--------|-------|
| **ego_movement**   | 53.85 | 44.44 |
| **obj_movement**   | 27.27 | 44.12 |
| **action_conseq**  | 75.00 | 65.85 |
| **perspective**    | 33.33 | 52.08 |
| **goal_aim**       | 28.57 | 48.78 |
| **average**        | 43.60 | 51.06 |

---

## 📂 Project Structure
This project is built on the **volcengine/VERL** framework, providing a unified pipeline for evaluating multimodal models on tasks such as SAT and VS1.

- **`evaluation/`**:Contains core evaluation logic.

- **`my_tools/`**: This folder includes all custom tools implemented for this project.

- **`scripts/`**: Shell scripts for running evaluations with a single command.

- **`verl/`**: The VERL submodule.

- **`outputs/`**: Stores all outputs generated during evaluation.


---

## ⚙️ Setup

### 1. Clone Repository
```bash
git clone https://github.com/sunnyduan-oss/internvl_verl_evaluate.git
cd internvl_eval_verl
```

### 2. Environment Setup

1. **Create conda environment:**

```bash
conda create -n verl_env python=3.11 -y
conda activate verl_env
```

2. **Install required packages for inference and evaluation:**

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
cd transformers && pip install -e . && cd ..
cd verl && pip install -e . && cd ..
```

## 💻 Evaluation

### Evaluation on VSI-Bench

1. **Download the dataset:**
```bash
# download the VSI-Bench dataset from Hugging Face
huggingface-cli download --resume-download nyu-visionx/VSI-Bench --local-dir <dataset-path> --repo-type dataset

unzip <dataset-path>/arkitscenes.zip -d <dataset-path>
unzip <dataset-path>/scannet.zip -d <dataset-path>
unzip <dataset-path>/scannetpp.zip -d <dataset-path>
```

2. **Downdload the model**:

```bash
modelscope download --model 'OpenGVLab/InternVL2-8B' --local_dir <model-path>
```

3. **Run evaluation:**
```bash
bash scripts/run_internvl2_sat.sh \
    --model_path <model-path> \
    --model_name InternVL2-8B \
    --data_path <dataset-path> \
    --data-name VSI_Bench \
    --data_file_name eval_vsibench.json \
    --output_dir <output-path> \
    --reward_function vsi_compute_score
```

### Evaluation on SAT
1. **Downdload the model:**
```bash
modelscope download --model 'OpenGVLab/InternVL3-2B' --local_dir <model-path>
```
2. **Run evaluation:**
```bash
bash scripts/run_internvl2_vsi.sh \
    --model_path <model-path> \
    --model_name InternVL3-2B \
    --data_path <dataset-path> \
    --data-name SAT \
    --data_file_name SAT_test.parquet \
    --output_dir <output-path> \
    --reward_function sat_compute_score
```
---

### ✅ Notes

- Replace placeholders like `<dataset-path>`, `<model-path>`, and `<output-path>` with your local directories.  
- Make sure the models and datasets are compatible with your GPU configuration.  
- VERL handles distributed evaluation automatically for batched inputs.  
- For best reproducibility, use the same Python environment (`verl_env`) and package versions as specified in `requirements.txt`.  
- Ensure `flash-attn` is installed correctly, especially if using GPU inference for large models.  
- Output metrics (accuracy, MRA) are saved under the specified `--output_dir` for further analysis or visualization.  
- When evaluating SAT or VSI-Bench, confirm that all data preprocessing steps (resizing, patching, etc.) are correctly executed.  
