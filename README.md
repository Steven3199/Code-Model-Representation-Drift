
# Understanding and Mitigating Representation Drift of Code Models under Software Distribution Shifts

This repository provides the code and datasets for our empirical study on the robustness of pretrained code models under software distribution shifts.

We investigate **OOD generalization, OOD detection, layer-wise representation diagnosis, and model adaptation** across seven downstream software engineering (SE) tasks, five pretrained models, and five distribution shift types.

## 1. Repository Structure

```text
Code-Model-Representation-Drift/
├── code/                           # Experimental implementations
│   ├── CodeS/
│   ├── CodeSearchNet/
│   ├── CodeUpdateArena/
│   ├── CodeXGLUE-BigCloneBench/
│   ├── CodeXGLUE-Bugs2Fix/
│   ├── CodeXGLUE-CodeTrans/
│   └── CodeXGLUE-Devign/
│
├── dataset/                        # Experimental datasets
│   ├── CodeS/
│   ├── CodeSearchNet/
│   ├── CodeUpdateArena/
│   ├── CodeXGLUE-BigCloneBench/
│   ├── CodeXGLUE-Bugs2Fix/
│   ├── CodeXGLUE-CodeTrans/
│   └── CodeXGLUE-Devign/
│
├── .gitattributes
└── README.md
```

## 2. Environment

Our experiments were conducted in Linux environments using Python and PyTorch.

The required dependencies may vary across models and downstream tasks. Please configure the corresponding environment before running each experiment.

## 3. Reproducing the Experiments

**Step 1: Clone the repository**

```bash
git clone https://github.com/Steven3199/Code-Model-Representation-Drift.git
cd Code-Model-Representation-Drift
```

**Step 2: Prepare the datasets and models**

The `dataset/` directory contains the experimental datasets organized by benchmark, while the `code/` directory contains the corresponding implementations.

Download the required pretrained model checkpoints from [Hugging Face](https://huggingface.co/) and configure their paths in the relevant scripts.

**Step 3: Run the experiments**

Navigate to the corresponding directory under `code/` and configure the dataset paths, model checkpoints, and experimental parameters.

The experiments are organized into five research questions:

- **RQ1 — OOD Generalization:** Evaluate ID and OOD performance under different distribution shifts.
- **RQ2 — OOD Detection:** Evaluate task-adapted OOD detection methods.
- **RQ3 — Representation Diagnosis:** Analyze layer-wise representations using Alignment, Anisotropy, Effective Rank, and Entropy.
- **RQ4 — Traditional Adaptation:** Evaluate Transfer Learning (TL), Elastic Weight Consolidation (EWC), and Replay.
- **RQ5 — Diagnostics-guided Adaptation:** Perform representation-guided data allocation, TPE-based configuration search, Pareto selection, and configuration transfer.

Run the corresponding experimental scripts for the selected dataset, model, and research question.

## 4. Experimental Settings

We evaluate five representative pretrained models:

- RoBERTa
- CodeBERT
- GraphCodeBERT
- CodeT5
- PLBART

Five types of distribution shifts are considered:

- **CST Shift**
- **Token Shift**
- **User Shift**
- **Time Shift**
- **Language Shift**

For further details on experimental configurations, evaluation metrics, and results, please refer to our paper.
