
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

Our experiments were conducted in a Linux environment. The main libraries and their versions are as follows:

| Component | Version |
|---|---|
| OS | Linux (Ubuntu, kernel 5.4.0-100-generic) |
| Python | 3.8.5 |
| TensorFlow | 2.5.0 |
| NumPy | 1.19.5 |
| Pandas | 1.3.4 |
| SciPy | 1.7.2 |
| scikit-learn | 1.0.1 |
| Matplotlib | 3.4.3 |

Additional dependencies are required for the corresponding model implementations.


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
