# SIGIR26-StructAlign
StructAlign: Structured Cross-Modal Alignment for Continual Text-to-Video Retrieval [SIGIR 26]

---

## 👥 Authors

Shaokun Wang<sup>1</sup>, Weili Guan<sup>1*</sup>, Jizhou Han<sup>2</sup>, Jianlong Wu<sup>1</sup>, Yupeng Hu<sup>3</sup>, Liqiang Nie<sup>1</sup>

<sup>1</sup> Harbin Institute of Technology (Shenzhen)   <sup>2</sup> Xi'an Jiaotong University  <sup>3</sup> Shandong University

\* Corresponding author

---

## 🔗 Links

- 📄 **Paper:** [ACM Digital Library](https://dl.acm.org/doi/10.1145/3805712.3809704)
- 💻 **Code Repository:** [GitHub](https://github.com/Mysteriousplayer/SIGIR26-StructAlign)

---

## 📢 Updates

- **[04/2026]** Paper accepted at SIGIR 2026
- **[07/2026]** Initial open-source release

---

## 📌 Overview

- 🎯 Task: Continual Text-to-Video Retrieval (CTVR)
- 🧠 Problem: Intra-modal drift + cross-modal misalignment
- ⚙️ Key Idea: Structured cross-modal alignment via ETF geometry 
- 🚀 Result: Good performance on MSRVTT & ACTNET 

---

## 📖 Abstract
![image](https://github.com/Mysteriousplayer/SIGIR26-StructAlign/blob/main/concept.png)

> Continual Text-to-Video Retrieval (CTVR) is a challenging multimodal continual learning setting, where models must incrementally learn new semantic categories while maintaining accurate text–video alignment for previously learned ones, thus making it particularly prone to catastrophic forgetting. A key challenge in CTVR is feature drift, which manifests in two forms: intra-modal feature drift caused by continual learning within each modality, and non-cooperative feature drift across modalities that leads to modality misalignment. To mitigate these issues, we propose StructAlign, a structured cross-modal alignment method for CTVR. First, StructAlign introduces a simplex Equiangular Tight Frame (ETF) geometry as a unified geometric prior to mitigate modality misalignment. Building upon this geometric prior, we design a cross-modal ETF alignment loss that aligns text and video features with category-level ETF prototypes, encouraging the learned representations to form an approximate simplex ETF geometry. In addition, to suppress intra-modal feature drift, we design a Cross-modal Relation Preserving loss, which leverages complementary modalities to preserve cross-modal similarity relations, providing stable relational supervision for feature updates. By jointly addressing non-cooperative feature drift across modalities and intra-modal feature drift, StructAlign effectively alleviates catastrophic forgetting in CTVR. Extensive experiments on benchmark datasets demonstrate that our method shows competitive advantages over state-of-the-art continual retrieval approaches.

---

## 🏗️ Framework

![image](https://github.com/Mysteriousplayer/SIGIR26-StructAlign/blob/main/framework.png)

- We propose StructAlign, a structured cross-modal alignment framework for CTVR, which explicitly models and mitigates catastrophic forgetting induced by both intra-modal feature drift and non-cooperative feature drift across modalities. 

- We introduce a simplex ETF geometric prior together with a cross-modal ETF alignment loss to enforce a well-separated category-level structure in the shared embedding space. In addition, we design a cross-modal relation preserving loss that leverages cross-modal similarity relations to constrain intra-modal feature updates during continual learning. 

---

## 📊 Datasets and Protocols

### 🎬 MSRVTT
- **Video clips**: 10,000 short videos    
- **Type**: Short video–text paired dataset

### 🎥 ACTNET
- **Video clips**: ~20,000 long, untrimmed videos     
- **Type**: Long-form activity recognition dataset  

## ⚙️ Evaluation Protocol

We follow the **CTVR protocol** proposed in [StableFusion](https://github.com/JasonCodeMaker/CTVR):

- All categories are evenly divided into **K tasks**
- Two settings are used:
  - **K = 10**
  - **K = 20**

| Dataset | #Category | Shot/Category | #Task   |
|---------|-----------|---------------|---------|
| MSRVTT  | 20        | 16            | K=10,20 |
| ACTNET  | 200       | 16            | K=10,20 |

## ⚙️ Repository Structure

```text
SIGIR26-StructAlign/
├── config/          # training and evaluation configs
├── datasets/        # dataset loaders and preprocessing utilities
├── evaluator/       # evaluation logic
├── model/           # model definitions
├── modules/         # losses and utilities
├── scripts/         # training / evaluation entry scripts
├── trainer/         # training framework
├── main.py          # training / evaluation launcher
└── requirements.txt
```

> Note: the `data/` directory is not included in this repository. Continual split files such as `data/MSRVTT_10_dataset.pkl` and `data/ACTNET_10_dataset.pkl` are required by the training scripts, and must be prepared by the user.

---

## ⚙️ Installation
Install the required dependencies in a Python environment.

### 1. Clone the repository

```bash
git clone https://github.com/Mysteriousplayer/SIGIR26-StructAlign.git
cd SIGIR26-StructAlign
```

### 2. Create and activate a conda environment

```bash
conda create -n structalign python=3.10 -y
conda activate structalign
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

---

## 🔄 Data Processing

### 🎬 MSRVTT
```
# Download MSRVTT data
wget https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip
unzip MSRVTT.zip -d datasets/MSRVTT

# Place raw videos in:
datasets/MSRVTT/MSRVTT_Videos

# Extract frame folders used by training
python datasets/utils/process_msrvtt.py

# Training scripts will read from:
datasets/MSRVTT/MSRVTT_Frames
```

### 🎥 ACTNET
```
# Download ActivityNet data from the official website
http://activity-net.org/download.html

# Place raw videos under a local ACTNET directory, then process them
python datasets/utils/process_actnet.py
```

The ACTNET training script currently uses an absolute `--videos_dir` path in `scripts/a_train_structalign.sh`. Before running ACTNET experiments, please update that path to your local processed frame directory.

### Expected directory structure

```text
SIGIR26-StructAlign/
├── config/
├── data/                     # required continual split .pkl files (user-prepared)
├── datasets/
│   ├── MSRVTT/
│   │   ├── MSRVTT_Videos/
│   │   └── MSRVTT_Frames/
│   └── ACTNET/              # optional local raw-data placement
├── model/
├── scripts/
└── trainer/
```

---

## 🏃 Training
After preparing the datasets, you can launch training with the provided scripts. Each script contains separate blocks for the 10-task and 20-task settings, so please comment or uncomment the block you want before running.

### MSRVTT

```bash
bash scripts/train_structalign.sh
```

### ACTNET

```bash
bash scripts/a_train_structalign.sh
```

---

## Evaluation

Use the corresponding evaluation block in the provided shell scripts, or run evaluation with the saved checkpoints by setting the appropriate config and `eval_path`. The script files already include example eval commands for both the 10-task and 20-task settings.

```bash
python main.py --config <config_path> --eval_path <checkpoint_dir>
```

---

## Main Configurations

- `config/sa_config.yaml`: main configuration for MSRVTT experiments
- `config/actnet_sa_config.yaml`: main configuration for ACTNET experiments

---

## 📈 Results
Results and logs will be saved under the specified output directory.   

---

## 🔍 Limitations
- **Predefined class number requirement:**  
  This work constructs the Simplex ETF geometry under the assumption that the total number of classes is known in advance. Although this requirement can be mitigated in practice by setting a sufficiently large upper bound on the number of classes, it may introduce mild redundancy in the representation space and reduce parameter efficiency in some scenarios.

- **Additional training overhead from regularization terms:**  
  The proposed cross-modal ETF alignment loss and cross-modal relation preserving loss are both regularization-based strategies. While they effectively improve anti-forgetting capability in continual learning, they also introduce a modest increase in training cost. 

---

## 📝 Citation
If you found our work useful for your research, please cite our work:
```
@inproceedings{StructAlign,
author = {Wang, Shaokun and Guan, Weili and Han, Jizhou and Wu, Jianlong and Hu, Yupeng and Nie, Liqiang},
title = {StructAlign: Structured Cross-Modal Alignment for Continual Text-to-Video Retrieval},
year = {2026},
booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1800-1811},
numpages = {12}
}
```

---

## 🙏 Acknowledgments
We thank the following repo providing helpful functions in our work. 

[StableFusion](https://github.com/JasonCodeMaker/CTVR)

[POLO](https://github.com/Mysteriousplayer/MM23-POLO)
