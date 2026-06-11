# SIGIR26-StructAlign
StructAlign: Structured Cross-Modal Alignment for Continual Text-to-Video Retrieval [SIGIR 26]

## 👥 Authors

Shaokun Wang<sup>1</sup>, Weili Guan<sup>1*</sup>, Jizhou Han<sup>2</sup>, Jianlong Wu<sup>1</sup>, Yupeng Hu<sup>3</sup>, Liqiang Nie<sup>1</sup>

<sup>1</sup> Harbin Institute of Technology (Shenzhen)  
<sup>2</sup> Xi'an Jiaotong University
<sup>3</sup> Shandong University

\* Corresponding author

## 🔗 Links

- 📄 **Paper:** [ACM Digital Library](https://...)
- 💻 **Code Repository:** [GitHub](https://github.com/Mysteriousplayer/SIGIR26-StructAlign)

## 📢 Updates

- **[04/2026]** Paper accepted at SIGIR 2026
- **[07/2026]** Initial open-source release

## 📖 Abstract
![image](https://github.com/Mysteriousplayer/SIGIR26-StructAlign/blob/main/fig1.png)

> Continual Text-to-Video Retrieval (CTVR) is a challenging multimodal continual learning setting, where models must incrementally learn new semantic categories while maintaining accurate text–video alignment for previously learned ones, thus making it particularly prone to catastrophic forgetting. A key challenge in CTVR is feature drift, which manifests in two forms: intra-modal feature drift caused by continual learning within each modality, and non-cooperative feature drift across modalities that leads to modality misalignment. To mitigate these issues, we propose StructAlign, a structured cross-modal alignment method for CTVR. First, StructAlign introduces a simplex Equiangular Tight Frame (ETF) geometry as a unified geometric prior to mitigate modality misalignment. Building upon this geometric prior, we design a cross-modal ETF alignment loss that aligns text and video features with category-level ETF prototypes, encouraging the learned representations to form an approximate simplex ETF geometry. In addition, to suppress intra-modal feature drift, we design a Cross-modal Relation Preserving loss, which leverages complementary modalities to preserve cross-modal similarity relations, providing stable relational supervision for feature updates. By jointly addressing non-cooperative feature drift across modalities and intra-modal feature drift, StructAlign effectively alleviates catastrophic forgetting in CTVR. Extensive experiments on benchmark datasets demonstrate that our method shows competitive advantages over state-of-the-art continual retrieval approaches.

## 🏗️ Framework

![image](https://github.com/Mysteriousplayer/SIGIR26-StructAlign/blob/main/fig2.png)

-We propose StructAlign, a structured cross-modal alignment framework for CTVR, which explicitly models and mitigates catastrophic forgetting induced by both intra-modal feature drift and non-cooperative feature drift across modalities. 

-We introduce a simplex ETF geometric prior together with a cross-modal ETF alignment loss to enforce a well-separated category-level structure in the shared embedding space. In addition, we design a cross-modal relation preserving loss that leverages cross-modal similarity relations to constrain intra-modal feature updates during continual learning.

-Extensive experiments on benchmark datasets demonstrate that StructAlign achieves competitive performance compared to state-of-the-art continual retrieval methods.

## 📊 Datasets and Protocols

## ⚙️ Installation
Install all requirements required to run the code on a Python 3.x by:
> First, you need activate a new conda environment.
> 
> pip install -r requirements.txt

## 🔄 Data Processing
All commands should be run under the project root directory. 

```
sh data_processing.sh
```

## 🏃 Training
After downloading the datasets you need, you can use this command to obtain training samples used in few-shot and easy-to-hard classification task.

```
sh run.sh
```

## 📈 Results
Results will be saved in log/.  

## 🔍 Limitations

## 📝 Citation
If you found our work useful for your research, please cite our work:
```
@inproceedings{StructAlign,
author = {Wang, Shaokun and Guan, Weili and Han, Jizhou and Wu, Jianlong and Hu, Yupeng and Nie, Liqiang},
title = {StructAlign: Structured Cross-Modal Alignment for Continual Text-to-Video Retrieval},
year = {2026},
booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {797–806},
numpages = {10}
}
```

## 🙏 Acknowledgments
We thank the following repo providing helpful functions in our work. 

DINO: https://github.com/facebookresearch/dino
