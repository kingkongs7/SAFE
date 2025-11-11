# 🔍 My Deepfake Detection Benchmark Extension

This repository accompanies our research paper on **SAFE**. Our work is built upon and extends the [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) framework, aiming to explore more accurate and generalizable detection pipelines through **multi-modal alignment**, **curriculum learning**, and **LoRA-based fine-tuning** on CLIP-like backbones.

## 📄 Paper Overview

* **Title**: *\[SAFE: Semantic- and Frequency-Enhanced Curriculum for Cross-Domain Deepfake Detection]*
* **Authors**: \[Yulin Yao, Kangfeng Zheng, Bin Wu, Jvjie Wang, Jiaqi Gao], et al.
* **Conference/Journal**: \[AAAI 2026]
* **Abstract**:
  \[Insert a concise version of your abstract here, 3-5 sentences summarizing the motivation, methodology, and key results.]

## 🧠 Key Contributions

* 🚧 **Framework Extension**: Based on DeepfakeBench, we introduce additional modules supporting CLIP + LoRA finetuning and multi-modal representations.
* 🔁 **Curriculum Learning**: A progressive data sampling schedule is designed to improve robustness and generalization.
* 📝 **Image-Text Alignment**: Inspired by recent multi-modal methods, we generate textual descriptions using ClipCap to guide representation learning.
* ⚙️ **Lightweight Fine-tuning**: LoRA modules are selectively injected into the CLIP encoder, enabling efficient domain adaptation with minimal parameters.

## 🗂️ Datasets

We conduct experiments primarily on:

* [FF++](https://github.com/ondyari/FaceForensics) (FaceForensics++)
* [DFDC](https://ai.facebook.com/datasets/dfdc)
* [Celeb-DF](https://github.com/yuezunli/Celeb-DF)
* [DF40](https://github.com/YZY-stack/DF40)

Preprocessing and dataloader pipelines follow DeepfakeBench with additional CLIP-based input transformations.
