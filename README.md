# INFANiTE
# [项目名 ProjectName] 🚀  
> 一句话概括你的项目（解决什么问题 / 提供什么能力）

[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](#)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)](#)
<!-- 你也可以加：arXiv / Paper / HuggingFace / Colab / DOI 等 -->

<p align="center">
  <img src="assets/teaser.png" width="85%" alt="teaser">
</p>

---

## ✨ Highlights
- ✅ **[亮点1]**：例如：支持 AoP / HC / GA / 平面分类 / 分割 等任务
- ✅ **[亮点2]**：例如：集成 USFM / SAMUS / nnUNet / FetalCLIP 等模型
- ✅ **[亮点3]**：例如：端到端 pipeline + 可复现训练脚本 + 统一评估指标
- ✅ **[亮点4]**：例如：跨设备数据泛化 / 多中心数据 / 轻量化部署

---

## 🧭 Table of Contents
- [项目概览](#-项目概览)
- [方法与模型](#-方法与模型)
- [环境与安装](#-环境与安装)
- [数据准备](#-数据准备)
- [快速开始](#-快速开始)
- [训练](#-训练)
- [评估](#-评估)
- [推理与可视化](#-推理与可视化)
- [结果](#-结果)
- [引用](#-引用)
- [许可证](#-license)
- [致谢](#-致谢)

---

## 📌 项目概览
本项目旨在 **[一句话说明目标]**。  
覆盖任务包括：
- **AoP 分割与测量**：pubic symphysis (PS) + fetal head (FH)
- **HC 分割与头围估计**：head mask → ellipse fit → HC
- **GA 回归**：从超声图像估计 gestational age
- **标准平面/脑亚平面分类**
- **腹部/胃分割**（可选）

> 若你有论文/报告：可在此补充 “方法动机 + 贡献点 + 适用场景”。

---

## 🧠 方法与模型
本项目集成/复现以下模型（按你的实际情况改）：

### Segmentation
- **USFM**：基于全球多设备超声数据预训练的 foundation model，微调用于下游分割。
- **SAMUS / AutoSAMUS**：面向超声的 SAM 适配方案（可选自动提示）。
- **nnUNet**：自动配置的医学影像分割框架，作为强基线。

### Regression / Classification
- **ConvNeXt-Tiny + 回归头**：ImageNet-1K 预训练初始化，端到端微调 GA。
- **ResNet-50 + 任务头**：加载 RadImageNet 等医学影像预训练权重（如使用）。
- **ViT / UNETR / LoRA / diffusion augmentation**：用于分类或增强（如使用）。

---

## 🧩 环境与安装

### 1) 克隆项目
```bash
git clone [你的repo地址].git
cd [你的repo目录]
