**Data-Analysis-and-Prediction-on-Academic-Citation-Networks** is a project dedicated to analyzing academic citation networks and predicting the categories of scholarly articles using advanced machine learning and graph-based models. It combines data preprocessing, network analysis, and predictive modeling to gain insights from large-scale graph datasets.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [Results](#results)
- [Files in the Repository](#files-in-the-repository)
- [Authors](#authors)

## Introduction
In this project, we analyze academic citation networks represented as directed graphs, where nodes are articles and edges represent citation relationships. Our goal is to predict the research category of each article using graph neural networks (GNNs). The project involves:
1. Data preprocessing, including normalization and graph setup.
2. Applying and comparing different models (MLP, GCN, GAT).
3. Evaluating model performance using validation accuracy and loss metrics.

## Features
- Comprehensive data preprocessing pipeline:
  - Feature normalization
  - Training-validation split
- Implementation of three graph-based models:
  - **Multi-Layer Perceptron (MLP)**
  - **Graph Convolutional Network (GCN)**
  - **Graph Attention Network (GAT)**
- Insights and visualizations of training performance.

## Setup
### Requirements
This project is implemented in Python and requires the following libraries:
- `PyTorch`
- `PyTorch Geometric`
- `NumPy`
- `Scikit-learn`

Install dependencies with:
```bash
pip install torch torchvision torchaudio torch-geometric numpy scikit-learn
```

### Dataset
The dataset represents an academic citation network with:
- 100,000 nodes (articles)
- 444,288 edges (citations)
- 128 features per node
- 40 target categories (classes)

## Usage

### Data Preprocessing
The preprocessing includes:
1. Feature normalization (mean-centering and scaling).
2. Splitting the data into training and validation subsets (80-20 split).

### Training the Model
To train a model, run the Python script:
```bash
python final_model_training.py
```

### Evaluating the Model
Detailed performance metrics (accuracy and loss) are logged during training. For deeper analysis, refer to the Jupyter notebook provided.

## Results
Model performance highlights:
- **MLP:** Validation accuracy: ~38.8%
- **GCN:** Validation accuracy: ~50.4%
- **GAT:** Validation accuracy: ~58.3%

The **Graph Attention Network (GAT)** model achieved the best results by leveraging attention mechanisms, though it required more computational resources and training time.

## Files in the Repository
- **`HW3_REPORT_345233563_345123624.pdf`**: Project report summarizing methods and results.
- **`final_model_training.py`**: Training script for GNN models.
- **`model_training.ipynb`**: Jupyter notebook for preprocessing, training, and evaluation.

## Authors
- **Daniel Dahan** 
- **Simon Bellilty** 

## Acknowledgments
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Colab Notebooks on GNNs](https://colab.research.google.com/)

