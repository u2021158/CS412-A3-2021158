# CS_412_A3
# Deep Neural Network Collaborative Filtering Recommender System

This repository contains implementations of two deep learning models for collaborative filtering recommendation systems using the MovieLens dataset:

- **Neural Collaborative Filtering (NCF)** using Multilayer Perceptrons (MLP)
- **Convolutional Neural Network based Collaborative Filtering (CNN_CF)**

---

## Project Overview

Recommender systems help provide personalized item suggestions by analyzing user preferences and past interactions. Traditional collaborative filtering methods have limitations in capturing complex, non-linear relationships. This project explores the use of deep neural networks to improve recommendation accuracy and personalization.

Two models are implemented and evaluated:

1. **NCF (MLP-based model):** Combines user and item embeddings and processes them through multiple fully connected layers to learn intricate user-item interaction patterns.

2. **CNN_CF (CNN-based model):** Uses convolutional layers to extract local and contextual interaction features from user and item embeddings treated as “2D” input.

---

## Dataset

The MovieLens-100k dataset is used, containing 100,000 ratings from ~943 users on ~1,682 movies. Ratings are binarized as implicit feedback: ratings >= 4 indicate positive interactions.

Dataset can be downloaded from [MovieLens website](https://grouplens.org/datasets/movielens/100k/).

---

## Features

- Data preprocessing with ID remapping to zero-based indices for embedding layers
- Negative sampling during training to generate implicit feedback
- Training pipeline for both NCF and CNN_CF models with binary cross-entropy loss
- Evaluation using Hit Rate (HR@10) and Normalized Discounted Cumulative Gain (NDCG@10)
- GPU support via PyTorch if available

---

## Requirements

- Python 3.7+
- PyTorch
- pandas
- scikit-learn
- tqdm (optional, for progress bars)

Install dependencies via:

```bash
pip install torch pandas scikit-learn tqdm
