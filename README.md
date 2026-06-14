# Disaster Tweet Classification

Classifying tweets as **disaster-related** or **not** using machine learning and deep learning models. During emergencies, platforms like Twitter become essential channels for real-time communication among victims, officials, and first responders. Automatically identifying genuine disaster tweets can speed up emergency response and help save lives.

## Overview

This project trains and compares multiple models on a labeled set of tweets to determine which approach best distinguishes real disaster reports from unrelated chatter. A fine-tuned **BERT** model is benchmarked against traditional machine learning baselines.

## Repository Structure

| File / Folder | Description |
|---------------|-------------|
| `D_T_C Dataset/` | Dataset of labeled tweets used for training and evaluation |
| `Tweet_Classification_Models.ipynb` | Traditional ML models (preprocessing, vectorization, training, evaluation) |
| `tweet_classification_bert.ipynb` | Fine-tuned BERT transformer model |
| `Disaster Tweet classification.pdf` | Project report |

## Models

- **BERT** — transformer-based contextual language model, fine-tuned for binary tweet classification
- **Traditional ML baselines** — classic models trained on vectorized tweet text for comparison

## Evaluation

Models are compared using **accuracy** and **confusion matrices** to measure how reliably each one separates disaster from non-disaster tweets.

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Nithindesu/Disaster_Tweet_Classification.git
cd Disaster_Tweet_Classification

# Install common dependencies
pip install numpy pandas scikit-learn matplotlib transformers torch

# Launch the notebooks
jupyter notebook
```

Open `Tweet_Classification_Models.ipynb` for the traditional ML pipeline, or `tweet_classification_bert.ipynb` for the BERT model.

## Tech Stack

- Python
- scikit-learn
- Hugging Face Transformers (BERT)
- PyTorch
- pandas, NumPy, Matplotlib
- Jupyter Notebook
