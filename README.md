# 📧 Phishing Email Detection — INST414 Final Project

This project detects phishing emails using both classical machine learning models and a fine-tuned BERT transformer. The entire process—from preprocessing to evaluation—is designed to be highly interpretable and fully reproducible.

---

## 🚀 Project Goal

To classify phishing emails using multiple modeling approaches trained on the CEAS 2008 dataset:

- Logistic Regression
- Decision Tree
- Fine-tuned BERT (`bert-base-uncased`)

The project compares performance across models and demonstrates the strengths of modern transformers in detecting deceptive email content.

---

## 🗂 Project Structure

| Folder / File              | Description                                                |
|---------------------------|------------------------------------------------------------|
| `data/`                   | Cleaned and engineered CEAS dataset                        |
| `docs/`                   | Report drafts and write-up resources                       |
| `models/`                 | Trained models + tokenized BERT inputs (see Google Drive)  |
| `notebooks/`              | Data cleaning, EDA, feature engineering, modeling          |
| `reports/`                | Evaluation reports and plots                               |
| `workingproject/`         | Modular Python scripts for reproducible pipelines          |
| `requirements.txt`        | Python dependency list                                     |
| `README.md`               | This file                                                  |

---

## 📓 Notebooks Overview

| Notebook                                           | Description                                           |
|----------------------------------------------------|-------------------------------------------------------|
| `data_cleaning.ipynb`                              | Cleans and parses the CEAS 2008 dataset               |
| `Data_visualization.ipynb`                         | Explores trends and patterns in email structure       |
| `Feature_engineering.ipynb`                        | Builds features from raw data (length, domain, etc.)  |
| `modeling_notebook.ipynb`                          | Defines pipeline structure for training               |
| `sprt3_classical_models.ipynb`                     | Trains Logistic Regression and Decision Tree models   |
| `sprint3_bert_modeling_training_REFACTORED_FIXED.ipynb` | Full training, saving, and inference for BERT         |

---



## 🧠 Classical Model Results

| Model              | Accuracy | F1 Score |
|--------------------|----------|----------|
| Logistic Regression | 72%      | 76%      |
| Decision Tree       | 93%      | 94%      |

---

## 🤖 BERT Model Summary

- Model: `bert-base-uncased` from Hugging Face
- Trained for **1 epoch** on RTX 2060 GPU
- Tokenized with HuggingFace tokenizer
- Results:
  - **Accuracy**: 99%
  - **F1 Score**: 0.99
- Saved model and tokenized inputs are provided separately

---

## ⚙️ Environment Setup

```bash
# Clone the repo
git clone https://github.com/LukeSomervell/INST414-project.git
cd INST414-project

# Create a virtual environment
python -m venv .venv
.\.venv\Scriptsctivate   # On Windows

# Install requirements
pip install -r requirements.txt
```

---

## 🧪 BERT Evaluation (No Retraining Needed)

You do **not** need to re-run training.

1. **Download model + inputs** from Google Drive:  
   https://drive.google.com/drive/folders/1PJcl2rpVTLJcPMRAX_FtJ1afJ_7FfysA

2. **Place files correctly**:
   - Place the **`Bert_finetuned/`** folder inside `models/`
   - Place `tokenized_inputs.pt` and `labels.pt` directly inside `models/`

3. **Run only this notebook**:
   - `notebooks/sprint3_bert_modeling_training_REFACTORED_FIXED.ipynb`
   - This notebook will:
     - Load the pretrained model
     - Load tokenized inputs
     - Run inference in under 1 minute

---

## 📋 Requirements

Tested on Python 3.12.  
Install the following package versions for reproducibility:

- `transformers==4.39.3`
- `torch==2.2.2+cu118`
- `scikit-learn==1.6.1`
- `pandas==2.2.2`
- `numpy==1.26.4`
- `matplotlib==3.8.4`
- `seaborn==0.13.2`
- `tqdm==4.67.1`

Optional for notebook execution:
- `jupyter==1.1.1`
- `ipykernel==6.29.5`
- `ipywidgets==8.1.6`
- `notebook==7.4.1`
- `jupyterlab==4.4.1`

---

## ▶️ How to Run the Full Pipeline

1. Set up your environment (see Environment Setup section above).
2. Run the notebooks in this order:
   - `notebooks/data_cleaning.ipynb`
   - `notebooks/Feature_engineering.ipynb`
   - `notebooks/sprt3_classical_models.ipynb`
   - `notebooks/sprint3_bert_modeling_training_REFACTORED_FIXED.ipynb`
3. (Optional) For modular code:
   - Explore `workingproject/` to run each step from the command line.

---




## 👤 Author

Luke Somervell  
INST414 — Spring 2025  
University of Maryland

