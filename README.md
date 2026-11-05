# SocialSentinel-AI — Social Media Bot Detection & Classification

An applied machine learning project exploring behavioural patterns in social media accounts to distinguish automated accounts (bots) from human users.

**Status:** Work in Progress

---

## Objective

Build a reproducible machine learning classification pipeline using publicly available labelled account data.

---

## Dataset

- **Source:** [Cresci-2017 Dataset (Kaggle)](https://www.kaggle.com/datasets/hashemalsalmi/cresci-2017)
- The dataset contains approximately 37,438 labelled account records.
- **License:** CC BY-SA 3.0.

---

## Planned Workflow

1. Dataset exploration and statistical analysis.
2. Data cleaning and preprocessing.
3. Behavioural feature engineering.
4. Train/test data splitting.
5. Logistic Regression and Random Forest classification.
6. Evaluation using Precision, Recall, F1-score and confusion matrix.
7. Model comparison and error analysis.
8. Flask-based prediction API.

---

## Tech Stack

- Python
- NumPy & Pandas
- Scikit-learn
- Matplotlib & Seaborn
- Flask
- Joblib

---

## Project Structure

```
SocialSentinel-AI/
│
├── data/
│   ├── raw/
│   │   └── cresci2017/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   └── 01_dataset_inspection.ipynb
│
├── docs/
│   └── dataset_inventory.md
│
├── src/
│   └── data/
│
├── .gitignore
└── README.md
```

---

## Current Progress

- [x] Project architecture and folder structure defined.
- [x] Dataset inventory documentation created.
- [x] Initial EDA Jupyter notebook (`notebooks/01_dataset_inspection.ipynb`) established.
- [ ] Raw Cresci-2017 data loading and cleaning.
- [ ] Model training and benchmarking.

---

## Limitations

The dataset represents historical social-media account behaviour. Model performance on this dataset does not establish reliability against newer or previously unseen types of automated accounts.

---

## Future Improvements

- Behavioural feature analysis.
- Model comparison and hyperparameter tuning.
- Threshold optimization.
- Flask API integration.
- Testing on an independent dataset.

---

## Attribution

Dataset: `airt-ml/twitter-human-bots`, Hugging Face / Cresci et al. 2017.

*This project is independent and is not affiliated with X or the dataset authors.*
