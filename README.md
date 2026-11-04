ThrowBot — Social Media Bot Detection & Classification

An applied machine learning project exploring behavioural patterns in social media accounts to distinguish automated accounts (bots) from human users.

Status: Work in Progress

Objective

Build a reproducible machine learning classification pipeline using publicly available labelled account data.

Dataset

Source: "Airt-ML Twitter Human Bots Dataset" (https://huggingface.co/datasets/airt-ml/twitter-human-bots)

The dataset contains approximately 37,438 labelled account records.

License: CC BY-SA 3.0.

Planned Workflow

1. Dataset exploration and statistical analysis.
2. Data cleaning and preprocessing.
3. Behavioural feature engineering.
4. Train/test data splitting.
5. Logistic Regression and Random Forest classification.
6. Evaluation using Precision, Recall, F1-score and confusion matrix.
7. Model comparison and error analysis.
8. Flask-based prediction API.

Tech Stack

- Python
- NumPy and Pandas
- Scikit-learn
- Matplotlib and Seaborn
- Flask
- Joblib

Project Structure

data/         Dataset storage
notebooks/    Exploratory data analysis
src/          ML implementation
models/       Saved models
reports/      Metrics and visualizations
app/          Flask API

Current Progress

- Dataset selected.
- Project architecture defined.
- Implementation in progress.

Limitations

The dataset represents historical social-media account behaviour. Model performance on this dataset does not establish reliability against newer or previously unseen types of automated accounts.

Future Improvements

- Behavioural feature analysis.
- Model comparison and hyperparameter tuning.
- Threshold optimization.
- Flask API integration.
- Testing on an independent dataset.

Attribution

Dataset: airt-ml/twitter-human-bots, Hugging Face.

This project is independent and is not affiliated with X or the dataset authors. 
