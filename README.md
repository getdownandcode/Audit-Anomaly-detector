# Audit Anomaly Detector

## Problem

Auditors review large transaction populations where unusual activity can be difficult to find manually. This app ranks transactions for investigation and gives a short reason for each rule or anomaly signal. A flag is a review lead, not a fraud conclusion.

## Dataset

The app uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), containing 284,807 transactions and 492 fraud labels (about 0.172%). It has time, amount, 28 PCA-anonymized features, and a fraud class. The anonymization removes business context such as merchant and category.

## Approach

Benford's Law compares the distribution of leading amount digits with the logarithmic distribution expected in many naturally occurring datasets. The chart reports mean absolute deviation (MAD) and the Nigrini conformity band; the chi-square p-value is shown as context.

Isolation Forest scores unusual combinations of log amount, hour, and available PCA features. Its normalized score contributes to a combined risk ranking.

Simple rules flag round amounts, repeated amounts within an hour, transactions from midnight to 5 a.m., and amounts above the 99th percentile. Each signal adds a readable reason to the ranked results.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

For automatic dataset download, configure Kaggle credentials using Kaggle's [API token instructions](https://www.kaggle.com/docs/api). The first run downloads the dataset into `data/creditcard.csv`, which is excluded from Git. Alternatively, download the CSV from the dataset page into that path, or upload a CSV with `Amount` and `Time` columns in the app. `Class` is optional and enables evaluation.

## Results

On the real Kaggle dataset, `prepare()` retains 282,982 rows with positive amounts, including 465 labeled frauds. Rows with zero or missing amounts are excluded from scoring. The following metrics are from `evaluate()` on those retained rows (`k=500`):

| Method | Precision@500 | Recall@500 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Combined | 0.256 | 0.2753 | 0.9148 | 0.1921 |
| Rules only | 0.004 | 0.0043 | 0.5450 | 0.0022 |
| Isolation Forest only | 0.324 | 0.3484 | 0.9510 | 0.1948 |

The combined heuristic did not outperform Isolation Forest alone on this dataset. Precision@500 and PR-AUC are more informative than accuracy with this class imbalance. The Streamlit app shows the full metrics table and the Benford chart; launch it locally to view the current app screenshot.

## Limitations

- The dataset is PCA-anonymized, so flags cannot be explained in business terms such as vendor, category, or approver.
- Fraud is about 0.17% of rows, so accuracy is not informative. Use precision@k and PR-AUC.
- Benford's Law is not proof of fraud. It can fail on fixed-range or assigned numbers and can flag ordinary variation at very large sample sizes.
- Unsupervised detection produces false positives; flagged items need human review.
- The weights and thresholds are heuristic.

## Future work

Apply the same pipeline to a real general-ledger or expense dataset with vendor and approver fields, add LLM-generated audit notes, and provide SHAP explanations.
