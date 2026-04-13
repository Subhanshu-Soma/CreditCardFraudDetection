# CreditCardFraudDetection
A project that determines credit card fraud from a given dataset and creates visuals depicting it

Place the Kaggle `creditcard.csv` file in the `data/` folder

Run:

```bash
python credit_card_fraud_detection/src/train_fraud_model.py
```

What it does:
- Loads and validates the dataset
- Removes outliers from non-fraud examples only
- Scales `Amount` and `Time`
- Handles class imbalance with SMOTE
- Trains a Random Forest classifier
- Evaluates ROC-AUC, confusion matrix, and classification metrics
- Saves plots and the trained model

Outputs:
- `outputs/fraud_confusion_matrix.png`
- `outputs/fraud_roc_curve.png`
- `outputs/fraud_feature_importance.png`
- `models/fraud_random_forest.joblib`
