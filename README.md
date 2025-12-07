# Titanic Survival Prediction – RandomForestClassifier

![Python](https://img.shields.io/badge/Python-3.11-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange) ![Dataset](https://img.shields.io/badge/Dataset-Kaggle-red)

---

## Project Overview

This project predicts passenger survival on the Titanic using **RandomForestClassifier**. The pipeline includes **data preprocessing**, **feature engineering**, **model training**, **hyperparameter tuning (GridSearchCV)**, and **evaluation**.

---

## Dataset

* **Source:** Kaggle Titanic dataset (`titanic.csv`)
* **Rows:** 891
* **Columns:** 12

**Features Used:**

| Feature    | Description                                      |
| ---------- | ------------------------------------------------ |
| Pclass     | Passenger class (1,2,3)                          |
| Sex        | Gender (0: male, 1: female)                      |
| Age        | Age in years (missing values filled with median) |
| SibSp      | Number of siblings/spouses aboard                |
| Parch      | Number of parents/children aboard                |
| Fare       | Passenger fare                                   |
| Embarked   | Port of embarkation (one-hot encoded)            |
| FamilySize | Engineered: SibSp + Parch + 1                    |
| IsAlone    | Engineered: 1 if passenger is alone, else 0      |

---

## Preprocessing & Feature Engineering

1. **Missing Values:**

   * `Age` → filled with median
   * `Fare` → filled with median (if missing)

2. **Categorical Encoding:**

   * `Sex` → 0: male, 1: female
   * `Embarked` → one-hot encoding

3. **Engineered Features:**

   * `FamilySize` = `SibSp` + `Parch` + 1
   * `IsAlone` = 1 if alone, else 0

---

## Model Training

* **Model:** RandomForestClassifier
* **Train/Test Split:** 80% / 20%
* **Scaling:** StandardScaler
* **Hyperparameter Tuning:** GridSearchCV (5-fold CV)

**GridSearchCV Best Parameters:**

```python
{
  'max_depth': 5,
  'max_features': 'sqrt',
  'min_samples_leaf': 1,
  'min_samples_split': 2,
  'n_estimators': 300
}
```

---

## Model Evaluation

* **Test Accuracy:** `~81%`

**Confusion Matrix & ROC Curve:**
![Confusion Matrix](path_to_confusion_matrix_image.png)
![ROC Curve](path_to_roc_curve_image.png)

**Feature Importance:**
![Feature Importance](path_to_feature_importance_image.png)

---

## Results & Insights

* `Sex` is the strongest predictor of survival.
* `FamilySize` and `IsAlone` features improve accuracy.
* RandomForest outperforms Logistic Regression with limited features.
* Hyperparameter tuning improved generalization, but accuracy remained ~81%.

---

## Next Steps

1. Extract **Title** from `Name` (Mr, Mrs, Miss, etc.)
2. Use **Cabin deck info**
3. Try **XGBoost / LightGBM** for potentially higher accuracy (~85–88%)
4. Apply **ensembles or stacking** to boost performance

---

## How to Run

1. Clone the repository
2. Place `train.csv` in the project folder
3. Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

4. Run the Jupyter Notebook or Python script

---

## Author

**Chinni Krishna Ronanki** – Data Science Enthusiast

