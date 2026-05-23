# House Prices — Advanced Regression

Kaggle competition: [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course)

Predict residential sale prices in Ames, Iowa using 79 structural and demographic features.

**Best public score: 15,550 RMSE**

---

## What's in the notebook

| Section | What it does |
|---|---|
| EDA | Distribution of `SalePrice`, missing value audit, correlation heatmap |
| Preprocessing | Median imputation + standard scaling for numeric, most-frequent imputation + one-hot encoding for categorical |
| Model comparison | 9 models benchmarked with GridSearchCV (3-fold CV) |
| Evaluation | R² score, predicted vs. actual plot, residual plot on validation set |
| Feature importance | Top XGBoost features by split gain |
| Submission | Final model retrained on full training set, predictions exported |

## Results

| Model | Validation RMSE |
|---|---|
| XGBoost | $32,637 |
| CatBoost | $32,801 |
| Random Forest | $34,272 |
| AdaBoost | $40,577 |
| Decision Tree | $45,146 |
| KNN | $45,583 |
| Ridge | $49,883 |
| Lasso | $57,398 |
| Linear Regression | $59,394 |

XGBoost won. Retrained on all 1,460 training rows before the final submission.

## Setup

```bash
git clone https://github.com/gritzoogil/house-prices-ml
cd house-prices-ml
pip install -r requirements.txt
```

Download the data from the [Kaggle competition page](https://www.kaggle.com/competitions/home-data-for-ml-course/data) and place `train.csv` and `test.csv` in a `data/` folder.

Then run `house_prices_EDA_Training.ipynb` top to bottom. The last cell writes `submission.csv`.

## Stack

Python, scikit-learn, XGBoost, CatBoost, Pandas, Matplotlib, Seaborn
