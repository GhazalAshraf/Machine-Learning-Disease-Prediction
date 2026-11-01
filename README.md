# Machine Learning Disease Prediction

A binary classification project that predicts whether a patient has a disease from their health measurements, lifestyle habits and demographic information. The notebook covers the full workflow: merging the raw data sources, exploratory analysis, feature engineering, outlier handling, tuning eight classifiers and comparing them with repeated k-fold cross-validation.

## Dataset

The data is split into a training set (800 patients, labelled) and a test set (225 patients, unlabelled). Each set comes as three Excel files that are joined on `PatientID`:

| File | Columns |
|------|---------|
| `*_health.xlsx` | Height, Weight, High_Cholesterol, Blood_Pressure, Mental_Health, Physical_Health, Checkup, Diabetes |
| `*_habits.xlsx` | Smoking_Habit, Drinking_Habit, Exercise, Fruit_Habit, Water_Habit |
| `*_demo.xlsx` | Name, Birth_Year, Region, Education, Disease (target, training set only) |

The target `Disease` is roughly balanced: about 51% of training patients are positive.

## Workflow

1. **Data loading**: merge the health, habits and demographic tables for both train and test, then stack them so that preprocessing is applied consistently.
2. **Exploratory analysis**: summary statistics, unique-value counts, histograms and box plots of the numeric variables.
3. **Feature engineering**
   - `sex` derived from the name title (`Mrs.`)
   - `age` derived from `Birth_Year`
   - Ordinal encoding of the survey answers: Drinking_Habit, Water_Habit, Fruit_Habit, Diabetes, Checkup
   - Binary encoding of Smoking_Habit and Exercise
   - Feature selection with Random Forest feature importance, which keeps the health, habit and demographic features and removes Region and Education
4. **Outlier treatment**: values outside 1.5 × IQR are capped for age, High_Cholesterol, Height, Weight and Blood_Pressure.
5. **Correlation check**: a correlation heatmap of all features (the highest correlation is between sex and height, around 0.6).
6. **Modelling**: each model is tuned with single-parameter sweeps followed by `GridSearchCV`, then evaluated with `RepeatedStratifiedKFold` (10 splits × 3 repeats). F1 is the primary metric. Features are standardised for the SVM.

The most important features, according to Random Forest, are fruit consumption, diabetes status, physical health, time since last checkup and mental health.

## Results

Mean cross-validated scores for each tuned model:

| Model | Best parameters | F1 | Accuracy | Recall | Precision |
|-------|-----------------|----|----------|--------|-----------|
| CatBoost | depth=8, iterations=250, learning_rate=0.2 | **0.989** | **0.989** | 0.989 | **0.990** |
| Gradient Boosting | max_depth=6, n_estimators=500, learning_rate=0.1 | 0.988 | 0.987 | **0.990** | 0.985 |
| XGBoost | max_depth=5, n_estimators=550, learning_rate=0.5 | 0.985 | 0.985 | 0.983 | 0.987 |
| Random Forest | n_estimators=100, max_depth=None | 0.979 | 0.978 | 0.978 | 0.979 |
| Decision Tree | criterion=entropy, max_depth=12 | 0.970 | 0.970 | 0.961 | 0.981 |
| SVM | kernel=rbf, C=10 (scaled features) | 0.961 | 0.960 | 0.959 | 0.964 |
| MLP | hidden_layer_sizes=78, activation=tanh, learning_rate_init=0.001 | 0.858 | 0.852 | 0.873 | 0.847 |
| Logistic Regression | penalty=l2, solver=lbfgs | 0.836 | 0.828 | 0.855 | 0.822 |

The tree-based ensembles achieve the highest scores, with F1 between 0.979 and 0.989. The final predictions on the test set are made with a Random Forest classifier.

## Getting started

### Requirements

- Python 3.10 (the version the notebook was last run with)
- The packages in [`requirements.txt`](requirements.txt), pinned to the versions from the last Colab run

```bash
pip install -r requirements.txt
```

### Running on Google Colab

1. Upload the six data files (`train_health.xlsx`, `train_habits.xlsx`, `train_demo.xlsx`, `test_health.xlsx`, `test_habits.xlsx`, `test_demo.xlsx`) to `MyDrive/git1/` in your Google Drive.
2. Open [`Machine Learning Project.ipynb`](Machine%20Learning%20Project.ipynb) in Colab and run all cells. The notebook mounts your Drive in its first cell.

### Running locally

1. Remove or skip the `drive.mount(...)` cell.
2. Change the `pd.read_excel(...)` paths to point at your local copies of the data files.
3. Install the requirements above and run the notebook with Jupyter.

The model comparison step re-runs cross-validation for all eight models and takes around 10 minutes.

## Repository structure

```
.
├── Machine Learning Project.ipynb   # Full analysis and modelling notebook
├── requirements.txt                 # Python dependencies
└── README.md
```

## Tech stack

Python · pandas · NumPy · scikit-learn · CatBoost · XGBoost · Matplotlib · seaborn
