# Insurence Premium Prediction

## Problem Statement:
The goal of this project is to give people an estimate of how much they need based on
their individual health situation. After that, customers can work with any health 
insurance carrier and its plans and perks while keeping the projected cost from our 
study in mind. This can assist a person in concentrating on the health side of an 
insurance policy rather han the ineffective part.

## Approach
**1. Data Collection :**

The data set was provide as a part of Kaggle competition and can be downloaded from [here](https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv)

**2. Exploratory Data Analysis:** 
Explore the dataset to gain insights into its structure, quality, and characteristics.
Identify missing values, outliers, and potential data quality issues.
Visualize data distributions, relationships, and patterns.

**3. Data Preprocessing:**
### Data Cleaning:
Handle missing values, outliers, and anomalies identified during EDA.
Correct any errors or inconsistencies in the dataset.

### Feature Engineering:
Create new features or modify existing ones to better represent the underlying patterns in the data.
Handle categorical variables through encoding or other techniques.

### Scaling/Normalization:
Scale numerical features if necessary to ensure all features contribute equally to model training.

**4. Model Building:**
I have train the model using diffrent types of Machine Learning algoriths such as Linear Regresson, Lasso , Ridge, RandomForest Regressor, Adaboost Regressor, GradientBoost Regressor, 

**5. Model Evaluation**
After training on a dedicated set, the model's effectiveness on unseen data is measured using specific metrics (e.g., accuracy, precision, recall). This ensures the model's ability to generalize and make accurate predictions in real-world scenarios. 

Below table show the model performance:

| **Algorithm** | **RMSE** | **MAE** | **R2 Score** |
|-----|-----|-----|-----|
| Linear Regression | 6085.568 | 4189.94 | 75.991 |
| Lasso | 6085.412 | 4190.698 | 75.9925 |
| Ridge | 6085.487 | 4190.878 | 75.9919 |
| Elastic Net | 6596.508 | 4639.68 | 71.790 |
| Random Forest Regressor | 4758.80 | 2583.506 | 85.31 |
| Adaboost Regressor | 5048.66 | 3937.046 | 83.47 |
| Gradient Boost Regressor | 5882.527 | 3339.62 | 77.56 |

## Tools and Technologies used
 ![Tools and Technologies](https://github.com/vikashdwivedi21/Insurance-Premium-Prediction/blob/master/images/Technologies%20used.png)

 ## Home Page
![Home](https://github.com/vikashdwivedi21/Insurance-Premium-Prediction/blob/master/images/UI/Home.png)

## Prediction Page
![Prediction](https://github.com/vikashdwivedi21/Insurance-Premium-Prediction/blob/master/images/UI/Prediction.png)

## Result
![Result](https://github.com/vikashdwivedi21/Insurance-Premium-Prediction/blob/master/images/UI/Result.png)

# How to Run?
 
 ### STEPS:

Clone the repository

```bash
Project repo: https://github.com/vikashdwivedi21/Insurance-Premium-Prediction.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -p env python==3.9 -y
```

```bash
conda activate env/
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### Step 03- Run application.py
```bash
python application.py
```

## Author : Vikash Dwivedi
LinkedIn : ![linkedIn](linkedin.com/in/vikash-dwivedi)