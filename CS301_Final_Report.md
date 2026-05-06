# CS301 Final Project Report: Heart Disease Prediction

**Author:** Krishi Shah | **Course:** CS301 | **Date:** May 5, 2026

---

## Problem Definition and Objective

Heart disease is one of the most common and serious health conditions, and being able to predict it early can make a big difference in a patient's outcome. The goal of this project was to build a machine learning model that can predict whether a patient has heart disease based on their medical and lifestyle information. This is a binary classification problem where the model outputs either 1 (heart disease) or 0 (no heart disease). The idea is that a tool like this could help doctors quickly identify patients who might need further testing. The project was broken into three stages: exploratory data analysis and a baseline model in Stage 1, model improvement and comparison in Stage 2, and a deployed interactive web application in Stage 3.

---

## Dataset Description and Preprocessing

The dataset used for this project is called `heart_disease_dataset.csv` and contains 1,000 patient records with 16 columns. Fifteen of those columns are input features and one is the target, `Heart Disease`. The features include a mix of numeric values like Age (years), Cholesterol (mg/dL), Blood Pressure (mmHg), Heart Rate (bpm), Exercise Hours, Stress Level (1–10 scale), and Blood Sugar (mg/dL), as well as categorical ones like Gender, Smoking status, Family History, Diabetes, Obesity, Exercise Induced Angina, and Chest Pain Type. The target is evenly split at roughly 60.8% no disease and 39.2% with disease, which shows a mild class imbalance.

Before training any model, the data had to be cleaned. The `Alcohol Intake` column was dropped because 340 out of 1,000 values were missing, which is 34% of the column — too much to fill in reliably without introducing bad data. No other columns had missing values, and there were no duplicate rows in the dataset. The target column was cast to integer to make sure there were no formatting issues during training. For preprocessing, numeric features were scaled using StandardScaler so all values were on the same range, which is especially important for logistic regression. Categorical features were converted to numbers using OneHotEncoder. Both steps were wrapped inside a scikit-learn Pipeline so that no data from the test set could leak into the training process. All splits used an 80/20 ratio with stratified sampling to keep the class balance the same in both the training and test sets.

---

## Exploratory Data Analysis

Before building any models, exploratory data analysis was done to understand the data and find patterns. Several plots were generated and saved. The first was a bar chart showing the distribution of the target column, which confirmed the roughly 60/40 split between patients with and without heart disease. This was useful to see upfront so that accuracy alone would not be trusted as the only metric.

The second visualization was a set of boxplots for each of the seven numeric features, broken down by heart disease status. The most noticeable findings were that patients with heart disease tended to have higher Stress Levels and higher Blood Sugar values compared to those without. Cholesterol and Blood Pressure showed a lot of overlap between the two groups, meaning they were not as useful on their own for separating the classes. Heart Rate and Exercise Hours showed almost no difference between the two groups.

The third visualization was a set of bar charts for the categorical features showing what percentage of each category had heart disease. The most important finding here was that patients with Exercise Induced Angina had a much higher rate of heart disease than those without it. Chest Pain Type also showed clear differences, with Typical Angina patients having higher disease rates than Asymptomatic ones. Smoking and Obesity showed smaller but still visible differences. Finally, a correlation heatmap of the numeric features showed that none of them were strongly correlated with each other, which meant including all of them made sense and no features needed to be removed to avoid multicollinearity.

---

## Model Selection and Justification

Three models were trained and compared across the two modeling stages. The first was Logistic Regression, which was used as a baseline because it is simple, fast, and easy to interpret. It works well when the relationship between features and the target is roughly linear, and it gave a good starting point to compare other models against.

In Stage 2, two more complex models were added: a Decision Tree and a Random Forest. The Decision Tree was chosen because it can capture non-linear patterns and interactions between features without needing to manually engineer them. It was tuned using GridSearchCV, which tested different combinations of settings like the splitting criterion, maximum depth, and minimum samples per leaf, using 5-fold cross-validation to find the best version. The Random Forest was added because it builds many decision trees and averages their results, which typically reduces overfitting compared to a single tree. It was tuned using RandomizedSearchCV, which randomly samples from a range of hyperparameter values and is faster than testing every combination.

---

## Evaluation Metrics and Performance Comparison

All three models were tested on the same 200-sample held-out test set. The main metrics used were accuracy and F1-score. Accuracy measures how many predictions were correct overall, and F1-score is the balance between precision and recall, which matters more in a medical context where missing a real case of heart disease is costly.

Logistic Regression reached an accuracy of 86% and an F1-score of 0.81. Looking at its confusion matrix, it missed 17 patients who actually had heart disease, which means its recall on the positive class was around 78%. The tuned Decision Tree and tuned Random Forest both scored perfectly — 100% accuracy and 1.00 F1-score — with no misclassifications at all. Their 5-fold cross-validation F1-score was 0.9984, which shows this was not just a lucky split but consistent performance across folds. The best Decision Tree settings were a max depth of 3 with Gini criterion, and the best Random Forest used 100 trees with a max depth of 15.

---

## Final Model Choice with Reasoning, Strengths, Limitations, and Trade-offs

The Random Forest was selected as the final model. Even though both the Decision Tree and the Random Forest scored the same on this dataset, the Random Forest is the better long-term choice because it is less likely to overfit if applied to new, noisier data. A single tree with only three levels achieving perfect accuracy suggests the dataset is quite clean and well-structured, which may not reflect real-world conditions. The Random Forest averages predictions from 100 trees, making it more stable and reliable overall. It also produces more trustworthy feature importance scores, and the top features it identified — Exercise Induced Angina, Chest Pain Type, and Blood Sugar — line up with what doctors typically look for in cardiovascular risk assessments.

The main limitation of the Random Forest is that it is harder to explain than a decision tree. A simple three-level tree can be walked through step by step, but a forest of 100 trees is not easy to interpret directly. In a real medical application, explainability matters a lot. Another limitation is that the near-perfect accuracy here likely comes from the clean, structured nature of this educational dataset, and performance on messier real-world data would probably be lower. Despite these trade-offs, the Random Forest remains the strongest and most generalizable choice among the three models tested.

The deployed application for Stage 3 is an interactive Dash web app where users can upload any CSV file, pick a target column, select features, train a model, and get a live prediction by entering patient values into a form. The app automatically detects whether the task is classification or regression and adjusts the available models accordingly. The deployed application URL is: **[https://your-app-url.onrender.com](https://your-app-url.onrender.com)** — replace this with your live deployment link before submitting.
