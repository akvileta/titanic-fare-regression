# titanic-fare-regression
A simple linear regression project predicting passenger Fare from the Kaggle Titanic dataset, built as a hands-on exercise in the full ML workflow: EDA, cleaning, encoding, model fitting, and evaluation.

Dataset
Uses train.csv and test.csv from the Kaggle Titanic competition. Download them from Kaggle and place them in the project folder before running.
Link to the files: https://www.kaggle.com/competitions/titanic/data

Project Steps
1. Explore the data: Checked data types, missing values, and distributions (histograms) for numeric columns.
2. Check relationships: Built a correlation matrix and heatmap to identify which variables relate to Fare. Pclass, Embarked, SibSp, Parch showed the strongest relationship.
3. Clean the data:
  3. 1. Filled missing Age with median, Embarked with mode
  3. 2. Dropped Cabin (too many missing values), plus PassengerId, Name, Ticket
  3. 3. Capped extreme Fare outliers at the 99th percentile
  3. 4. Handled missing/zero Fare values in the test set using training-set statistics (avoiding data leakage)
  3. 5. Encode categorical variables

4. Fit the model: Trained a LinearRegression model, first using Pclass alone, then expanded with SibSp, Parch, and Embarked.
5. Evaluate performance: Measured RMSE and R² on the held-out test set.
6. Visualize results:  Predicted vs. actual scatter plot, residual plot.

Results
RMSE,	R²
Pclass only	39.82	0.41
+ SibSp, Parch, Embarked	36.88	0.49

Adding family size and embarkation port improved the model, but ~51% of the variation in Fare remains unexplained — likely due to non-linear effects or missing signal (e.g., from Cabin or Age). A log-transformed target is a natural next experiment.

Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
Next Steps
Try log-transforming Fare to address skew
Add Age back in as a feature
Explore non-linear models (e.g., decision trees) for comparison
