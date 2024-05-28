#!pip install lazypredict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sklearn libraries
from sklearn import datasets 
from sklearn import metrics 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesRegressor
from sklearn.metrics import mean_squared_error,r2_score
#GRIDCV is used for hyperparameter tuning.

#print common info
def df_check(dataframe):
  print('- INFO - ', '\n')
  print(dataframe.info(), '\n')
  print('- SHAPE - ', '\n')
  print(dataframe.shape, '\n')
  print('- DESCRIBE - ', '\n')
  print(dataframe.describe(), '\n')
  print('- DESCRIBE O - ', '\n')
  print(dataframe.describe(include='object'), '\n')  
  print('- NULL - ', '\n')
  print(dataframe.isnull().sum(), '\n')
  print('- DUPLICATED - ', '\n')
  print(dataframe.duplicated().sum())
  print('- UNIQUE - ', '\n')
  print(dataframe.nunique())


#Common Process

#1. Data Loading and Initial Exploration
df =pd.read_csv("filename.csv")

#2 Data Visualization
sns.scatterplot(data=df, x='Rating', y='Price')
plt.title('Rating vs Price')
plt.show()

#3 Data Preprocessing
   #3.1 dropping unwated/unused columns
df.drop(columns=['Unnamed: 0','Name'],inplace=True)
   #3.2 Fill the null values with Mode - most frequent value
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

    #3.3 Encoding the categorical values.
        #     ordinal categorical features:
        # 	it has the order
        # 		Education Level: High School, Bachelor's Degree, Master's Degree, Ph.D.
        # 		Customer Satisfaction: Very Unsatisfied, Unsatisfied, Neutral, Satisfied, Very Satisfied
        # 		Size: Small, Medium, Large, Extra Large
        # 		Rank: Bronze, Silver, Gold, Platinum
        # One-Hot Encoding: For nominal categorical features.
        # 	Nominal Categorical features:
        # 	 no order 
        # 		Color: Red, Blue, Green, Yellow
        # 		Country: USA, Canada, Mexico, UK
        # 		Brand: Apple, Samsung, Nokia, Sony
        # 		Type of Vehicle: Car, Truck, Motorcycle, Bicycle
        # Target Encoding: For high cardinality categorical features.
        # 		High Cardinality
        # 			High cardinality features are categorical variables that have a large number of unique categories or levels. This can pose challenges in data processing and model building
le = LabelEncoder()
categorical_columns = ['Camera', 'Processor_name', 'Screen_resolution', 'Display', 'Battery', 'External_Memory']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

#4 Feature Engineering
df.drop_duplicates(inplace=True) # Remove duplicate rows if any
#4.1 one hot encoding which gives the dummy column name with  value try or false
dummie_df = pd.get_dummies(df, drop_first=True) # Create dummy variables for categorical features

#4.2 Create corelation map to identify which feature is most usefull for the prediction
plt.figure(figsize=(12,15))
sns.heatmap(df.corr())
plt.show()

#5 Model Training and Evaluation
X = dummie_df.drop(columns='Price')
y = dummie_df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Determine feature importance using Random Forest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Plot feature importance
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importances')
plt.show()

#Use lazy regressor to use multiple regression models and find the best suiting model
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
models # Display the performance of the models

#Model Tuning
# Replace whitespace in feature names
X_train.columns = [col.replace(" ", "_") for col in X_train.columns]
X_test.columns = [col.replace(" ", "_") for col in X_test.columns]

param_grid = {
    'num_leaves': [31, 50, 70, 90],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [20, 40, 60, 80]
}

# GridSearchCV to find the best parameters
lgbm = LGBMRegressor(verbose=-1)  
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)


print("Best parameters found: ", grid_search.best_params_)

#Finding the accuracy calculation methods.
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R-Squared: {r2}')

#Initialize ExtraTreesRegressor
etr = ExtraTreesRegressor(random_state=42)

# Calculate cross-validation scores
cv_scores = cross_val_score(etr, X, y, cv=5, scoring='r2')  # You can change scoring to 'neg_mean_squared_error' for MSE

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
accuracy = accuracy_score(y_test_binned, y_pred_class)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_binned, y_pred_class, average='weighted')
conf_matrix = confusion_matrix(y_test_binned, y_pred_class)

# Print classification metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")
# Plot ROC curve (for each class)
fpr = {}
tpr = {}
roc_auc = {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_binned == i, etc.predict_proba(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

#Plotting the final results.
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()

