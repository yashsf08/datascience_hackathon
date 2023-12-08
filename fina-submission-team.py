import pickle as pkl
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

from catboost import CatBoostClassifier



# generate a random dataset with 7500 records and 7 features
# X, y = make_classification(n_samples=7500, n_features=7, random_state=42)

# define the parameter grid to search over
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

scoring = ['accuracy', 'roc_auc']

# create a random forest classifier
rfc = RandomForestClassifier()

# perform a grid search with cross-validation
grid_search = GridSearchCV(rfc, param_grid=param_grid, scoring=scoring, cv=5, refit='accuracy', return_train_score=True)
grid_search.fit(X_train, y_train)


# print the best parameters and score
print("Best parameters: ", grid_search.best_params_)




# Dump File

# Separate the ID column and the target column
ids = train_data['id']
y_train = train_data['target']

# Drop the ID and target columns from the training data
X_train = train_data.drop(['id', 'target'], axis=1)

# Train a Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict the target values for the training data
y_pred = clf.predict(X_train)

# Combine the ID and predicted values into a DataFrame
output_df = pd.DataFrame({'id': ids, 'predicted_target': y_pred})

# Dump the DataFrame into a CSV file
output_df.to_csv('output.csv', index=False)



def dump_data(ids, y_dump):
    output_df = pd.DataFrame({'id': ids, 'predicted_target': y_pred})
    output_df.to_csv('output.csv', index=False)
    return


if __name__ == "__main__":
    dump_data(ids, y_dump)
    
    
