import argparse
import os
import pandas as pd
import numpy as np
import pickle
import json

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import azureml.core
from azureml.core import Run
from azureml.core.model import Model

print("**********************************************")
print("INSIDE train.py")
print("**********************************************")

print("1. Parse arguments")
print('.............................................')
parser = argparse.ArgumentParser("train")
parser.add_argument("--model_name", type=str, help="model name", dest="model_name", required=True)
parser.add_argument("--build_number", type=str, help="build number", dest="build_number", required=True)

args = parser.parse_args()

print("Argument 1: %s" % args.model_name)
print("Argument 2: %s" % args.build_number)
print('')
print('')

print("2. Read training data from remote storage")
print('.............................................')
# Pandas dataframe
df = pd.read_csv('https://mlopssa.blob.core.windows.net/chd-dataset/framingham.csv')
print('')
print('')

print("3. Cleanse/transform data")
print('.............................................')
# create a boolean array of smokers
smoke = (df['currentSmoker']==1)
# Apply mean to NaNs in cigsPerDay but using a set of smokers only
df.loc[smoke,'cigsPerDay'] = df.loc[smoke,'cigsPerDay'].fillna(df.loc[smoke,'cigsPerDay'].mean())

# Fill out missing values
df['BPMeds'].fillna(0, inplace = True)
df['glucose'].fillna(df.glucose.mean(), inplace = True)
df['totChol'].fillna(df.totChol.mean(), inplace = True)
df['education'].fillna(1, inplace = True)
df['BMI'].fillna(df.BMI.mean(), inplace = True)
df['heartRate'].fillna(df.heartRate.mean(), inplace = True)
print('..3. completed')
print('')
print('')

print("4. Train model")
print('.............................................')
# Features and label
features = df.iloc[:,:-1]
result = df.iloc[:,-1] # the last column is what we are about to forecast

# Train & Test split
X_train, X_test, y_train, y_test = train_test_split(features, result, test_size = 0.2, random_state = 14)

# RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.12
sfm = SelectFromModel(clf, threshold=0.12)

# Train the selector
sfm.fit(X_train, y_train)

# Features selected
featureNames = list(features.columns.values) # creating a list with features' names
print("Feature names:")
for featureNameListindex in sfm.get_support(indices=True):
    print(featureNames[featureNameListindex])

# Feature importance
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# With only imporant features. Can check X_important_train.shape[1]
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

clfModel = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
clfModel.fit(X_important_train, y_train)
print("..4. completed")
print('')
print('')

print("5. Save model to disk")
print('.............................................')
os.makedirs('./outputs', exist_ok=True)
modelFilename = './outputs/chd-rf-model'
pickle.dump(clfModel, open(modelFilename, 'wb'))
print("..5. completed")

print("6. Register model into model registry")
print('.............................................')
os.chdir("./outputs")
run = Run.get_context()

# Register model to Azure ML Model Registry
modelDescription = 'Model to predict coronary heart disease'
model = Model.register(
    model_path='chd-rf-model',  # this points to a local file
    model_name=args.model_name,  # this is the name the model is registered as
    tags={"type": "classification", "run_id": run.id, "build_number": args.build_number},
    description=modelDescription,
    workspace=run.experiment.workspace
)
os.chdir("..")

print("Model registered: {} \nModel Description: {} \nModel Version: {}".format(model.name, 
                                                                                model.description, model.version))

print("..6. completed")

print("*********************************************")
print("EXITING train.py")
print("*********************************************")



