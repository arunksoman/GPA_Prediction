import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
DATASET_PATH = os.path.join(DATASET_DIR, 'dataset.csv')
df = pd.read_csv(DATASET_PATH)

# We do not want patient id here for analysis
df.drop('patient_id', axis=1, inplace=True)

columns = list(df.columns.values)

# One-hot encoding
one_hot = pd.get_dummies(df.patient_gender).iloc[:, :]
df = pd.concat([df, one_hot], axis=1)
# Since we one hot encoded we do not want that column anymore
df.drop('patient_gender', axis=1, inplace=True)

# Now we have to rearrange our columns to get the target as last column
df = df[['patient_age', 'female', 'male', 'patient_tbil', 'patient_dbil', 
    'patient_tc', 'patient_tp', 'patient_bun', 'patient_ua', 'patient_tg', 
    'patient_alb', 'patient_alkp', 'patient_crea', 'patient_ckmb', 
    'patient_glu', 'patient_ca', 'patient_target']]

new_columns = ['patient_age', 'female', 'male', 'patient_tbil', 'patient_dbil',
            'patient_tc', 'patient_tp', 'patient_bun', 'patient_ua', 'patient_tg',
            'patient_alb', 'patient_alkp', 'patient_crea', 'patient_ckmb',
            'patient_glu', 'patient_ca', 'patient_target']

# Training data do not contains any headers or target values.
# So convert entire dataset to numpy array
array = df.values
# As we said earlier Training data do not contains any headers or target values
# So just slice the numpy array
X = array[:,0:16]
# Then create our test data. Test data contains only target values
Y = array[:,16]
# Splitting data for training and validation
validation_size = 0.20
seed = 3
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, shuffle=True, random_state=seed)
clf_knn = KNeighborsClassifier(n_neighbors = 17, metric = 'minkowski', p = 2)

# K_Fold cross validation
def cross_validation():
    kfold = KFold(n_splits=5, random_state=seed)
    cv_results = cross_val_score(clf_knn, X_train, Y_train, cv=kfold, scoring=scoring)
    msg = "%s: %f (%f)" % ('K-Nearest Neighbour', cv_results.mean(), cv_results.std())
    print(msg)
    return cv_results.mean(), cv_results.std()

def test(test_data):
    make_2d = []
    make_2d.append(test_data)
    test_array = np.array(make_2d).astype(float)
    print("Actual Testing data: ", test_array)
    predictions = clf_knn.predict(test_array)
    print(predictions)
    return predictions[0]