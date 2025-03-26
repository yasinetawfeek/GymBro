import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import  RandomForestClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV
from lime import lime_tabular
from mediapipe_handler import MediaPipeHandler
from get_work_out_labels import add_workout_label_back
import seaborn as sns
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE


mediapipe_model = MediaPipeHandler()
training_dataset=mediapipe_model.read_csv_to_pd("D:\\DesD_AI_pathway\\AI\\data\\train_new.csv")
testing_dataset=mediapipe_model.read_csv_to_pd("D:\\DesD_AI_pathway\\AI\\data\\test_new.csv")

print("loading dataset....")
"""
Removes original feature and splits it into x,y,z components

"""
def Preprocess_data(dataframe,columns_to_flatten):
    final_df=dataframe.copy()
    # Expanding each column into 3 separate columns (x, y, z) and appending it to the final dataframe.
    for column in columns_to_flatten:
        # print(np.vstack(dataframe[column]).astype(float))
        expanded_df=pd.DataFrame(np.vstack(dataframe[column]).astype(float), 
                           columns=[column+'_x', column+'_y', column+'_z'],
                           index=dataframe.index)
        new_df = pd.concat([dataframe.drop(column, axis=1), expanded_df], axis=1)
        for new_column in new_df.columns:
            final_df[new_column] = new_df[new_column]

    return final_df.drop(columns=columns_to_flatten,axis=1)

"""
Splits dataset into X_train,y_train or X_test,y_test, if you give it training dataset then X_train and y_train

"""
def Return_X_y(dataframe,columns_to_delete):
    X=dataframe.drop(columns=columns_to_delete)
    y=dataframe['label']
    return X,y





# training_dataset_preprocessed=Preprocess_data(training_dataset,features_to_split)
# X_train, y_train = Return_X_y(training_dataset_preprocessed,['label','muscle group','image','Unnamed: 0'])


# testing_dataset_preprocessed=Preprocess_data(testing_dataset,features_to_split)
# X_test, y_test = Return_X_y(testing_dataset_preprocessed,['label','muscle group','image','Unnamed: 0'])


def train_model(training_dataset,testing_dataset):
    print("trained!")
    # return 92.0
    features_to_split=['left_shoulder',
       'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
       'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index',
       'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee',
       'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
       'left_foot_index', 'right_foot_index']

    training_dataset_preprocessed=Preprocess_data(training_dataset,features_to_split)
    X_train, y_train = Return_X_y(training_dataset_preprocessed,['label','muscle group','image','Unnamed: 0'])
    testing_dataset_preprocessed=Preprocess_data(testing_dataset,features_to_split)
    X_test, y_test = Return_X_y(testing_dataset_preprocessed,['label','muscle group','image','Unnamed: 0'])

    param_grid = {
    'n_estimators': [1000],
    'max_depth': [20],

    }
    random_tree_model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=random_tree_model,
        param_grid=param_grid,
        cv=3,  # 5-fold cross-validation
        n_jobs=-1,  # Use all available cores
        verbose=2,
        scoring='accuracy'
    )
    grid_search.fit(X_train,y_train)
    y_predictions=grid_search.predict(X_test)
    accuracy = accuracy_score(y_test,y_predictions)
    report = classification_report(y_test,y_predictions)
    confusion_matrix_values = confusion_matrix(y_test,y_predictions)


    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", (accuracy*100),"%")
    print("Classification Report:\n", report)

    return (accuracy*100)

# def train_model(model,X_train,y_train,X_test,y_test):
#     param_grid = {
#     'n_estimators': [1000],
#     'max_depth': [20],

#     }
#     grid_search = GridSearchCV(
#         estimator=model,
#         param_grid=param_grid,
#         cv=3,  # 5-fold cross-validation
#         n_jobs=-1,  # Use all available cores
#         verbose=2,
#         scoring='accuracy'
#     )
#     grid_search.fit(X_train,y_train)
#     y_predictions=grid_search.predict(X_test)
#     accuracy = accuracy_score(y_test,y_predictions)
#     report = classification_report(y_test,y_predictions)
#     confusion_matrix_values = confusion_matrix(y_test,y_predictions)


#     print("Best Parameters:", grid_search.best_params_)
#     print("Accuracy:", (accuracy*100),"%")
#     print("Classification Report:\n", report)

#     return (accuracy*100)






train_model(training_dataset,testing_dataset)