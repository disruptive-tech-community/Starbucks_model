import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from test_results import test_results, score

data_dir = './data/'

# Normalising the floats in the dataframe
train_data = pd.read_csv('C:\\Users\\John\\PycharmProjects\\Recommendation'
                         '-Engine\\data\\starbucks_training.csv')

df = train_data.loc[:,['V1','V2','V3','V4','V5','V6','V7']]
df_normalized = df.loc[:,['V2','V3']]
df_normalized = df_normalized.apply(lambda x:((x - min(x))/(max(x) - min(x))),axis = 0)
df.drop(labels = ['V2','V3'], axis = 1, inplace=True)
df = pd.concat([df, df_normalized], axis = 1)

df_dummy = df.loc[:,['V1','V4','V5','V6','V7']].astype('category')
df_dummy = pd.get_dummies(df_dummy)
df.drop(labels = ['V1','V4', 'V5', 'V6', 'V7'], axis = 1, inplace=True)

df_imbalanced = pd.concat([df,df_dummy], axis = 1)
X_imbalanced =  df_imbalanced
y_imbalanced = train_data.purchase

sm = SMOTE(random_state=42, n_jobs=8, sampling_strategy = 0.8)
X_resampled, y_resampled = sm.fit_sample(X_imbalanced, y_imbalanced)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2,
                                                    random_state=42)

#clf = LogisticRegression(random_state=42)
clf = RandomForestClassifier(random_state=42, n_estimators = 100, max_depth = 3)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
#print(y_pred.sum())
#print(y_test.sum())
precision_score(y_test, y_pred)


def promotion_strategy(df, clf):
    '''
    INPUT
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion_df - np.array with the values
                   'Yes' or 'No' related to whether or not an
                   individual should recieve a promotion
                   should be the length of df.shape[0]

    Ex:
    INPUT: df

    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2

    OUTPUT: promotion

    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and
    the last should not.
    '''

    # Data preprocessing
    df_normalized = df.loc[:, ['V2', 'V3']]
    df_normalized = df_normalized.apply(
        lambda x: ((x - min(x)) / (max(x) - min(x))), axis=0)
    df.drop(labels=['V2', 'V3'], axis=1, inplace=True)
    df = pd.concat([df, df_normalized], axis=1)

    df_dummy = df.loc[:, ['V1', 'V4', 'V5', 'V6', 'V7']].astype('category')
    df_dummy = pd.get_dummies(df_dummy)
    df.drop(labels=['V1', 'V4', 'V5', 'V6', 'V7'], axis=1, inplace=True)
    df_imbalanced = pd.concat([df, df_dummy], axis=1)


    # clf = LogisticRegression(random_state=42)
    promotion = clf.predict(df_imbalanced)

    return promotion


test_results(promotion_strategy, clf)