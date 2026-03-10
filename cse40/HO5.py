import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay

import xgboost as xgb
from xgboost import XGBClassifier

def clean_col_00(df):

    df['col_00'] = df['col_00'].apply(lambda x: x.split()[0]).astype(int)

    return df

def clean_col_01(df):

    df['col_01'] = df['col_01'].astype(float)

    return df

def clean_col_02(df):

    col_2 = df['col_02']
    
    col_2 = col_2.apply(lambda x: [major.strip() for major in x.split(',')])
    
    exploded = col_2.explode()

    col_2_one_hot_encoded = (
        pd.get_dummies(exploded)
        .groupby(exploded.index)
        .sum()
    )

    df.drop(columns=['col_02'], inplace=True)

    df = df.join(col_2_one_hot_encoded)

    return df

def clean_col_03(df):

    col_3 = df['col_03']
    
    col_3 = col_3.apply(lambda x: [major.strip() for major in x.split(',')])
    
    exploded = col_3.explode()

    col_3_one_hot_encoded = (
        pd.get_dummies(exploded)
        .groupby(exploded.index)
        .sum()
    )

    df.drop(columns=['col_03'], inplace=True)

    df = df.join(col_3_one_hot_encoded)

    return df

def clean_col_04(df):

    df['col_04'] = df['col_04'].astype(float)

    return df

def clean_col_05(df):

    df['col_05'] = df['col_05'].astype(int)

    return df

def clean_col_07(df):

    col_07 = df['col_07']

    exploded = col_07.explode()

    col_07_one_hot_encoded = (
        pd.get_dummies(exploded)
        .groupby(exploded.index)
        .sum()
    )

    df.drop(columns=['col_07'], inplace=True)

    df = df.join(col_07_one_hot_encoded)

    return df

def clean_col_08(df):

    df['col_08'] = df['col_08'].astype(int)

    return df

def clean_col_09(df):

    df['col_09'] = df['col_09'].astype(int)

    return df

def train_KNN_model(df):

    X = df.drop(columns=['label'], axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    neigh = KNeighborsClassifier(n_neighbors=9)
    neigh.fit(X_train, y_train)

    return neigh

def train_logistical_regression_model(df):

    X = df.drop(columns=['label'], axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":

    df  = pd.read_csv("data.txt", sep='\t')

    df = df.map(lambda x: x.strip().lower() if isinstance(x, str) else x)
    df.replace(["nan", "n/a", pd.NA, '?'], np.nan, inplace=True)
    df.dropna(axis=0, inplace=True, ignore_index=True)

    df = clean_col_00(df)
    df = clean_col_01(df)
    df = clean_col_02(df)
    df = clean_col_03(df)
    df = clean_col_04(df)
    df = clean_col_05(df)
    # column 6 is fine
    df = clean_col_07(df)
    df = clean_col_08(df)
    df = clean_col_09(df)


    train_KNN_model(df=df)

    