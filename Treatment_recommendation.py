# Model for Treatment Recommendation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef

# Lodading dataset 
df = pd.read_excel('/workspace/HCC/0_All_rev.xlsx', 
                   sheet_name='All', engine='openpyxl')

df = df.fillna(0)

for i in df.columns[1:-1]:
    for j in range(len(df)):
        df.loc[j,i] = float(df.loc[j,i])

df_KU = df[df['Center'] == 1]
df_BH = df[df['Center'] == 2]
df_SM = df[df['Center'] == 3]
df_SN = df[df['Center'] == 4]
df_CM = df[df['Center'] == 5]
df_SV = df[df['Center'] == 6]
df_AM = df[df['Center'] == 7]
df_CA = df[df['Center'] == 8]
df_IH = df[df['Center'] == 9]


# 1. Model training and validation
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 999)

X = np.array(df_AM.loc[:, 'age':'Meta'])
y = np.array(df_AM.loc[:, 'Tx'])

# Model (Voting classifier)
lr = LogisticRegression(random_state=999, multi_class='multinomial')
rf = RandomForestClassifier(random_state = 999)
lgbm = LGBMClassifier(random_state=999)
sv = svm.SVC(random_state=999, probability=True)
mlp = MLPClassifier(random_state=999, max_iter=300)
eclf_A = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('lgbm', lgbm), ('svm', sv), ('mlp', mlp)], voting='soft')

# 1.1. Internal validation with 5-fold cross validation
df = pd.DataFrame()

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # preprocessing for internal datasets
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # model training
    eclf.fit(X_train, y_train)
    
    # 1st result prediction
    y_pred = eclf.predict(X_test)

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, y_pred)
    df.loc[i, 'Recall_macro'] = recall_score(y_test, y_pred, average='macro')
    df.loc[i, 'Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    df.loc[i, 'F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    df.loc[i, 'Kappa'] = cohen_kappa_score(y_test, y_pred)
    df.loc[i, 'MCC'] = matthews_corrcoef(y_test, y_pred)
    
    # 2nd prediction
    test_results = eclf.transform(X_test)

    results = []
    pred_1 = []
    pred_2 = []

    for result in test_results:
        result = result.reshape(5,6)
        result = np.sum(result, axis=0)
        results.append(np.argsort(result))
        pred_1.append(np.argsort(result)[-1])
        pred_2.append(np.argsort(result)[-2])

    # 2nd result summation
    pred_total = pred_1.copy()

    for j, real in enumerate(y_test):
        if pred_1[j] == real:
            pass
        else:
            if pred_2[j] == real:
                pred_total[j] = pred_2[j] 

    y_pred = pred_total

    df.loc[i, '2_Accuracy'] = accuracy_score(y_test, y_pred)
    df.loc[i, '2_Recall_macro'] = recall_score(y_test, y_pred, average='macro')
    df.loc[i, '2_Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    df.loc[i, '2_F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    df.loc[i, '2_Kappa'] = cohen_kappa_score(y_test, y_pred)
    df.loc[i, '2_MCC'] = matthews_corrcoef(y_test, y_pred)
    
for col in df.columns:    
    df.loc['Mean', col] = df[col].mean()
    df.loc['SD', col] = df[col].std()
    
df_in = df.copy()

# 1.2. External validation after training with whole internal dataset
# preprocessing for external validation datasets
scaler = MinMaxScaler().fit(X)
_X = scaler.transform(X)

df_KU.loc[:, 'age':'Meta'] = scaler.transform(df_KU.loc[:, 'age':'Meta'])
df_BH.loc[:, 'age':'Meta'] = scaler.transform(df_BH.loc[:, 'age':'Meta'])
df_SM.loc[:, 'age':'Meta'] = scaler.transform(df_SM.loc[:, 'age':'Meta'])
df_SN.loc[:, 'age':'Meta'] = scaler.transform(df_SN.loc[:, 'age':'Meta'])
df_CM.loc[:, 'age':'Meta'] = scaler.transform(df_CM.loc[:, 'age':'Meta'])
df_SV.loc[:, 'age':'Meta'] = scaler.transform(df_SV.loc[:, 'age':'Meta'])
df_CA.loc[:, 'age':'Meta'] = scaler.transform(df_CA.loc[:, 'age':'Meta'])
df_IH.loc[:, 'age':'Meta'] = scaler.transform(df_IH.loc[:, 'age':'Meta'])

# model training
eclf.fit(_X, y)

# model testing: external validation
centers = ["KU", "BH", "SM", "SN", "CM", "SV", "CA", "IH"]
mod = sys.modules[__name__]

df = pd.DataFrame()

for i, center in enumerate(centers):
    test_data = getattr(mod, f"df_{center}")

    X_test = np.array(test_data.loc[:, 'age':'Meta'])
    y_test = np.array(test_data.loc[:, 'Tx'])
    
    # 1st result prediction
    y_pred = eclf.predict(X_test)

    df.loc[i, 'Center'] = center
    df.loc[i, 'Accuracy'] = accuracy_score(y_test, y_pred)
    df.loc[i, 'Recall_macro'] = recall_score(y_test, y_pred, average='macro')
    df.loc[i, 'Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    df.loc[i, 'F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    df.loc[i, 'Kappa'] = cohen_kappa_score(y_test, y_pred)
    df.loc[i, 'MCC'] = matthews_corrcoef(y_test, y_pred)

    # 2nd result summation
    test_results = eclf.transform(X_test)

    results = []
    pred_1 = []
    pred_2 = []

    for result in test_results:
        result = result.reshape(5,6)
        result = np.sum(result, axis=0)
        results.append(np.argsort(result))
        pred_1.append(np.argsort(result)[-1])
        pred_2.append(np.argsort(result)[-2])

    pred_total = pred_1.copy()

    for j, real in enumerate(y_test):
        if pred_1[j] == real:
            pass
        else:
            if pred_2[j] == real:
                pred_total[j] = pred_2[j] 

    y_pred = pred_total

    df.loc[i, '2_Accuracy'] = accuracy_score(y_test, y_pred)
    df.loc[i, '2_Recall_macro'] = recall_score(y_test, y_pred, average='macro')
    df.loc[i, '2_Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    df.loc[i, '2_F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    df.loc[i, '2_Kappa'] = cohen_kappa_score(y_test, y_pred)
    df.loc[i, '2_MCC'] = matthews_corrcoef(y_test, y_pred)    
    
for col in df.columns[1:]:
    df.loc['Mean', col] = df[col].mean()
    df.loc['SD', col] = df[col].std()
    
df_ex = df.copy()

# 1.3. Training and validation with individual dataset 
df_all = pd.DataFrame()

for n, center in enumerate(centers):
    test_data = getattr(mod, f"df_{center}")

    X = np.array(test_data.loc[:, 'age':'Meta'])
    y = np.array(test_data.loc[:, 'Tx'])
    df = pd.DataFrame()
    
    # model training with individual dataset with 5-fold CV
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):

        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # preprocessing 
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # model training
        eclf.fit(X_train, y_train)

        # 1st result prediction
        y_pred = eclf.predict(X_test)

        df.loc[i, 'Accuracy'] = accuracy_score(y_test, y_pred)
        df.loc[i, 'Recall_macro'] = recall_score(y_test, y_pred, average='macro')
        df.loc[i, 'Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
        df.loc[i, 'F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
        df.loc[i, 'Kappa'] = cohen_kappa_score(y_test, y_pred)
        df.loc[i, 'MCC'] = matthews_corrcoef(y_test, y_pred)

        # 2nd prediction
        test_results = eclf.transform(X_test)

        results = []
        pred_1 = []
        pred_2 = []

        for result in test_results:
            result = result.reshape(5,6)
            result = np.sum(result, axis=0)
            results.append(np.argsort(result))
            pred_1.append(np.argsort(result)[-1])
            pred_2.append(np.argsort(result)[-2])

        # 2nd result summation
        pred_total = pred_1.copy()

        for j, real in enumerate(y_test):
            if pred_1[j] == real:
                pass
            else:
                if pred_2[j] == real:
                    pred_total[j] = pred_2[j] 

        y_pred = pred_total

        df.loc[i, '2_Accuracy'] = accuracy_score(y_test, y_pred)
        df.loc[i, '2_Recall_macro'] = recall_score(y_test, y_pred, average='macro')
        df.loc[i, '2_Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
        df.loc[i, '2_F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
        df.loc[i, '2_Kappa'] = cohen_kappa_score(y_test, y_pred)
        df.loc[i, '2_MCC'] = matthews_corrcoef(y_test, y_pred)

    for col in df.columns:    
        df.loc['Mean', col] = df[col].mean()
        df.loc['SD', col] = df[col].std()
        
    df_all = pd.concat([df_all, df])


# 2. Top classifiers sorted by accuracy for internal dataset

# ML classifiers
lr = LogisticRegression(random_state=999) 
dt = DecisionTreeClassifier(random_state=999) 
et = ExtraTreesClassifier(random_state=999) 
rf = RandomForestClassifier(random_state = 999) 
ada = AdaBoostClassifier(random_state=999)
gbc = GradientBoostingClassifier(random_state=999)
hgbc = HistGradientBoostingClassifier(random_state=999)
xgboost = xgb.XGBClassifier(random_state=999, verbosity=0) 
lightgbm = LGBMClassifier(random_state=999, verbosity=-1, force_row_wise=True) 
catboost = CatBoostClassifier(random_state=999, verbose=0)
nb = GaussianNB() 
nbb = BernoulliNB()
gpc = GaussianProcessClassifier(kernel=1.0*RBF(1.0), random_state=999) 
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
linearsvm = svm.SVC(random_state=999, kernel='linear', probability=True)
mlp = MLPClassifier(random_state=999) 
knn = KNeighborsClassifier(n_neighbors=6)
km = KMeans(n_clusters=6, random_state=999)

clfs = [lr, knn, nb, dt, linearsvm, gpc, mlp, rf, qda, ada, gbc,
       lda, et, xgboost, lightgbm, catboost, hgbc, nbb, km]

clf_name = ['lr', 'knn', 'nb', 'dt', 'linearsvm', 'gpc', 'mlp', 
            'rf', 'qda', 'ada', 'gbc', 'lda', 'et', 'xgboost', 'lightgbm', 'catboost', 
            'hgbc', 'nbb', 'km']

df_all = pd.DataFrame()

for i, clf in enumerate(tqdm(clfs)):
    
    df = pd.DataFrame()

    for f, (train_index, test_index) in enumerate(skf.split(X, y)):

        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        # model training with each classifier for internal dataset with 5-fold CV
        clf.fit(X_train, y_train)
        
        # model testing 
        y_pred = clf.predict(X_test)

        df.loc[f, 'Classifier'] = clf_name[i]
        df.loc[f, 'Accuracy'] = accuracy_score(y_test, y_pred)
        df.loc[f, 'Recall_macro'] = recall_score(y_test, y_pred, average='macro')
        df.loc[f, 'Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
        df.loc[f, 'F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
        df.loc[f, 'Kappa'] = cohen_kappa_score(y_test, y_pred)
        df.loc[f, 'MCC'] = matthews_corrcoef(y_test, y_pred)

    for col in df.columns[1:]:    
        df.loc['Mean', col] = df[col].mean()
        df.loc['SD', col] = df[col].std()
        
    df_all = pd.concat([df_all, df])

# 3. Various number of classifiers composing voting classifier

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 999)

X = np.array(df_AM.loc[:, 'age':'Meta'])
y = np.array(df_AM.loc[:, 'Tx'])

# Top 7 classifiers
linearsvm = svm.SVC(random_state=999, kernel='linear', probability=True)
gpc = GaussianProcessClassifier(kernel=1.0*RBF(1.0), random_state=999) 
rf = RandomForestClassifier(random_state = 999) 
et = ExtraTreesClassifier(random_state=999) 
hgbc = HistGradientBoostingClassifier(random_state=999)
lgbm = LGBMClassifier(random_state=999)
lr = LogisticRegression(random_state=999, multi_class='multinomial')
mlp = MLPClassifier(random_state=999, max_iter=300)

# Top 1
df = pd.DataFrame()

acc, rec_mac, pre_wei, f1_wei, cks, mcc = [], [], [], [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # preprocessing
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # model training
    linearsvm.fit(X_train, y_train)
    
    # model testing
    y_pred = linearsvm.predict(X_test)

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, y_pred)
    df.loc[i, 'Recall_macro'] = recall_score(y_test, y_pred, average='macro')
    df.loc[i, 'Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    df.loc[i, 'F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    df.loc[i, 'Kappa'] = cohen_kappa_score(y_test, y_pred)
    df.loc[i, 'MCC'] = matthews_corrcoef(y_test, y_pred)
    
for col in df.columns:    
    df.loc['Mean', col] = df[col].mean()
    df.loc['SD', col] = df[col].std()
    
df_in_1 = df.copy()

# Top 3
eclf = VotingClassifier(estimators=[('linearsvm', linearsvm), ('gpc', gpc), ('rf', rf)], 
                        voting='soft')
df = pd.DataFrame()

acc, rec_mac, pre_wei, f1_wei, cks, mcc = [], [], [], [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # preprocessing
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # model training
    eclf.fit(X_train, y_train)
    
    # model testing
    y_pred = eclf.predict(X_test)

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, y_pred)
    df.loc[i, 'Recall_macro'] = recall_score(y_test, y_pred, average='macro')
    df.loc[i, 'Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    df.loc[i, 'F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    df.loc[i, 'Kappa'] = cohen_kappa_score(y_test, y_pred)
    df.loc[i, 'MCC'] = matthews_corrcoef(y_test, y_pred)
    
for col in df.columns:    
    df.loc['Mean', col] = df[col].mean()
    df.loc['SD', col] = df[col].std()
    
df_in_3 = df.copy()


# Top 5
eclf = VotingClassifier(estimators=[('linearsvm', linearsvm), ('gpc', gpc), ('rf', rf), 
                                    ('et', et), ('hgbc', hgbc)], voting='soft')
df = pd.DataFrame()

acc, rec_mac, pre_wei, f1_wei, cks, mcc = [], [], [], [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # preprocessing
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # model training
    eclf.fit(X_train, y_train)
    
    # model testing
    y_pred = eclf.predict(X_test)

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, y_pred)
    df.loc[i, 'Recall_macro'] = recall_score(y_test, y_pred, average='macro')
    df.loc[i, 'Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    df.loc[i, 'F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    df.loc[i, 'Kappa'] = cohen_kappa_score(y_test, y_pred)
    df.loc[i, 'MCC'] = matthews_corrcoef(y_test, y_pred)
    
for col in df.columns:    
    df.loc['Mean', col] = df[col].mean()
    df.loc['SD', col] = df[col].std()
    
df_in_5 = df.copy()


# Top 7
eclf = VotingClassifier(estimators=[('linearsvm', linearsvm), ('gpc', gpc), ('rf', rf), 
                                    ('et', et), ('hgbc', hgbc), ('lgbm', lgbm),
                                    ('lr', lr)], voting='soft')
df = pd.DataFrame()

acc, rec_mac, pre_wei, f1_wei, cks, mcc = [], [], [], [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # preprocessing
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # model training
    eclf.fit(X_train, y_train)
    
    # model testing
    y_pred = eclf.predict(X_test)

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, y_pred)
    df.loc[i, 'Recall_macro'] = recall_score(y_test, y_pred, average='macro')
    df.loc[i, 'Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    df.loc[i, 'F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    df.loc[i, 'Kappa'] = cohen_kappa_score(y_test, y_pred)
    df.loc[i, 'MCC'] = matthews_corrcoef(y_test, y_pred)
    
for col in df.columns:    
    df.loc['Mean', col] = df[col].mean()
    df.loc['SD', col] = df[col].std()
    
df_in_7 = df.copy()
final = pd.concat([df_in_1, df_in_3, df_in_5, df_in_7])