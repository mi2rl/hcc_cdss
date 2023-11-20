# Model for Survival Prediction

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from lifelines import KaplanMeierFitter
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import brier_score
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import integrated_brier_score


# Loading dataset
df = pd.read_excel('./Dataset.xlsx', sheet_name='Sheet1', engine='openpyxl')

# 9 institutional datasets        
df_KU = df[df['Center'] == 1]
df_BH = df[df['Center'] == 2]
df_SM = df[df['Center'] == 3]
df_SN = df[df['Center'] == 4]
df_CM = df[df['Center'] == 5]
df_SV = df[df['Center'] == 6]
df_AM = df[df['Center'] == 7]
df_CA = df[df['Center'] == 8]
df_IH = df[df['Center'] == 9]
df_ex = df[df['Center'] != 7]

# 1. Model training and validation

# 1.1. Internal validation with 5-fold cross validation
X = df_AM.loc[:, 'age':'Tx']
y = Surv.from_dataframe('death_01', 'death_mo', df_AM)
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 999)
rsf = RandomSurvivalForest(random_state=999)

df = pd.DataFrame()
for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train = X.loc[train_index, :]
    y_train = y[train_index]
    X_test = X.loc[test_index, :]
    y_test = y[test_index]

    # model training for internal dataset with 5-fold CV
    estimator = rsf.fit(X_train, y_train)
    
    event_min = y_train[y_train["death_01"] == 1]["death_mo"].min()
    event_max = y_train[y_train["death_01"] == 1]["death_mo"].max()
    idx = np.where((event_min <= y_test["death_mo"]) & (y_test["death_mo"] < event_max))
    idx = np.array(list(idx)).flatten()

    X_test = X_test.loc[X_test.index[idx], :]
    y_test = y_test[idx]

    # model testing for internal dataset
    chf_funcs = estimator.predict_cumulative_hazard_function(X_test)
    surv_funcs = estimator.predict_survival_function(X_test)
    times = np.percentile(y_test["death_mo"], np.linspace(5, 81, 15))
    risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) 

    c_index = estimator.score(X_test, y_test)
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times) 
    preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
    IBS = integrated_brier_score(y_train, y_test, preds, times)    

    df.loc[i, 'C_index'] = c_index
    df.loc[i, 'Mean_auc'] = mean_auc
    df.loc[i, 'IBS'] = IBS
    
for col in df.columns:
    df.loc['Mean', col] = df[col].mean()
    df.loc['SD', col] = df[col].std()
    
internal = df.copy()

# 1.2. External validation after training with whole internal dataset

# model training with whole internal dataset
X = df_AM.loc[:, 'age':'Tx']
y = Surv.from_dataframe('death_01', 'death_mo', df_AM)
estimator = rsf.fit(X, y)

# model testing for external validation datasets 
centers = ["KU", "BH", "SM", "SN", "CM", "SV", "CA", "IH"]
mod = sys.modules[__name__]

def external_validaiton(model, y_train):
    
    df = pd.DataFrame()

    for i, center in enumerate(centers):
        test_data = getattr(mod, f"df_{center}")
        
        test_data.reset_index(drop=True, inplace=True)
        
        X_test = test_data.loc[:, 'age':'Tx']
        y_test = Surv.from_dataframe('death_01', 'death_mo', test_data)

        event_min = y_train[y_train["death_01"] == 1]["death_mo"].min()
        event_max = y_train[y_train["death_01"] == 1]["death_mo"].max()
        idx = np.where((event_min <= y_test["death_mo"]) & (y_test["death_mo"] < event_max))
        idx = np.array(list(idx)).flatten()

        X_test = X_test.loc[X_test.index[idx], :]
        y_test = y_test[idx]

        chf_funcs = estimator.predict_cumulative_hazard_function(X_test)
        surv_funcs = estimator.predict_survival_function(X_test)
        times = np.percentile(y_test["death_mo"], np.linspace(5, 81, 15))
        risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) 
        
        c_index = estimator.score(X_test, y_test)
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times) 
        preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
        IBS = integrated_brier_score(y_train, y_test, preds, times)    

        df.loc[i, 'Center'] = center
        df.loc[i, 'C_index'] = c_index
        df.loc[i, 'Mean_auc'] = mean_auc
        df.loc[i, 'IBS'] = IBS
    
    return (df)

# external validation results
df = external_validaiton(estimator, y)

for col in df.columns[1:]:
    df.loc['Mean', col] = df[col].mean()
    df.loc['SD', col] = df[col].std()
    
external = df.copy()

# 1.3. Training and validation with individual dataset 

df_all = pd.DataFrame()

for n, center in enumerate(centers):
    
    test_data = getattr(mod, f"df_{center}")
    test_data.reset_index(drop=True, inplace=True)
    
    X = test_data.loc[:, 'age':'Tx']
    Tx = test_data.loc[:, 'Tx']
    y = Surv.from_dataframe('death_01', 'death_mo', test_data)
    df = pd.DataFrame()
    
    # model training with individual dataset with 5-fold CV
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train = X.loc[train_index, :]
        y_train = y[train_index]
        X_test = X.loc[test_index, :]
        y_test = y[test_index]
        
        estimator = rsf.fit(X_train, y_train)

        event_min = y_train[y_train["death_01"] == 1]["death_mo"].min()
        event_max = y_train[y_train["death_01"] == 1]["death_mo"].max()
        idx = np.where((event_min <= y_test["death_mo"]) & (y_test["death_mo"] < event_max))
        idx = np.array(list(idx)).flatten()
        X_test = X_test.loc[X_test.index[idx], :]
        y_test = y_test[idx]
        
        # model testing for individual dataset
        chf_funcs = estimator.predict_cumulative_hazard_function(X_test)
        surv_funcs = estimator.predict_survival_function(X_test)
        times = np.percentile(y_test["death_mo"], np.linspace(5, 81, 15))
        risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) 

        c_index = estimator.score(X_test, y_test)
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times) 
        preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
        IBS = integrated_brier_score(y_train, y_test, preds, times)    

        df.loc[i, 'C_index'] = c_index
        df.loc[i, 'Mean_auc'] = mean_auc
        df.loc[i, 'IBS'] = IBS
        
    for col in df.columns:
        df.loc['Mean', col] = df[col].mean()
        df.loc['SD', col] = df[col].std()
        
    df_all = pd.concat([df_all, df])

# 2. Model training and validation for each specific treatment

# 2.1. Internal validation
Tx = df_AM.loc[:, 'Tx']

X.reset_index(drop=True, inplace=True)
Tx_name = ['RFA', 'Op', 'TACE', 'TACE+RT', 'Sorafenib', 'None']
df = pd.DataFrame()

# 5-fold CV stratified by treatment
for i, (train_index, test_index) in enumerate(skf.split(X, Tx)):

    X_train_fold = X.loc[train_index, :]
    y_train_fold = y[train_index]
    X_test_fold = X.loc[test_index, :]
    y_test_fold = y[test_index]

    # composing dataset including only specific treatment
    for tx, tx_name in enumerate(Tx_name):
        train_idx = X_train_fold[X_train_fold['Tx'] == tx].index
        test_idx = X_test_fold[X_test_fold['Tx'] == tx].index
        
        X_train = X.loc[train_idx, :]
        y_train = y[train_idx]
        X_test = X.loc[test_idx, :]
        y_test = y[test_idx]
        
        # model training with training dataset including only specific treatment
        estimator = rsf.fit(X_train, y_train)
        
        if len(X_test) == 0:
            df.loc[i, f'{tx}_Tx_name'] = tx_name
            df.loc[i, f'{tx}_C_index'] = 0
            df.loc[i, f'{tx}_Mean_auc'] = 0
            df.loc[i, f'{tx}_IBS'] = 0
                
        else:
            event_min = y_train[y_train["death_01"] == 1]["death_mo"].min()
            event_max = y_train[y_train["death_01"] == 1]["death_mo"].max()
            idx = np.where((event_min <= y_test["death_mo"]) & (y_test["death_mo"] < event_max))
            idx = np.array(list(idx)).flatten()

            X_test = X_test.loc[X_test.index[idx], :]
            y_test = y_test[idx]

            # model testing with training dataset including only specific treatment
            chf_funcs = estimator.predict_cumulative_hazard_function(X_test)
            surv_funcs = estimator.predict_survival_function(X_test)
            times = np.percentile(y_test["death_mo"], np.linspace(5, 81, 10))
            times = np.unique(times)
            risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) 

            c_index = estimator.score(X_test, y_test)
            auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times) 
            preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
            IBS = integrated_brier_score(y_train, y_test, preds, times)    

            df.loc[i, f'{tx}_Tx_name'] = tx_name
            df.loc[i, f'{tx}_C_index'] = c_index
            df.loc[i, f'{tx}_Mean_auc'] = mean_auc
            df.loc[i, f'{tx}_IBS'] = IBS  


# 2.2. External validation
y_train = y.copy()
df = pd.DataFrame()

for i, center in enumerate(centers):
    print (i)
    
    test_data = getattr(mod, f"df_{center}")
    test_data.reset_index(drop=True, inplace=True)

    X_test = test_data.loc[:, 'age':'Tx']
    y_test = Surv.from_dataframe('death_01', 'death_mo', test_data)

    for tx, tx_name in enumerate(Tx_name):
        
        train_idx = X[X['Tx'] == tx].index
        X_train = X.loc[train_idx, :]
        y_train = y[train_idx]
        estimator = rsf.fit(X_train, y_train)

        test_idx = X_test[X_test['Tx'] == tx].index
        new_X_test = X.loc[test_idx, :]
        new_y_test = y[test_idx]
        
        event_min = y_train[y_train["death_01"] == 1]["death_mo"].min()
        event_max = y_train[y_train["death_01"] == 1]["death_mo"].max()
        idx = np.where((event_min <= new_y_test["death_mo"]) & (new_y_test["death_mo"] < event_max))
        idx = np.array(list(idx)).flatten()

        new_X_test = new_X_test.loc[new_X_test.index[idx], :]
        new_y_test = new_y_test[idx]
        
        if len(new_X_test) < 2:
            df.loc[i, f'{tx}_Tx_name'] = tx_name
            df.loc[i, f'{tx}_Test_no'] = len(new_X_test)
            df.loc[i, f'{tx}_C_index'] = 0
            df.loc[i, f'{tx}_Mean_auc'] = 0
            df.loc[i, f'{tx}_IBS'] = 0
            
        else:
            chf_funcs = estimator.predict_cumulative_hazard_function(new_X_test)
            surv_funcs = estimator.predict_survival_function(new_X_test)
            times = np.percentile(np.unique(new_y_test["death_mo"]), np.linspace(5, 81, 10))
            times = np.unique(times)
            risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) 

            c_index = estimator.score(new_X_test, new_y_test)
            auc, mean_auc = cumulative_dynamic_auc(y_train, new_y_test, risk_scores, times) 
            preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
            IBS = integrated_brier_score(y_train, new_y_test, preds, times)    

            df.loc[i, f'{tx}_Tx_name'] = tx_name
            df.loc[i, f'{tx}_Test_no'] = len(new_X_test)
            df.loc[i, f'{tx}_C_index'] = c_index
            df.loc[i, f'{tx}_Mean_auc'] = mean_auc
            df.loc[i, f'{tx}_IBS'] = IBS

# 2.3. Individual training & validation
df_all = pd.DataFrame()

for n, center in enumerate(centers):
    
    print (n)
    test_data = getattr(mod, f"df_{center}")
    test_data.reset_index(drop=True, inplace=True)
    
    X = test_data.loc[:, 'age':'Tx']
    Tx = test_data.loc[:, 'Tx']
    y = Surv.from_dataframe('death_01', 'death_mo', test_data)
    df = pd.DataFrame()
    
    for i, (train_index, test_index) in enumerate(skf.split(X, Tx)):

        X_train = X.loc[train_index, :]
        y_train = y[train_index]
        X_test = X.loc[test_index, :]
        y_test = y[test_index]
        
        estimator = rsf.fit(X_train, y_train)

        for tx, tx_name in enumerate(Tx_name):
            tmp_idx = X_test[X_test['Tx'] == tx].index
            new_X_test = X.loc[tmp_idx, :]
            new_y_test = y[tmp_idx]

            event_min = y_train[y_train["death_01"] == 1]["death_mo"].min()
            event_max = y_train[y_train["death_01"] == 1]["death_mo"].max()
            idx = np.where((event_min <= new_y_test["death_mo"]) & (new_y_test["death_mo"] < event_max))
            idx = np.array(list(idx)).flatten()

            new_X_test = new_X_test.loc[new_X_test.index[idx], :]
            new_y_test = new_y_test[idx]

            if len(new_X_test) < 2:
                df.loc[i, f'{tx}_Tx_name'] = tx_name
                df.loc[i, f'{tx}_Test_no'] = len(new_X_test)
                df.loc[i, f'{tx}_C_index'] = 0
                df.loc[i, f'{tx}_Mean_auc'] = 0
                df.loc[i, f'{tx}_IBS'] = 0
            
            else:
                chf_funcs = estimator.predict_cumulative_hazard_function(new_X_test)
                surv_funcs = estimator.predict_survival_function(new_X_test)
                times = np.percentile(new_y_test["death_mo"], np.linspace(5, 81, 10))
                times = np.unique(times)
                risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) 

                try:
                    c_index = estimator.score(new_X_test, new_y_test)
                    auc, mean_auc = cumulative_dynamic_auc(y_train, new_y_test, risk_scores, times) 
                    preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
                    IBS = integrated_brier_score(y_train, new_y_test, preds, times)    

                    df.loc[i, f'{tx}_Tx_name'] = tx_name
                    df.loc[i, f'{tx}_C_index'] = c_index
                    df.loc[i, f'{tx}_Mean_auc'] = mean_auc
                    df.loc[i, f'{tx}_IBS'] = IBS

                except:
                    ValueError
                    df.loc[i, f'{tx}_Tx_name'] = tx_name
                    df.loc[i, f'{tx}_C_index'] = 'Value error'
                    df.loc[i, f'{tx}_Mean_auc'] = 'Value error'
                    df.loc[i, f'{tx}_IBS'] = 'Value error'                
            
    df_all = pd.concat([df_all, df])
