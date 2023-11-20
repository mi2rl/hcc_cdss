import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


# Dataset lodaing
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


# Cascaded RF: Internal dataset training and validation with 5-fold CV

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 999)

X = np.array(df_AM.loc[:, 'age':'Meta'])
y = np.array(df_AM.loc[:, 'Tx'])
df = pd.DataFrame()

acc, rec_mac, pre_wei, f1_wei, cks, mcc = [], [], [], [], [], []

for i, (train_index, test_index) in enumerate(skf.split(X, y)):

    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Model training
    _X, _y = RFA_or_op(X_train, y_train)    
    clf_RFAOP = RFC_GS(_X, _y)

    _X, _y = RFA_vs_op(X_train, y_train) 
    clf_RFA_OP = RFC_GS(_X, _y)

    _X, _y = TACE_vs_etc(X_train, y_train) 
    clf_TACE = RFC_GS(_X, _y)

    _X, _y = TACL_vs_etc(X_train, y_train) 
    clf_TACL = RFC_GS(_X, _y)

    _X, _y = Sora_vs_etc(X_train, y_train) 
    clf_sorafenib = RFC_GS(_X, _y)
    
    # 1st prediction
    y_pred = Prediction(X_test)

    df.loc[i, 'Accuracy'] = accuracy_score(y_test, y_pred)
    df.loc[i, 'Recall_macro'] = recall_score(y_test, y_pred, average='macro')
    df.loc[i, 'Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    df.loc[i, 'F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    df.loc[i, 'Kappa'] = cohen_kappa_score(y_test, y_pred)
    df.loc[i, 'MCC'] = matthews_corrcoef(y_test, y_pred)
    
    pred_1 = y_pred.copy()
    pred_2 = []

    # Alternative options
    for j, Tx in enumerate(pred_1):
        features = X_test[j]
        pred_2.append(Alternatives(features, Tx))
        
    pred_2 = np.array(pred_2).reshape(-1)

    # 2nd results summation
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
    
cascaded_CV_internal = df.copy()


# Cascaded RF: training with whole internal dataset and validation with external datasets

# Model training with whole internal dataset
_X, _y = RFA_or_op(X, y)    
clf_RFAOP = RFC_GS(_X, _y)

_X, _y = RFA_vs_op(X, y) 
clf_RFA_OP = RFC_GS(_X, _y)

_X, _y = TACE_vs_etc(X, y) 
clf_TACE = RFC_GS(_X, _y)

_X, _y = TACL_vs_etc(X, y) 
clf_TACL = RFC_GS(_X, _y)

_X, _y = Sora_vs_etc(X, y) 
clf_sorafenib = RFC_GS(_X, _y)

df = pd.DataFrame()

for i, center in enumerate(centers):
    
    test_data = getattr(mod, f"df_{center}")
    X_test = np.array(test_data.loc[:, 'age':'Meta'])
    y_test = np.array(test_data.loc[:, 'Tx'])
    y_pred = Prediction(X_test)

    # 1st prediction
    df.loc[i, 'Center'] = center
    df.loc[i, 'Accuracy'] = accuracy_score(y_test, y_pred)
    df.loc[i, 'Recall_macro'] = recall_score(y_test, y_pred, average='macro')
    df.loc[i, 'Precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    df.loc[i, 'F1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    df.loc[i, 'Kappa'] = cohen_kappa_score(y_test, y_pred)
    df.loc[i, 'MCC'] = matthews_corrcoef(y_test, y_pred)
    
    # 2nd results summation & contraindication for first Tx option
    pred_1 = y_pred.copy()
    pred_2 = []

    # Alternative options
    for j, Tx in enumerate(pred_1):
        features = X_test[j]
        pred_2.append(Alternatives(features, Tx))
        
    pred_2 = np.array(pred_2).reshape(-1)

    # 2nd results summation
    pred_total = pred_1.copy()
    
    for j, real in enumerate(y_test):
        if pred_1[j] == real:
            pass
        else:
            if pred_2[j] == real:
                pred_total[i] = pred_2[j] 

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
    
cascaded_whole_external = df.copy()

# Functions

# Treatment: 0=RFA, 1=op, 2=TACE, 3=TACL, 4=sorafenib, 5=BSC(best supportive care)
# Data modofication for cascaded training: RFA or op -> 1,  others -> 0
def RFA_or_op(X, y):
    X_target = []
    y_target = []
    
    for i, _X in enumerate(X): 
        if y[i] < 2:
            X_target.append(np.array(_X))
            y_target.append(1)
        else:
            X_target.append(np.array(_X))
            y_target.append(0)

    X_target = np.array(X_target)
    y_target = np.array(y_target)
    
    return(X_target, y_target)


# Data modofication for cascaded training: RFA -> 1, Op -> 0, Others -> pass
def RFA_vs_op(X, y):
    X_target = []
    y_target = []
    
    for i, _X in enumerate(X): 
        if y[i] == 0: 
            X_target.append(np.array(_X))
            y_target.append(1)
        elif y[i] == 1:
            X_target.append(np.array(_X))
            y_target.append(0)
        else:
            pass

    X_target = np.array(X_target)
    y_target = np.array(y_target)
    
    return(X_target, y_target)


# Data modofication for cascaded training: RFA, op -> pass, TACE -> 1, others -> 0
def TACE_vs_etc(X, y): 
    X_target = []
    y_target = []
    
    for i, _X in enumerate(X): 
        if y[i] < 2:
            pass
        elif y[i] == 2: 
            X_target.append(np.array(_X))
            y_target.append(1)
        else:
            X_target.append(np.array(_X))
            y_target.append(0)

    X_target = np.array(X_target)
    y_target = np.array(y_target)
    
    return(X_target, y_target)


# Data modofication for cascaded training: RFA, op, TACE -> pass, TACE+RT -> 1, others -> 0
def TACL_vs_etc(X, y): 
    X_target = []
    y_target = []
    
    for i, _X in enumerate(X): 
        if y[i] < 3:
            pass
        elif y[i] == 3: 
            X_target.append(np.array(_X))
            y_target.append(1)
        else:
            X_target.append(np.array(_X))
            y_target.append(0)

    X_target = np.array(X_target)
    y_target = np.array(y_target)
    
    return(X_target, y_target)


# Data modofication for cascaded training: RFA, op, TACE, TACL -> pass, sorafenib -> 1, BSC -> 0
def Sora_vs_etc(X, y): 
    X_target = []
    y_target = []
    
    for i, _X in enumerate(X): 
        if y[i] < 4:
            pass
        elif y[i] == 4: 
            X_target.append(np.array(_X))
            y_target.append(1)
        else:
            X_target.append(np.array(_X))
            y_target.append(0)

    X_target = np.array(X_target)
    y_target = np.array(y_target)
    
    return(X_target, y_target)


# Cascaded prediction 
def Prediction (X_test):
    y_pred = []
    
    for i, _X in enumerate(X_test):
        clf_data = np.expand_dims(np.array(_X), axis = 0)
        result = clf_RFAOP.predict(clf_data)

        if result == 1.: # pred RFA+op? yes
            result = clf_RFA_OP.predict(clf_data)
            if result == 1.:# pred RFA? yes
                y_pred.append(0) # RFA
            else: # pred op
                y_pred.append(1) # Op
        else:
            result = clf_TACE.predict(clf_data)
            if result == 1.:
                y_pred.append(2) # TACE
            else:
                result = clf_TACL.predict(clf_data)
                if result == 1.:
                    y_pred.append(3) # TACL
                else:
                    result = clf_sorafenib.predict(clf_data)
                    if result == 1.:
                        y_pred.append(4) # sorafenib
                    else:
                        y_pred.append(5) # none

    y_pred = np.array(y_pred)
                        
    return (y_pred)


# Training with searching for the best parameters using grid search
def RFC_GS(X, y):

    rf = RandomForestClassifier(random_state = 999) # class_weight = 'balanced'
    parameters = {'criterion':('gini', 'entropy'), 
                  'n_estimators':np.arange(60, 300, 20), 
                  'max_depth':[int(x) for x in np.linspace(10, 110, num = 6)]}
    
    gs = GridSearchCV(rf, parameters)
    gs = gs.fit(X, y) 
    print(gs.best_params_)

    clf = RandomForestClassifier(random_state = 999,
                                 n_estimators=gs.best_params_['n_estimators'], 
                                 criterion=gs.best_params_['criterion'], 
                                 max_depth=gs.best_params_['max_depth'], 
                                 )
    clf = clf.fit(X, y)
    
    return(clf)


# Rule-based selection of alternative treatment options
def Alternatives (features, selected_trt):
    
    alternatives = []

    if selected_trt == 0: # RFA
        alternatives.append(2) # TACE

    elif selected_trt == 1: # OP    
        # if RFA_feasibility = 0, RFA
        if features[17] == 0: alternatives.append(0) 
        # Pvi_loca = 0, TACE
        elif features[18] == 0: alternatives.append(2) 
        else: alternatives.append(3) # TACL

    elif selected_trt == 2: # TACE - no alternative treatments
        alternatives.append(2)

    elif selected_trt == 3: # TACL
        alternatives.append(4) # Sorafenib
        #alternatives.append(5) # supportive care

    elif selected_trt == 4: # Sorafenib
        alternatives.append(5) # supportive care

    elif selected_trt == 5: # supportive care - no alternative treatments
        alternatives.append(5)
    #elif selected_trt == 6: # Others - no alternative treatments

    return (alternatives)
