import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.metrics import integrated_brier_score


# Dataset loading
df = pd.read_excel('/Nas/0_All_rev.xlsx', 
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
df_ex = df[df['Center'] != 7]


# train and testset splitting of internal dataset (No 5-fold CV)
df_train, df_test = train_test_split(df_AM, test_size=0.2,
                                     random_state=999, stratify=df_AM['Tx'])

X_train = np.array(df_train.loc[:, 'age':'Meta'])
y_train = np.array(df_train.loc[:, 'Tx'])
X_test = np.array(df_test.loc[:, 'age':'Meta'])
y_test = np.array(df_test.loc[:, 'Tx'])

# ensemble voting classifier 
linearsvm = svm.SVC(random_state=999, kernel='linear', probability=True)
gpc = GaussianProcessClassifier(kernel=1.0*RBF(1.0), random_state=999) 
rf = RandomForestClassifier(random_state = 999) 
et = ExtraTreesClassifier(random_state=999) 
hgbc = HistGradientBoostingClassifier(random_state=999)

eclf = VotingClassifier(estimators=[('linearsvm', linearsvm), ('gpc', gpc), ('rf', rf), 
                                    ('et', et), ('hgbc', hgbc)], voting='soft')

# preprocessing and model training for treatment recommendation
_scaler = MinMaxScaler().fit(X_train)
X_train = _scaler.transform(X_train)
eclf.fit(X_train, y_train)

# obtaining the first and second treatment options
for i in tqdm(df_test.index):
    
    test_data = np.array(df_test.loc[i, 'age':'Meta'])
    test_data = test_data.reshape(1, -1)
    test_data = _scaler.transform(test_data)
    
    results = eclf.predict_proba(test_data).flatten()
    final = np.argsort(results)

    df_test.loc[i, 'pred_1'] = final[-1]
    df_test.loc[i, 'prob_1'] = results[final[-1]]
    df_test.loc[i, 'pred_2'] = final[-2]
    df_test.loc[i, 'prob_2'] = results[final[-2]]


# definition for each test group
df_op = df_test[df_test['Tx'] == 1] # patients group of real treatment = op
df_TACE = df_test[df_test['Tx'] == 2] # patients group of real treatment = TACE
op_op = df_op[df_op['pred_1'] == 1].copy() # patients group of real treatment = op, model prediction = op
op_TACE = df_op[df_op['pred_1'] == 2].copy() # patients group of real treatment = op, model prediction = TACE


# Kaplan-Meier plot for group comparison
time1, event1 = op_op.loc[:,'death_mo'], op_op.loc[:,'death_01']
time2, event2 = op_TACE.loc[:,'death_mo'], op_TACE.loc[:,'death_01']
kmf1 = KaplanMeierFitter()
kmf1.fit(time1, event1, label="Real=Resection & Prediction=Resection")
kmf2 = KaplanMeierFitter()
kmf2.fit(time2, event2, label="Real=Resection & Prediction=TACE")

ax = plt.subplot(111)
kmf1.plot(ax=ax, lw=2)
kmf2.plot(ax=ax, lw=2)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
plt.xticks([0, 20, 40, 60, 80, 100])
plt.grid(True)
plt.legend(loc='upper right', prop={'size':14})
fig = plt.gcf()
plt.show()
results = logrank_test(time1, time2, event_observed_A=event1, event_observed_B=event2)


# Model training for survival prediction
X_train = df_train.loc[:, 'age':'Tx']
y_train = Surv.from_dataframe('death_01', 'death_mo', df_train)
X_test = df_test.loc[:, 'age':'Tx']
y_test = Surv.from_dataframe('death_01', 'death_mo', df_test)
estimator = rsf.fit(X_train, y_train)


# Evaluation of survival prediction for each group
# 1) Real = Resection, Pred = Resection, input Tx = Resection 
df_test = op_op.copy() 
X_test = df_test.loc[:, 'age':'Tx']
y_test = Surv.from_dataframe('death_01', 'death_mo', df_test)
c_index, mean_auc, IBS = evaluation(estimator, X_test, y_test, y_train)
prediction_plot(estimator, X_test, df_test)

# 2) Real = Resection, Pred = TACE, input Tx = Resection
df_test = op_TACE.copy()
X_test = df_test.loc[:, 'age':'Tx']
y_test = Surv.from_dataframe('death_01', 'death_mo', df_test)


# 3) Real = Resection, Pred = TACE, input Tx = TACE
df_test = op_TACE.copy()
X_test.loc[:, 'Tx'] = 2 # intentional modification of Tx from resection to TACE
y_test = Surv.from_dataframe('death_01', 'death_mo', df_test)


# evaliation metrics
def evaluation(estimator, X_test, y_test, y_train):    
    chf_funcs = estimator.predict_cumulative_hazard_function(X_test)
    surv_funcs = estimator.predict_survival_function(X_test)
    times = np.arange(6, 90, 6)
    risk_scores = np.row_stack([chf(times) for chf in chf_funcs]) 

    c_index = estimator.score(X_test, y_test)
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times) 
    preds = np.asarray([[fn(t) for t in times] for fn in surv_funcs])
    IBS = integrated_brier_score(y_train, y_test, preds, times)
    
    return(c_index, mean_auc, IBS)

# comparison plot for model prediction vs. Kaplan-Meier plot
def prediction_plot(estimator, X_test, df_test):    
    surv_funcs = estimator.predict_survival_function(X_test)
    
    x_time = surv_funcs[0].x
    y_survival = np.array([surv_funcs[i].y for i in range(len(surv_funcs))])

    surv_average = np.average(y_survival, axis=0)
    surv_std = np.std(y_survival, axis=0)
    surv_upper = np.minimum(surv_average + surv_std, 1)
    surv_lower = np.maximum(surv_average - surv_std, 0)

    t_test_A = np.array(df_test.loc[:,'death_mo'])
    e_test_A = np.array(df_test.loc[:,'death_01'])

    ax = plt.subplot(111)
    kmf = KaplanMeierFitter()
    kmf.fit(t_test_A, e_test_A, label='Ground truth')
    kmf.plot(ax=ax, lw=1)

    plt.plot(x_time, surv_average, label='Prediction', color='red', lw=1)
    plt.fill_between(x_time, surv_lower, surv_upper, color='red', alpha=.2)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([0, 20, 40, 60, 80, 100])
    plt.grid(True)
    plt.legend(['Ground truth', 'Prediction'], loc='upper right', prop={'size':14})
    fig = plt.gcf()
    plt.show()
    
    return()

