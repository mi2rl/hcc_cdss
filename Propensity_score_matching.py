import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.manifold import TSNE
import seaborn as sns


# Loading dataset
df = pd.read_excel('./Dataset.xlsx', sheet_name='Sheet1', engine='openpyxl')
df_AM = df[df['Center'] == 7]
df_ex = df[df['Center'] != 7]


##### 1. Propensity score matching #####

# df_op = Resection group in external dataset, df_TACE = TACE group in external dataset
df_op = df_ex[df_ex['Tx'] == 1].copy() 
df_TACE = df_ex[df_ex['Tx'] == 2].copy() 

X_treated = df_TACE.loc[:, 'age':'Meta'].values 
X_control = df_op.loc[:, 'age':'Meta'].values 

# Standard normalization 
scaler = StandardScaler()
X_treated_scaled = scaler.fit_transform(X_treated)
X_control_scaled = scaler.transform(X_control)

# Propensity score model fitting 
model = LogisticRegression(solver='lbfgs')
model.fit(np.vstack((X_treated_scaled, X_control_scaled)), np.hstack((np.ones(len(X_treated_scaled)), np.zeros(len(X_control_scaled)))))

# Calculating propensity score
propensity_scores_treated = model.predict_proba(X_treated_scaled)[:, 1]
propensity_scores_control = model.predict_proba(X_control_scaled)[:, 1]

control_idx = []  
treated_idx = []  

# Select the most similar control group data for each data point in the treatment group
for i, t in enumerate(propensity_scores_treated):
    distances = euclidean_distances(t.reshape(-1, 1), propensity_scores_control.reshape(-1, 1))  
    indices = np.argsort(distances)  

    # Check if the index has already been selected and, if not, add it to the matched pair
    for j in indices[0]:
        if j not in control_idx:
            if distances[0][j] < 0.1:
                control_idx.append(j)
                treated_idx.append(i)
                break
                
_df_op = df_op.reset_index().copy()
_df_TACE = df_TACE.reset_index().copy()

df_TACE_sorted = _df_TACE.loc[treated_idx].copy()
df_op_sorted = _df_op.loc[control_idx].copy()


# Model training for treatment recommendation with whole internal dataset 
X = np.array(df_AM.loc[:, 'age':'Meta'])
y = np.array(df_AM.loc[:, 'Tx'])

scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)

lr = LogisticRegression(random_state=999, multi_class='multinomial')
rf = RandomForestClassifier(random_state = 999)
lgbm = LGBMClassifier(random_state=999)
sv = svm.SVC(random_state=999, probability=True)
mlp = MLPClassifier(random_state=999, max_iter=300)
eclf = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('lgbm', lgbm), ('svm', sv), ('mlp', mlp)], voting='soft')

eclf.fit(X, y)

# Model testing  
for i in df_ex.index:    
    test_data = np.array(df_ex.loc[i, 'age':'Meta'])
    test_data = test_data.reshape(1, -1)
    test_data = scaler.transform(test_data)
    
    results = eclf.predict_proba(test_data).flatten()
    final = np.argsort(results)
    df_ex.loc[i, 'pred_1'] = final[-1]


##### 2. Confusion matrix before and after PSM #####

df_op = df_op_sorted.copy() # matched resection group
df_TACE = df_TACE_sorted.copy() # matched TACE group
_df_op = df_ex[df_ex['Tx'] == 1].copy() # original resection group 
_df_TACE = df_ex[df_ex['Tx'] == 2].copy() # original TACE group

# Retrieve the prediction values
for i in df_ex.index:
    for j in df_op.index:
        if df_op.loc[j, 'index'] == i:
            df_op.loc[j, 'pred_1'] = df_ex.loc[i, 'pred_1']  
            
for i in df_ex.index:
    for j in df_TACE.index:
        if df_TACE.loc[j, 'index'] == i:
            df_TACE.loc[j, 'pred_1'] = df_ex.loc[i, 'pred_1']  

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          reverse=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if reverse == True:
        cm = cm[::-1,::-1]
    cm_origin = cm.copy()
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)        

    fmt = '.2f' if normalize else 'd'
    thresh = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt)+'\n({})'.format(cm_origin[i,j]),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=15)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
    return()

classe_name = ['RFA', 'Resection', 'TACE', 'TACE+RT', 'Sorafenib', 'None']

# Confusion matrix for Matched resection group
y_real = np.array(df_op.loc[:, 'Tx'])
y_pred = np.array(df_op.loc[:, 'pred_1'])
cm = confusion_matrix(y_real, y_pred)

plt.figure(figsize=(8,8))
classes = [classe_name[int(i)] for i in sorted(df_op.loc[:, 'pred_1'].unique())]
plot_confusion_matrix(cm, classes=classes, normalize=True, reverse=False)

# Confusion matrix for original resection group
y_real = np.array(_df_op.loc[:, 'Tx'])
y_pred = np.array(_df_op.loc[:, 'pred_1'])
cm = confusion_matrix(y_real, y_pred)

plt.figure(figsize=(8,8))
classes = [classe_name[int(i)] for i in sorted(_df_op.loc[:, 'pred_1'].unique())]
plot_confusion_matrix(cm, classes=classes, normalize=True, reverse=False)

# Confusion matrix for Matched TACE group
y_real = np.array(df_TACE.loc[:, 'Tx'])
y_pred = np.array(df_TACE.loc[:, 'pred_1'])
cm = confusion_matrix(y_real, y_pred)

plt.figure(figsize=(8,8))
classes = [classe_name[int(i)] for i in sorted(df_TACE.loc[:, 'pred_1'].unique())]
plot_confusion_matrix(cm, classes=classes, normalize=True, reverse=False)

# Confusion matrix for original TACE group
y_real = np.array(_df_TACE.loc[:, 'Tx'])
y_pred = np.array(_df_TACE.loc[:, 'pred_1'])
cm = confusion_matrix(y_real, y_pred)

plt.figure(figsize=(8,8))
classes = [classe_name[int(i)] for i in sorted(_df_TACE.loc[:, 'pred_1'].unique())]
plot_confusion_matrix(cm, classes=classes, normalize=True, reverse=False)


##### 3. Feature analysis using t-SNE #####

# 20 features before feature reduction
AMC_TACE = np.array(df_AM[df_AM['Tx'] == 2].loc[:, 'age':'Meta']) # TACE group in training dataset
AMC_op = np.array(df_AM[df_AM['Tx'] == 1].loc[:, 'age':'Meta']) # Resection group in training dataset
correct = np.array(df_TACE[df_TACE['pred_1'] == 2].loc[:, 'age':'Meta']) # Pred = TACE in matched TACE group
incorrect = np.array(df_TACE[df_TACE['pred_1'] == 1].loc[:, 'age':'Meta']) # Pred = Resection in matched TACE group

_AMC_TACE = scaler.transform(AMC_TACE)
_AMC_op = scaler.transform(AMC_op)
_correct = scaler.transform(correct)
_incorrect = scaler.transform(incorrect)

# Feature reduction using t-SNE
_X = []
_y = []

for n in _AMC_TACE: 
    _X.append(n)
    _y.append(0)
    
for n in _AMC_op: 
    _X.append(n)
    _y.append(1)
    
for n in _correct: 
    _X.append(n)
    _y.append(2)
    
for n in _incorrect: 
    _X.append(n)
    _y.append(3)

T = TSNE(n_components = 2, perplexity=150).fit_transform(_X)
df = pd.DataFrame(T, columns=['1st dimension', '2nd dimension'])

for i, y in enumerate(_y):
    if y == 0:
        df.loc[i, 'Label'] = 'TACE group in the training dataset'
    elif y == 1:
        df.loc[i, 'Label'] = 'Resection group in the training dataset'
    elif y == 2:
        df.loc[i, 'Label'] = 'Correct prediction (=TACE) in the TACE group'
    elif y == 3:
        df.loc[i, 'Label'] = 'Incorrect prediction (=Resection) in the TACE group'

fig = plt.figure(figsize=(8,8)) 
fig.set_facecolor('white') 

sns.set_palette("bright")
sns.scatterplot(data=df, x='1st dimension', y='2nd dimension', hue='Label',
                alpha=0.7, s=70, style='Label')

plt.title('2-Dimensional t-SNE', fontsize=18)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=14)
plt.show() 