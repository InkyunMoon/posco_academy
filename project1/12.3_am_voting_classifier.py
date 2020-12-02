import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df_tele = pd.read_csv('C:/Users/moon/Documents/github/posco_academy/project1/B4_카드_DataSet/tele_for_modele.csv')
df_tele.age = df_tele.age.map(lambda x: str(x)+'대')
df_tele.info()

df_tele.drop('card_type', axis=1, inplace = True)

scaler = StandardScaler()
scaled_2 = scaler.fit_transform(df_tele[['no_call_past','call_duration']])
df_scaled_2 = pd.DataFrame({'no_call_past':scaled_2[:,0],'call_duration':scaled_2[:,1]})
Y = df_tele['is_contract']
Y = Y.map({'yes':1,'no':0})
X = df_tele.drop(['is_contract','no_call_past','call_duration'], axis=1)
X = pd.concat([X,df_scaled_2], axis=1)
X = pd.get_dummies(X)

kfold = StratifiedKFold(n_splits = 3)

trainX,testX,trainY,testY = train_test_split(X,Y,test_size = 0.2, random_state = 1768, stratify=Y)

random_state = 1768
classifiers = []
classifiers.append(SVC(random_state=random_state))
# classifiers.append(DecisionTreeClassifier(random_state=random_state))
# classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
# classifiers.append(LogisticRegression(random_state = random_state))

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, trainX, y = trainY, scoring = "accuracy", cv = kfold, n_jobs=-1))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"Accuracy(mean)":cv_means,"std_of_accuracy": cv_std,"Algorithm":["SVC","RandomForest","GradientBoosting","MultipleLayerPerceptron","KNeighboors"]})
cv_res.sort_values(by='Accuracy(mean)', axis = 0, ascending=False)

g = sns.barplot('Accuracy(mean)','Algorithm',data=cv_res,palette='Set3',orient='h',xerr=cv_std)
g.set_xlabel('Mean Accuracy')
g = g.set_title('Cross validation scores')

RFC = RandomForestClassifier()
rf_param_grid = {'max_depth':[None],
                'max_features':[1,3,10],
                'min_samples_split':[2,3,10],
                'min_samples_leaf':[1,3,10],
                'bootstrap':[False],
                'n_estimators':[100,300],
                'criterion':['gini']}
gsRFC = GridSearchCV(RFC, param_grid = rf_param_grid, cv = kfold, scoring = 'accuracy', n_jobs = 4, verbose = 1)
gsRFC.fit(trainX, trainY)
RFC_best = gsRFC.best_estimator_
gsRFC.best_score_

GBC = GradientBoostingClassifier()
gb_param_grid={'loss':['deviance'],
              'n_estimators':[100,200,300],
              'learning_rate':[0.1,0.05,0.01],
              'max_depth':[4,8],
              'min_samples_leaf':[100,150],
              'max_features':[0.3,0.1]
              }
gsGBC = GridSearchCV(GBC,param_grid=gb_param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1)
gsGBC.fit(trainX, trainY)
GBC_best = gsGBC.best_estimator_
gsGBC.best_score_

SVMC = SVC(probability=True)
svc_param_grid = {'kernel': 'rbf', 
                  'gamma': [ 0.001, 0.01, 0.1, 1], # 마진 설정시 이상치를 얼마나 허용할 것인지
                  'C': [1, 10, 50, 100,200,300]} # 결정 경계를 얼마나 유연하게 설정할 것인지
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(trainX,trainY)
SVMC_best = gsSVMC.best_estimator_
# Best score
gsSVMC.best_score_

MLP = MLPClassifier()
mlp_params_grid = {'hidden_layer_sizes':[10,30,50,70,90,110,130],
                   'alpha':[0.0002, 0.0005,0.001,0.005,0.01]}
gsMLP = GridSearchCV(MLP, param_grid = mlp_params_grid, cv = kfold, scoring= 'accuracy', n_jobs = 4, verbose = 1)
gsMLP.fit(trainX,trainY)
MLP_best = gsMLP.best_estimator_
gsMLP.best_score_

KNN = KNeighborsClassifier()
knn_params_grid = {'n_neighbors':[3,5,9,13,20,30,50],
                   'leaf_size':[15,30,45,60,80]}
gsKNN = GridSearch(KNN, param_group = knn_params_grid, cv = kfold, scoring = 'accuracy', n_jobs = 4, verbose = 1)
gsKNN.fit(trainX, trainY)
KNN_best = gsKNN.best_estimator_
gsKNN.best_score_

test_Survived_RFC = pd.Series(RFC_best.predict(testX), name="RFC")
test_Survived_SVC = pd.Series(SVMC_best.predict(testX), name="SVMC")
test_Survived_GBC = pd.Series(gsGBC_best.predict(testX), name="GBC")
test_Survived_MLP = pd.Series(gsMLP_best.predict(testX), name="MLP")
test_Survived_KNN = pd.Series(gsKNN_best.predict(testX), name="KNN")

# ensemble_results = pd.concat([test_Survived_RFC,test_Survived_SVMC,test_Survived_gsGBC,test_Survived_MLP, test_Survived_KNN],axis=1)
# g= sns.heatmap(ensemble_results.corr(),annot=True)

# 임시로 돌려보기
# SVMC.fit(trainX,trainY)
# GBC.fit(trainX,trainY)
# RFC.fit(trainX,trainY)
# MLP.fit(trainX,trainY)
# KNN.fit(trainX,trainY)

votingC = VotingClassifier(estimators=[('rfc', RFC), ('gbc', GBC), ('svc', SVMC), ('knn',KNN),('MLP', MLP)], voting='soft', n_jobs=4)
votingC = votingC.fit(trainX, trainY)

conf_voting = confusion_matrix(votingC.predict(testX), testY)
conf_voting
(6628+224)/(6628+627+224+149)
