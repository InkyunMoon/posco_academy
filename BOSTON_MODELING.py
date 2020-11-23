import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV

path = 'C:\\Users\\moon\\Documents\\github\\posco_academy\\'

df_boston = pd.read_csv(path + 'boston.csv' , engine='python' ,encoding='CP949')
df_boston.rename(columns={'癤풫EDV':'MEDV'}, inplace = True)
df_boston.head()

df_boston.CHAS = df_boston.CHAS.astype('object')
df_boston.RAD = df_boston.RAD.astype('object')
df_boston.TAX = df_boston.TAX.astype('object')
df_boston.INDUS = df_boston.INDUS.astype('object')

df_boston.info()

Y = df_boston.MEDV
X = df_boston.drop('MEDV', axis = 1)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2)
trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size = 0.3)

# 디폴트 모델로 돌려본다.
random_state = 1234
classifiers = []
classifiers.append(DecisionTreeRegressor(random_state = 1234))
classifiers.append(RandomForestRegressor(random_state = 1234))
classifiers.append(GradientBoostingRegressor(random_state = 1234))

cv_results = []
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier, trainX, y=trainY, scoring='r2', cv=5))
    
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
    
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,\
                       "Algorithm":["DecisionTree","RandomForest","GradientBoosting"]})

gridX, gridY = pd.concat([trainX,validX]), pd.concat([trainY,validY])
    
# DECISION TREE
DTC = DecisionTreeRegressor()
DTC_param_grid = {'max_depth':[2,4,6,8,10],
                 'min_samples_leaf':[3,5,7,9,11],
                 'min_samples_split':[5,10,15,20,25]}
gsadaDTC = GridSearchCV(DTC, param_grid = DTC_param_grid, cv = 8, scoring = 'r2', verbose =1, n_jobs = 4)
gsadaDTC.fit(pd.concat([trainX,validX]), pd.concat([trainY,validY]))

DTC_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_

# RANDOM FOREST
RFC = RandomForestRegressor()    
rf_param_grid = {'max_depth':[None],
                'max_features':[3,5,7,9,11],
                'min_samples_split':[2,3,5,7,9],
                'min_samples_leaf':[1,3,5,7,9],
                'bootstrap':[False],
                'n_estimators':[100,200,300]}
gsRFC = GridSearchCV(RFC,param_grid=rf_param_grid, cv=2, scoring='r2', n_jobs=4, verbose=1)

gsRFC.fit(gridX, gridY)
RFC_best = gsRFC.best_estimator_

gsRFC.best_score_


# GRADIENT BOOST
GBC = GradientBoostingRegressor()
gb_param_grid={'learning_rate':[0.1, 0.075 , 0.05, 0.025 ,0.01],
              'max_depth':[2,4,6,8],
              'min_samples_leaf':[10,25,50,100,150],
              'max_features':[0.5,0.3,0.1]}

gsGBC = GridSearchCV(GBC,param_grid=gb_param_grid, cv=5, scoring='r2', n_jobs=4, verbose=1)

gsGBC.fit(gridX, gridY)

GB_best = gsGBC.best_estimator_
gsGBC.best_score_


def get_score_df(classifier):
    return pd.DataFrame({'Features':trainX.columns,'score':classifier.feature_importances_}).sort_values('score',ascending=False).reset_index(drop=True)

def get_graph(classifier_best):
    return sns.barplot(y = get_score_df(classifier_best).Features, x = get_score_df(classifier_best).score).set_title(str(classifier_best).split('(')[0])
get_score_df(RFC_best)
get_score_df(DTC_best)
get_score_df(GB_best)

get_graph(RFC_best)
get_graph(DTC_best)
get_graph(GB_best)
