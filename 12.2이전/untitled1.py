import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, minmax_scale, robust_scale

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve,GridSearchCV

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from statsmodels.api import Logit

from statsmodels.api import Logit
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.tree import export_graphviz
import graphviz

from sklearn.decomposition import PCA
import statsmodels.api as sm

# import warnings
# warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('font', family = 'MalgunGothic')

# # 데이터 전처리
# In[113]:
path = 'C:/Users/moon/Documents/posco_academy/practice_data/3_bigdata/'
df_raw = pd.read_csv(path + "SCALE불량.csv", engine ="python", encoding = "CP949")

# 스트링 인트로 변환 
df_raw['SCALE'] = df_raw['SCALE'].map(({'불량':0, '양품':1}))

df_raw['FUR_NO_ROW'] = df_raw['FUR_NO_ROW'].map(({1:'1번', 2:'2번'}))

# In[114]:
# 결측치 파악
df_raw.isnull().sum().sum()


# 데이터 타입 확인
df_raw.info()






#-------------------------데이터 드랍--------------------------------------
#@@@@@
df_raw.drop("SPEC", axis=1, inplace=True)
df_raw.drop("PLATE_NO", axis=1, inplace=True)
df_raw.drop("ROLLING_DATE", axis=1, inplace=True)
df_raw.drop("FUR_EXTEMP", axis=1, inplace=True)

df_raw
# 이상치 확인을 위해 연속변수를 대상으로 boxplot 그리기
cont_values = ['PT_THK','PT_WDTH','PT_LTH','PT_WGT','FUR_HZ_TEMP','ROLLING_DESCALING',
               'FUR_HZ_TIME', 'FUR_SZ_TIME', 'FUR_SZ_TEMP', 'FUR_TIME','ROLLING_TEMP_T5']

for i in cont_values:
    plt.boxplot(df_raw[i])
    plt.title(i)
    plt.show()

from scipy import stats

# 3씨그마 이후를 이상치라고 판단
zscore_threshold = 3

cont_list = ['PT_THK', 'PT_WDTH', 'FUR_HZ_TIME', 'FUR_SZ_TIME',
             'ROLLING_TEMP_T5']

for i in cont_list:
    df_temp = df_raw[i]
    print(i)
    print(df_temp[(np.abs(stats.zscore(df_temp)) > zscore_threshold)].values)
    print()


# temp는 확실한 이상치이기 때문에
# 평균값으로 이상치를 대체해준다.

df_raw['ROLLING_TEMP_T5'].replace(0,int(df_raw['ROLLING_TEMP_T5'].mean()),inplace=True)

###
# -------------------------------- FUR_SZ_TEMP - ROLLING_TEMP_T5  변수 추가 -----------------------------
df_raw['SUB_TEMP'] = df_raw['FUR_SZ_TEMP'] - df_raw['ROLLING_TEMP_T5']

df_raw[['FUR_SZ_TEMP','ROLLING_TEMP_T5','SUB_TEMP']]



############### 파생변수 생성으로 인한 기존 변수 삭제 ####################3
df_raw.drop("FUR_SZ_TEMP", axis=1, inplace=True)



### 연속형 설명변수들과 목표변수와 scatter 

plt.rc('font', size = 10)

for i in df_raw.columns:
    try:
        plt.scatter(df_raw[i]+np.random.normal(0.1,0.03,len(df_raw)),
                    df_raw['SCALE']+np.random.normal(0.1,0.03,len(df_raw)),
                   c = df_raw['SCALE'])
        plt.xlabel(i)
        plt.show()
    except Exception:
        pass



# rolling descaling이 홀수이면
# 항상 불량이기때문에 다른 변수의 영향을 조금더 보고자
# 홀수인 경우를 제거해준다.

df_raw = df_raw.query('ROLLING_DESCALING != 5& ROLLING_DESCALING != 7 & ROLLING_DESCALING != 9')


# 홀수가 삭제되어 변형된 데이터 확인

plt.scatter(df_raw['ROLLING_DESCALING']+np.random.normal(0.1,0.03,len(df_raw)),
            df_raw['SCALE']+np.random.normal(0.1,0.03,len(df_raw)),
           c = df_raw['SCALE'])
plt.xlabel('ROLLING_DESCALING')
plt.show()


# ROLLING DESCALING 열을 삭제해준다.

df_raw.drop('ROLLING_DESCALING', axis=1, inplace=True)



# SCATTER를 그리기 위해 변형
df_raw['HSB'] = df_raw['HSB'].map(({'미적용':0, '적용':1}))


# HSB와 SCALE의 상관관계 분석

plt.scatter(df_raw['HSB']+np.random.normal(0.1,0.03,len(df_raw)),
            df_raw['SCALE']+np.random.normal(0.1,0.03,len(df_raw)),
           c = df_raw['SCALE'])
plt.xlabel('HSB')
plt.show()


# HSB가 미실시인 경우를 삭제해준다.

df_raw = df_raw.query('HSB == 1')



# HSB 미실시인 것 제거 확인

plt.scatter(df_raw['HSB']+np.random.normal(0.1,0.03,len(df_raw)),
            df_raw['SCALE']+np.random.normal(0.1,0.03,len(df_raw)),
           c = df_raw['SCALE'])
plt.xlabel('HSB')
plt.show()


# 이제 HSB 열을 제거해준다.
df_raw.drop('HSB',axis=1, inplace=True)



df_raw.columns



df_raw = df_raw.reset_index(drop=True)


# 목표변수와 설명변수 데이터 분리 
df_x = df_raw.drop("SCALE", axis = 1, inplace = False) 
df_y = df_raw["SCALE"]


# train, test 데이터 분리
df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(df_x, df_y, test_size = 0.3, random_state = 1234)

print(df_train_x.shape)
print(df_train_y.shape)
print(df_test_x.shape)
print(df_test_y.shape)

# DT
DTC = DecisionTreeClassifier()
DTC_param_grid = {'max_depth':[4,6,8,10,12],
                 'min_samples_leaf':[3,5,7,9,11,15],
                 'min_samples_split':[5,10,15,20,25,35]}
DTC = GridSearchCV(DTC, param_grid = DTC_param_grid, cv = kfold, scoring = 'r2', verbose =1, n_jobs = 4)
DTC.fit(df_train_x, df_train_y)

DTC_best = DTC.best_estimator_
DTC.best_score_

# Random Forest Classifier
RFC = RandomForestClassifier()
rf_param_grid = {'max_depth':[4,6,8,10,12],
                'max_features':[1,3,10],
                'min_samples_split':[2,3,5,10,15],
                'min_samples_leaf':[1,3,5,10,15],
                'bootstrap':[False],
                'n_estimators':[100,300,500],
                'criterion':['gini']}

RFC = GridSearchCV(RFC, param_grid = rf_param_grid, cv = kfold, scoring = 'accuracy', n_jobs = 4, verbose = 1)
RFC.fit(df_train_x, df_train_y)
RFC_best = RFC.best_estimator_
RFC.best_score_


# GBC
GBC = GradientBoostingClassifier()
gb_param_grid={'loss':['deviance'],
              'n_estimators':[100,200,300],
              'learning_rate':[0.1,0.05,0.01,0.005],
              'max_depth':[4,8,12],
              'min_samples_leaf':[100,150,300],
              'max_features':[0.3,0.1]
              }

GBC = GridSearchCV(GBC,param_grid=gb_param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1)
GBC.fit(df_train_x, df_train_y)
GBC_best = GBC.best_estimator_
GBC.best_score_

# # SVC
# SVMC = SVC(probability=True)
# svc_param_grid = {'kernel': ['rbf'], 
#                   'gamma': [ 0.001, 0.01, 0.1, 1], # 마진 설정시 이상치를 얼마나 허용할 것인지
#                   'C': [1, 10, 50, 100,200,300, 1000]} # 결정 경계를 얼마나 유연하게 설정할 것인지

# SVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

# SVMC.fit(df_train_x, df_train_y)

# SVMC_best = SVMC.best_estimator_

# # Best score
# SVMC.best_score_

# # learning curve
# def plot_learning_curve(estimator, title, X,y,ylim=None,cv=None,n_jobs=-1, train_sizes=np.linspace(0.1,1,5)):
#     plt.figure(figsize=(6,3))
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel('Training examples')
#     plt.ylabel('Score')
#     train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
    
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color = 'r')
    
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = 'g')
#     plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label = 'Training score')
#     plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', label = 'Cross-validation score')
    
#     plt.legend(loc='best')
#     return plt
# # 선택 필요
# # g = plot_learning_curve(RFC.best_estimator_,'RF learning curves', df_train_x, df_train_y, cv=kfold)
# # g = plot_learning_curve(SVMC.best_estimator_,'SVC learning curves', df_train_x, df_train_y, cv=kfold)
# # g = plot_learning_curve(GBC.best_estimator_,'Gradient Boosting learning curves', df_train_x, df_train_y, cv=kfold)

def get_score_df(classifier):
    return pd.DataFrame({'Features':df_train_x.columns,'score':classifier.feature_importances_}).sort_values('score',ascending=False).reset_index(drop=True)

def get_graph(classifier_best):
    return sns.barplot(y = get_score_df(classifier_best).Features, x = get_score_df(classifier_best).score).set_title(str(classifier_best).split('(')[0])

get_score_df(RFC_best)
get_score_df(GBC_best)
get_score_df(DTC_best)

get_graph(RFC_best)
get_graph(GBC_best)
get_graph(DTC_best)