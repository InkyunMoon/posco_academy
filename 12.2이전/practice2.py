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

# In[115]:


# 데이터 타입 확인
df_raw.info()


# In[ ]:





# In[116]:


#-------------------------데이터 드랍--------------------------------------
#@@@@@
df_raw.drop("SPEC", axis=1, inplace=True)
df_raw.drop("PLATE_NO", axis=1, inplace=True)
df_raw.drop("ROLLING_DATE", axis=1, inplace=True)
df_raw.drop("FUR_EXTEMP", axis=1, inplace=True)

df_raw
# In[117]:
# 이상치 확인을 위해 연속변수를 대상으로 boxplot 그리기
cont_values = ['PT_THK','PT_WDTH','PT_LTH','PT_WGT','FUR_HZ_TEMP','ROLLING_DESCALING',
               'FUR_HZ_TIME', 'FUR_SZ_TIME', 'FUR_SZ_TEMP', 'FUR_TIME','ROLLING_TEMP_T5']

for i in cont_values:
    plt.boxplot(df_raw[i])
    plt.title(i)
    plt.show()

# In[118]:
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


# In[119]:


# temp는 확실한 이상치이기 때문에
# 평균값으로 이상치를 대체해준다.

df_raw['ROLLING_TEMP_T5'].replace(0,int(df_raw['ROLLING_TEMP_T5'].mean()),inplace=True)

#@@@@@
# In[ ]:

# In[120]:
#################################################################################################################
# -------------------------------- FUR_SZ_TEMP - ROLLING_TEMP_T5  변수 추가 -----------------------------
df_raw['SUB_TEMP'] = df_raw['FUR_SZ_TEMP'] - df_raw['ROLLING_TEMP_T5']

df_raw[['FUR_SZ_TEMP','ROLLING_TEMP_T5','SUB_TEMP']]


# In[122]:


############### 파생변수 생성으로 인한 기존 변수 삭제 ####################3
df_raw.drop("FUR_SZ_TEMP", axis=1, inplace=True)


# In[123]:


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


# In[124]:


# rolling descaling이 홀수이면
# 항상 불량이기때문에 다른 변수의 영향을 조금더 보고자
# 홀수인 경우를 제거해준다.

df_raw = df_raw.query('ROLLING_DESCALING != 5& ROLLING_DESCALING != 7 & ROLLING_DESCALING != 9')


# In[125]:


# 홀수가 삭제되어 변형된 데이터 확인

plt.scatter(df_raw['ROLLING_DESCALING']+np.random.normal(0.1,0.03,len(df_raw)),
            df_raw['SCALE']+np.random.normal(0.1,0.03,len(df_raw)),
           c = df_raw['SCALE'])
plt.xlabel('ROLLING_DESCALING')
plt.show()


# In[126]:


# ROLLING DESCALING 열을 삭제해준다.

df_raw.drop('ROLLING_DESCALING', axis=1, inplace=True)


# In[127]:


# SCATTER를 그리기 위해 변형
df_raw['HSB'] = df_raw['HSB'].map(({'미적용':0, '적용':1}))


# In[128]:


# HSB와 SCALE의 상관관계 분석

plt.scatter(df_raw['HSB']+np.random.normal(0.1,0.03,len(df_raw)),
            df_raw['SCALE']+np.random.normal(0.1,0.03,len(df_raw)),
           c = df_raw['SCALE'])
plt.xlabel('HSB')
plt.show()


# In[129]:


# HSB가 미실시인 경우를 삭제해준다.

df_raw = df_raw.query('HSB == 1')


# In[130]:


# HSB 미실시인 것 제거 확인

plt.scatter(df_raw['HSB']+np.random.normal(0.1,0.03,len(df_raw)),
            df_raw['SCALE']+np.random.normal(0.1,0.03,len(df_raw)),
           c = df_raw['SCALE'])
plt.xlabel('HSB')
plt.show()


# In[140]:


# 이제 HSB 열을 제거해준다.
df_raw.drop('HSB',axis=1, inplace=True)


# In[141]:


df_raw.columns


# In[142]:


df_raw = df_raw.reset_index(drop=True)


# In[ ]:





# In[ ]:





# In[143]:


# 목표변수와 설명변수 데이터 분리 
df_x = df_raw.drop("SCALE", axis = 1, inplace = False) 
df_y = df_raw["SCALE"]


# train, test 데이터 분리

df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(df_x, df_y, test_size = 0.3, random_state = 1234)

print(df_train_x.shape)
print(df_train_y.shape)
print(df_test_x.shape)
print(df_test_y.shape)


# ### 7:3 / 8:2 중에 7:3으로 나눈이유
# 데이터수가 많으면 train데이터에 더 많이 할당하는데 
# 데이터가 많지 않으므로 7:3 으로 나눴다.

# In[144]:


# ---------------스케일 변환 대상 변수- 선택 ----------------------

df_num = df_x.select_dtypes(exclude = "object")


df_char = df_x.select_dtypes(include = "object")

df_num.head()

# In[145]:

df_char.info()
df_char.head()


# In[146]:


# ----------------------숫자 변수------스케일 변환 -------------------------



v_feature_name = df_num.columns

scaler=StandardScaler()
df_scaled = scaler.fit_transform(df_num)
df_scaled = pd.DataFrame(df_scaled, columns=v_feature_name)
df_scaled.head()


# In[147]:


# -------------------------string 변수 더미화----------------------------------------

df_dummy = pd.get_dummies(df_char)
df_dummy


# In[148]:


# --------------------- 데이터 합치기 num + char -------------------------------
df_x = pd.concat([df_dummy, df_scaled], axis=1)
df_x


# In[149]:


# -------------------train, test 데이터 분리--------------------------------------------

df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(df_x, df_y, test_size = 0.3, random_state = 1234)

print(df_train_x.shape)
print(df_train_y.shape)
print(df_test_x.shape)
print(df_test_y.shape)

#@@@@@@@@@@@@@@@@@@@@@@@@
# In[150]:


# ------------------- n_estimator 값 선정을 위한 --------oob error 확인----------------------------
oob_error = []

for v_n_estimator in range(1,151):
    rf = RandomForestClassifier(n_estimators=v_n_estimator, oob_score= True, random_state=1234)
    rf.fit(df_train_x, df_train_y)
    oob_error.append(1-rf.oob_score_)
    
    
plt.plot(range(1,151), oob_error)
plt.ylabel("oob_error")
plt.xlabel("n_estimator")

pd.Series(oob_error)[55:70]

# n_estimator = 57 로 결정
para_depth = [depth for depth in range(7,20)]
para_leaf = [leaf for leaf in range(1, 11)]

estimator = RandomForestClassifier(n_estimators=57, random_state=1234)

param_grid = {'max_depth':para_depth, 
              'min_samples_leaf':para_leaf,
              'criterion':['gini', 'entropy'],
              'bootstrap':[True, False]}

grid_dt = GridSearchCV(estimator, param_grid, scoring="accuracy", n_jobs=-1)
grid_dt.fit(df_train_x, df_train_y)

print("best estimator model: \n{}".format(grid_dt.best_estimator_))
print("\nbest parameter: \n{}".format(grid_dt.best_params_))
print("\nbest score:\n{}".format(grid_dt.best_score_.round(3)))
# In[152]:


# max _ depth

rf_final = RandomForestClassifier(bootstrap= False, criterion= 'gini', max_depth=13, min_samples_leaf= 1, random_state=1234)
rf_final.fit(df_train_x, df_train_y)

y_pred = rf_final.predict(df_test_x)

v_feature_name = df_train_x.columns

df_importance = pd.DataFrame()
df_importance['Feature'] = v_feature_name
df_importance['Importance'] = rf_final.feature_importances_

df_importance.sort_values('Importance',ascending = False, inplace=True)
df_importance.round(3)


# In[153]:


plt.figure(figsize=(50, 50))
plt.rc('font', size = 50)
df_importance.sort_values('Importance',ascending = True, inplace=True)
coordinates = range(len(df_importance))
plt.barh(y=coordinates,width=df_importance["Importance"])
plt.yticks(coordinates, df_importance['Feature'])
plt.xlabel('Importance of variables')
plt.ylabel('variables')


# In[82]:


sns.pairplot(df_x)


# In[154]:


######################### 변수중요도 분석을 통한 주요변수 추출 #####################################33


# --------------------- 기준 :  feature's importance 지수가 0.01 이상인 변수들만 추출 ------------------------------
importance_col = ['ROLLING_TEMP_T5',
                  'SUB_TEMP',
                  'HSB_1',
                  'HSB_0',
                  'FUR_HZ_TEMP',
                  'PT_WDTH',
                  'ROLLING_DESCALING',
                  'PT_THK','FUR_SZ_TIME','PT_LTH', 'STEEL_KIND_C0']

# In[155]:


#################### 데이터 드랍 ##################################


df_x_final = df_x.loc[:, importance_col]
############################ 모델링에 활용할 최종 설명변수 df_x_final ##########################
df_train_x_final


# In[156]:
# 데이터 분리
df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(df_x_final, df_y, test_size = 0.3, random_state = 1234)

print(df_train_x.shape)
print(df_train_y.shape)
print(df_test_x.shape)
print(df_test_y.shape)


# In[ ]:

# # Classification model

# ### ---1-------------------------gradient boosting-------------------------

# In[157]:
df_train_y = df_train_y.astype('int')
df_test_y = df_test_y.astype('int')

train_a = []; test_a = []

para_ntree = [n * 10 for n in range(1,16)]

for v_n_estimators in para_ntree:
    model = GradientBoostingClassifier(n_estimators = v_n_estimators,learning_rate=0.1, random_state=1234)
    model.fit(df_train_x, df_train_y)
    train_a.append(model.score(df_train_x, df_train_y))
    test_a.append(model.score(df_test_x, df_test_y))
    
df_score_n = pd.DataFrame()
df_score_n['n_estimator'] = para_ntree
df_score_n['TrainScore'] = train_a
df_score_n['testScore'] = test_a
df_score_n.round(3)


# In[159]:

plt.rc('font', size = 10)

plt.plot(para_ntree, train_a, linestyle = "-", label = "Train Score")
plt.plot(para_ntree, test_a, linestyle = "--", label = "Test Score")

plt.ylabel("score"); plt.xlabel("n_estimator")
plt.legend()

# ### n_estimator = 40으로 선정

# In[160]:


estimator = GradientBoostingClassifier(n_estimators= 40, learning_rate=0.1, random_state=1234)

para_depth = [depth for depth in range(1,11)]
para_leaf = [leaf*10 for leaf in range(1, 11)]


param_grid = {'max_depth':para_depth, 
              'min_samples_leaf':para_leaf,
              'learning_rate':[0.5,0.25,0.1,0.05]}

gb = GridSearchCV(estimator, param_grid, scoring="accuracy", n_jobs=-1)
gb.fit(df_train_x, df_train_y)

print("best estimator model: \n{}".format(gb.best_estimator_))
print("\nbest parameter: \n{}".format(gb.best_params_))
print("\nbest score:\n{}".format(gb.best_score_.round(3)))


# 
# ### ---2---------------------------- svm ------------------------------------------------------------

# In[161]:


estimator = SVC(random_state=1234)

para_c = [10 ** c for c in range(-2,2)]
para_gamma = [10 ** gamma for gamma in range(-2, 2)]


param_grid_svm = {'C' : para_c, "gamma" : para_gamma}


svm_ = GridSearchCV(estimator, param_grid_svm, scoring="accuracy", n_jobs=-1)
svm_.fit(df_train_x, df_train_y)

print("best estimator model: \n{}".format(svm_.best_estimator_))
print("\nbest parameter: \n{}".format(svm_.best_params_))
print("\nbest score:\n{}".format(svm_.best_score_.round(3)))


# ### ---3------------------- Decision Tree --------------------------------------

# In[162]:


estimator = DecisionTreeClassifier()

para_leaf = [n_leaf for n_leaf in range(1,20)]
para_split = [n_split*2 for n_split in range(2,20)]
para_depth = [depth for depth in range(1,11)]

param_grid = {'max_depth':para_depth,'min_samples_split':para_split, 'min_samples_leaf':para_leaf}
grid_dt = GridSearchCV(estimator, param_grid, scoring="accuracy", n_jobs=-1)
grid_dt.fit(df_train_x, df_train_y)

print("best estimator model: \n{}".format(grid_dt.best_estimator_))
print("\nbest parameter: \n{}".format(grid_dt.best_params_))
print("\nbest score:\n{}".format(grid_dt.best_score_.round(3)))


# In[163]:



tree_high = DecisionTreeClassifier(max_depth=6, min_samples_leaf=6, min_samples_split=6)
tree_high.fit(df_train_x, df_train_y)


export_graphviz(tree_high, out_file="decision_tree.dot", class_names=['불량', '양품'],
               feature_names=importance_col, impurity=True, filled=True, )


with open("decision_tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


# 트리 결과 이미지로 저장
dot = graphviz.Source(dot_graph)
dot.render(filename='decision_tree')


# ### ---4------------------------ Random Forest --------------------------

# In[164]:


# n_estimators 선정

oob_error = []

for v_n_estimators in range(1,151):
    rf = RandomForestClassifier(n_estimators = v_n_estimators,oob_score = True,random_state = 1234)
    rf.fit(df_train_x,df_train_y)
    oob_error.append(1-rf.oob_score_)
    


# In[165]:


plt.plot(range(1,151),oob_error)
plt.ylabel('oob error')
plt.xlabel('n_estimators')


# In[166]:


estimator = RandomForestClassifier(n_estimators=40,random_state=1234)

para_leaf = [n_leaf for n_leaf in range(1,20)]
para_split = [n_split*2 for n_split in range(2,20)]
para_depth = [depth for depth in range(1,11)]

param_grid = {'max_depth':para_depth,'min_samples_split':para_split, 'min_samples_leaf':para_leaf}
grid_dt = GridSearchCV(estimator, param_grid, scoring="accuracy", n_jobs=-1)
grid_dt.fit(df_train_x, df_train_y)

print("best estimator model: \n{}".format(grid_dt.best_estimator_))
print("\nbest parameter: \n{}".format(grid_dt.best_params_))
print("\nbest score:\n{}".format(grid_dt.best_score_.round(3)))


# In[168]:


# tree_high = RandomForestClassifier(max_depth=4, min_samples_leaf=2, min_samples_split=8)
# tree_high.fit(df_train_x, df_train_y)

# estimator = tree_high.estimator_[2]

# export_graphviz(estimator, out_file="RandomForest_tree.dot", class_names=['불량', '양품'],
#                feature_names=importance_col, impurity=True, filled=True, )


# with open("RandomForest_tree.dot") as f:
#     dot_graph = f.read()
# display(graphviz.Source(dot_graph))


# # 트리 결과 이미지로 저장
# dot = graphviz.Source(dot_graph)
# dot.render(filename='RandomForest_tree')


# ### ---5----------------------- 인공신경망 ----------------------------------

# In[169]:


estimator = MLPClassifier(random_state=1234)

para_solver = ['lbfgs','sgd','adam']
para_function = ['logistic','tanh','relu']
para_hidden = [hidden * 20 for hidden in range(2,11)]
para_batch = [20* batch for batch in range(3, 10)]

param_grid = {'hidden_layer_sizes':(para_hidden, para_hidden),
              'activation':para_function, 
              'solver':para_solver,
              'batch_size': para_batch}

mlp = GridSearchCV(estimator, param_grid, scoring="accuracy", n_jobs=-1)
mlp.fit(df_train_x, df_train_y)

print("best estimator model: \n{}".format(mlp.best_estimator_))
print("\nbest parameter: \n{}".format(mlp.best_params_))
print("\nbest score:\n{}".format(mlp.best_score_.round(3)))


# ### ---6--------------------로지스틱 회귀분석 ------------------------------

# In[170]:


for i in df_train_x.columns:
    print(i +"+")


# In[171]:
## 목적변수와 설명변수 다시 합쳐야 로지스틱회귀분석 할수있어서 합침. 

df_train = pd.concat([df_train_y, df_train_x], axis=1)

print(df_train.head())
# In[172]:


log_model = Logit.from_formula("""SCALE ~ 
                                    ROLLING_TEMP_T5+
                                    SUB_TEMP+
                                    PT_THK+
                                    FUR_SZ_TIME+
                                    FUR_HZ_TEMP+
                                    PT_WDTH+
                                    PT_LTH+
                                    PT_WGT+
                                    FUR_TIME+
                                    FUR_HZ_TIME""", df_train)

log_result = log_model.fit()

print(log_result.summary())

# In[173]:


log_coef = pd.DataFrame({"Coef": log_result.params.values[1:]},
                       index = log_model.exog_names[1:])

log_coef.plot.barh(y = "Coef", legend = False)


# In[ ]:




