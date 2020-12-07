import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, minmax_scale, robust_scale, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from statsmodels.api import Logit
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn.decomposition import PCA
import statsmodels.api as sm
from collections import Counter as counter
from sklearn.preprocessing import LabelEncoder

plt.rcParams['axes.unicode_minus'] = False
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('font', family = 'MalgunGothic')

path = 'C:/Users/moon/Documents/posco_academy/practice_data/3_bigdata/'
df_raw = pd.read_csv(path + "SCALE불량.csv", engine ="python", encoding = "CP949")

df_raw['SCALE'] = df_raw['SCALE'].map(({'불량':0, '양품':1})).astype('object')
df_raw['FUR_NO_ROW'] = df_raw['FUR_NO_ROW'].astype('object')
df_raw['HSB'] = df_raw['HSB'].map(({'적용':1, '미적용':0})).astype('object')
# df_raw['WORK_GROUP']

df_raw.info()

desc = df_raw.describe()
df_raw.duplicated().sum()

df_raw.SPEC.value_counts()
# pd.options.display.max_columns = None

# 연속형 변수 탐색

# 상관계수 1제거
df_raw.drop("FUR_EXTEMP", axis=1, inplace=True)

df_contin = df_raw.select_dtypes(exclude='object')
df_contin.info()

plt.figure(figsize=(18,14))
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df_contin.corr(), dtype=bool))
sns.heatmap(df_contin.corr(), annot = True, mask=mask) # annot_kws={"size": 12}

# 이상치 확인을 위해 연속변수를 대상으로 boxplot 그리기
cont_values = df_contin.columns
# ['PT_THK','PT_WDTH','PT_LTH','PT_WGT','FUR_HZ_TEMP','ROLLING_DESCALING',
#                'FUR_HZ_TIME', 'FUR_SZ_TIME', 'FUR_SZ_TEMP', 'FUR_TIME', 'FUR_EXTEMP','ROLLING_TEMP_T5']

for i in cont_values:
    plt.boxplot(df_raw[i])
    plt.title(i, fontsize = 40)
    plt.show()

def detect_outliers(df, n, features):
    outlier_indices = []
    
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5*IQR
        
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        # outlier_list_col 은 조건에 맞는 인덱스 번호 리턴
        
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v>n)
    # 탐지된 이상치가 n개 이상인 샘플에 대한 인덱스 리스트를 만든다.
    return multiple_outliers

# temp는 확실한 이상치. 평균값으로 이상치를 대체해준다.
df_raw['ROLLING_TEMP_T5'].replace(0,int(df_raw['ROLLING_TEMP_T5'].mean()),inplace=True)
df_raw['ROLLING_TEMP_T5'].min() # 0값 제거 확인

outliers_idx = detect_outliers(df_raw, 1, df_contin.columns)
len(outliers_idx)
df_raw = df_raw.drop(outliers_idx, axis=0).reset_index(drop=True)

# 범주형 변수 처리
df_raw.drop("ROLLING_DATE", axis=1, inplace=True) #@@@@@
df_raw.PLATE_NO.value_counts()
df_raw.drop("PLATE_NO", axis=1, inplace=True)

df_raw.SPEC.value_counts()
df_raw.drop("SPEC", axis=1, inplace=True)


# 목표변수와 연속형 변수 시각화
# plt.rc('font', size = 45)
for i in df_raw.columns:
    print(i)
    try:
        plt.scatter(df_raw[i], df_raw['SCALE'], c = df_raw['SCALE'], s = 200)
        plt.title(i, fontsize = 40)
        plt.show()
    except Exception:
        pass

cat_values = df_raw.select_dtypes(include='object')
# HSB
plt.scatter(df_raw['HSB'].astype('int'),
                    df_raw['SCALE'], c = df_raw['SCALE'], s = 200)
plt.title(i, fontsize = 40)
plt.show()

# 파생변수 추가 -> FUR_SZ_TEMP - ROLLING_TEMP_T5  변수 추가
df_raw['SUB_TEMP'] = df_raw['FUR_SZ_TEMP'] - df_raw['ROLLING_TEMP_T5']
df_raw[['FUR_SZ_TEMP','ROLLING_TEMP_T5','SUB_TEMP']]

# 기존 변수 삭제
df_raw.drop("FUR_SZ_TEMP", axis=1, inplace=True)


# 모델링 시작
# 목표변수와 설명변수 데이터 분리 
df_x = df_raw.drop("SCALE", axis = 1, inplace = False) 
df_y = df_raw["SCALE"]

df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(df_x, df_y, test_size = 0.25, random_state = 1234)
# 데이터 수 많지 않으므로 0.75:0.25

df_num = df_x.select_dtypes(exclude = "object")
df_char = df_x.select_dtypes(include = "object")
df_char['STEEL_KIND'].value_counts()
df_char.head()

# 연속형 자료에 대해서 정규화를 진행한다.
v_feature_name = df_num.columns

scaler=StandardScaler()
df_scaled = scaler.fit_transform(df_num)
df_scaled = pd.DataFrame(df_scaled, columns=v_feature_name)
df_scaled.head()

# 범주형 변수에 대해서 더미화를 진행한 뒤, 두 데이터를 합친다.
df_dummy = pd.get_dummies(df_char)

df_x = pd.concat([df_dummy, df_scaled], axis=1)
df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(df_x, df_y, test_size = 0.3, random_state = 1234)
df_train_y = df_train_y.astype('int')
df_test_y = df_test_y.astype('int')
## 디폴트 세팅으로 모델 돌려보기

kfold = StratifiedKFold(n_splits = 3)

random_state = 1234
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
cv_results = []

for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, df_train_x, y = df_train_y, scoring = "accuracy", cv = kfold, n_jobs=4))
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","RandomForest","GradientBoosting","MultipleLayerPerceptron","KNeighbors","LogisticRegression"]})
cv_res = cv_res.sort_values(by='CrossValMeans', ascending=False).reset_index(drop=True).round(3)

g = sns.barplot('CrossValMeans','Algorithm',data=cv_res.sort_values('CrossValMeans',ascending = False),palette='Set3',orient='h',xerr=cv_std)
g.set_xlabel('Mean Accuracy')
g = g.set_title('Cross validation scores')

# KNeighbors 제외
# df_train_x_float = df_train_x.select_dtypes(include = 'float64')


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


plt.figure(figsize=(50, 50))
plt.rc('font', size = 50)
df_importance.sort_values('Importance',ascending = True, inplace=True)
coordinates = range(len(df_importance))
plt.barh(y=coordinates,width=df_importance["Importance"])
plt.yticks(coordinates, df_importance['Feature'])
plt.xlabel('Importance of variables')
plt.ylabel('variables')


sns.pairplot(df_x)

#변수중요도 분석을 통한 주요변수 추출


# 기준 :  feature's importance 지수가 0.01 이상인 변수들만 추출 ------------------------------
importance_col = ['ROLLING_TEMP_T5',
                  'SUB_TEMP',
                  'HSB_1',
                  'HSB_0',
                  'FUR_HZ_TEMP',
                  'PT_WDTH',
                  'ROLLING_DESCALING',
                  'PT_THK','FUR_SZ_TIME','PT_LTH', 'STEEL_KIND_C0']

#데이터 드랍

df_x_final = df_x.loc[:,importance_col]
#모델링에 활용할 최종 설명변수 df_x_final
# df_train_x_final

# 데이터 분리
df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(df_x_final, df_y, test_size = 0.3, random_state = 1234)

print(df_train_x.shape)
print(df_train_y.shape)
print(df_test_x.shape)
print(df_test_y.shape)

#Classification model
#gradient boosting

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

plt.rc('font', size = 10)

plt.plot(para_ntree, train_a, linestyle = "-", label = "Train Score")
plt.plot(para_ntree, test_a, linestyle = "--", label = "Test Score")

plt.ylabel("score"); plt.xlabel("n_estimator")
plt.legend()

#n_estimator = 40으로 선정
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

#svm
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


# Random Forest


# n_estimators 선정

oob_error = []

for v_n_estimators in range(1,151):
    rf = RandomForestClassifier(n_estimators = v_n_estimators,oob_score = True,random_state = 1234)
    rf.fit(df_train_x,df_train_y)
    oob_error.append(1-rf.oob_score_)
    


plt.plot(range(1,151),oob_error)
plt.ylabel('oob error')
plt.xlabel('n_estimators')


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


# tree_high = RandomForestClassifier(max_depth=4, min_samples_leaf=2, min_samples_split=8)
# tree_high.fit(df_train_x, df_train_y)

# estimator = tree_high.estimator_[2]

# export_graphviz(estimator, out_file="RandomForest_tree.dot", class_names=['불량', '양품'],
#                 feature_names=importance_col, impurity=True, filled=True, )


# with open("RandomForest_tree.dot") as f:
#     dot_graph = f.read()
# display(graphviz.Source(dot_graph))


# # 트리 결과 이미지로 저장
# dot = graphviz.Source(dot_graph)
# dot.render(filename='RandomForest_tree')

#인공신경망

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


#로지스틱 회귀분석

for i in df_train_x.columns:
    print(i +"+")

## 목적변수와 설명변수 다시 합쳐야 로지스틱회귀분석 할수있어서 합침. 

df_train = pd.concat([df_train_y, df_train_x], axis=1)

print(df_train.head())


log_model = Logit.from_formula("""SCALE ~ 
                                    ROLLING_TEMP_T5+
                                    SUB_TEMP+
                                    PT_THK+
                                    FUR_SZ_TIME+
                                    FUR_HZ_TEMP+
                                    PT_WDTH+
                                    PT_LTH+
                                    PT_WGT+
                                    FUR_HZ_TIME""", df_train)

log_result = log_model.fit()

print(log_result.summary())

log_coef = pd.DataFrame({"Coef": log_result.params.values[1:]},
                       index = log_model.exog_names[1:])

log_coef.plot.barh(y = "Coef", legend = False)


###

log_model2 = Logit.from_formula("""SCALE ~ 
                                    ROLLING_TEMP_T5+
                                    SUB_TEMP+
                                    PT_THK+
                                    FUR_SZ_TIME+
                                    
                                    PT_WDTH+
                                    PT_LTH+
                                    PT_WGT+
                                    FUR_TIME+
                                    FUR_HZ_TIME""", df_train)
log_result = log_model.fit()

print(log_result.summary())

log_coef = pd.DataFrame({"Coef": log_result.params.values[1:]},
                       index = log_model.exog_names[1:])

log_coef.plot.barh(y = "Coef", legend = False)
