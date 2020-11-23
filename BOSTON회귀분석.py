import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.api import qqplot, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 
from scipy.stats import randint

from sklearn.metrics import confusion_matrix 


# - MEDV : 주택가격(중앙값)
# - CRIM : 범죄율
# - ZN : 주거지 비율
# - INDUS : 비소매업 비율
# - CHAS : 강 조망 여부(1-조망,0-비조망)
# - NOX : 산화질소 농도
# - RM : 주거당 평균 객실 수
# - AGE : 노후 건물 비율
# - DIS : 중심지(노동센터) 접근 거리
# - RAD : 고속도로 접근 편이성 지수
# - TAX : 재산세율
# - PTRATIO : 학생당 교사 비율
# - B : 흑인 인구 비율
# - LSTAT : 저소득층 비율

# In[23]:


df_raw = pd.read_csv('C:/Users/moon/Documents/github/posco_academy/boston.csv',engine='python',encoding='CP949')

# In[24]:


df_raw.head()
df_raw.rename(columns={'癤풫EDV':'MEDV'}, inplace=True)


# In[25]:


df_raw.info()


# In[26]:


df_raw.CHAS = df_raw.CHAS.astype('object')
df_raw.RAD = df_raw.RAD.astype('object')
df_raw.TAX = df_raw.TAX.astype('object')

df_raw.info()

# In[7]:


# sns.pairplot(df_raw)


# In[27]:


#회귀모델 생성
reg_model=smf.ols(formula="MEDV ~ CRIM+ZN+INDUS+C(CHAS)+NOX+RM+AGE+DIS+C(RAD)+C(TAX)+PTRATIO+B+LSTAT", data=df_raw)
reg_result = reg_model.fit()
print(reg_result.summary())


# In[28]:


#다중공선성 확인
df_raw_x2 = df_raw.drop("MEDV",axis=1)[['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS','PTRATIO', 'B', 'LSTAT']]
df_raw_x2_const = add_constant(df_raw_x2)
df_vif = pd.DataFrame()
df_vif["variable"] = df_raw_x2_const.columns
df_vif["VIF"] = [variance_inflation_factor(df_raw_x2_const.values, i)                 for i in range(df_raw_x2_const.shape[1])]
df_vif.sort_values("VIF",inplace=True)
df_vif.round(3)


# VIF>10인 변수가 없음

# In[32]:


df_raw_x = df_raw.drop("MEDV",axis=1)
df_raw_y = df_raw["MEDV"]
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=5).fit(df_raw_x,df_raw_y)
selected_cols = df_raw_x.columns[rfe.support_]
removed_cols = df_raw_x.columns[~rfe.support_]
print("Selected Variances: {}".format(selected_cols))
print("Removed Variances: {}".format(removed_cols))


# In[33]:


rfe_reg_model = smf.ols(formula='MEDV~CHAS+NOX+RM+DIS+PTRATIO',data=df_raw)
rfe_reg_result = rfe_reg_model.fit()
print(rfe_reg_result.summary())


# In[34]:


#비표준화 회귀계수를 통해 변수 중요도 확인
df_reg_coef = pd.DataFrame({"Coef":rfe_reg_result.params.values[1:]},index=selected_cols)
df_reg_coef.plot.barh(y="Coef",legend=False)


# In[35]:


#표준화 회귀계수를 통해 변수 중요도 확인
scaler = StandardScaler()
cols=df_raw_x.columns
np_scaled = scaler.fit_transform(df_raw_x)
df_scaled = pd.DataFrame(np_scaled, columns=cols)
df_scaled["MEDV"] = df_raw["MEDV"]
reg_model_scaled = smf.ols(formula="MEDV~CHAS+NOX+RM+DIS+PTRATIO",data=df_scaled)
reg_result_scaled = reg_model_scaled.fit()
print(reg_result_scaled.summary())

df_reg_coef = pd.DataFrame({"Coef":reg_result_scaled.params.values[1:]}, index=selected_cols)
df_reg_coef.plot.barh(y= "Coef",legend=False)


# In[ ]:





# In[ ]:




