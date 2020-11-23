import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from collections import Counter as counter

path = 'C:/Users/moon/Documents/posco_academy/practice_data/3_bigdata/'
df_boston = pd.read_csv(path + 'BOSTON_HOUSING.csv')

df_boston.shape
df_boston.info()
df_boston.CHAS = df_boston.CHAS.astype('object')
df_boston.shape
# 데이터의 결측치를 확인한다.
df_boston.isnull().sum() # 결측치 없음

describe = df_boston.describe()

# 전체 변수에 대해서 박스플랏을 그려본다.
features = df_boston.columns
nrows, ncols = 7, 2

fig, axes = plt.subplots(nrows, ncols, figsize=(18,18))

sns.boxplot(x=features[13], data=df_boston)
    
# 상관관계 히트맵으로 종속변수 MEDV와 관련이 있어보이는 변수를 찾는다.
plt.figure(figsize = (15,12))
sns.heatmap(df_boston.corr(), annot=True, annot_kws={'size':18})
# 양의 상관관계: RM
# 음의 상관관계: LSTAT, INDUS, TAX, PTRATIO
# 상관관계가 높은 변수부터 낮은 변수들까지 하나한 종속변수와 비교해보도록 한다.

#0) MEDV
df_boston.MEDV.describe()
sns.distplot(df_boston.MEDV)
# 다수의 값이 50에 몰려있는 것을 확인할 수 있다.

def desc_(feature):
    return df_boston[feature].describe()

def dist_(feature, feature2='MEDV'):
    sns.distplot(df_boston[feature])
    
def scatter_(feature, feature2='MEDV'):
    sns.scatterplot(x = feature, y= feature2, data = df_boston)

figure, ((ax1,ax2,ax3,ax4,ax5,ax6,ax7), (ax8,ax9,ax10,ax11,ax12,ax13,ax14)) = plt.subplots(nrows=2, ncols=7, figsize=(35,10))
sns.distplot(df_boston[features[0]], ax=ax1)
sns.distplot(df_boston[features[1]], ax=ax2)
sns.distplot(df_boston[features[2]], ax=ax3)
sns.distplot(df_boston[features[3]], ax=ax4)
sns.distplot(df_boston[features[5]], ax=ax6)
sns.distplot(df_boston[features[6]], ax=ax7)
sns.distplot(df_boston[features[7]], ax=ax8)
sns.distplot(df_boston[features[8]], ax=ax9)
sns.distplot(df_boston[features[9]], ax=ax10)
sns.distplot(df_boston[features[10]], ax=ax11)
sns.distplot(df_boston[features[11]], ax=ax12)
sns.distplot(df_boston[features[12]], ax=ax13)
sns.distplot(df_boston[features[13]], ax=ax14)

# scatter plot
figure, ((ax1,ax2,ax3,ax4,ax5,ax6,ax7), (ax8,ax9,ax10,ax11,ax12,ax13,ax14)) = plt.subplots(nrows=2, ncols=7, figsize=(35,10))
sns.scatterplot(x = features[0], y= 'MEDV', data = df_boston, ax=ax1)
sns.scatterplot(x = features[1], y= 'MEDV', data = df_boston, ax=ax2)
sns.scatterplot(x = features[2], y= 'MEDV', data = df_boston, ax=ax3)
sns.scatterplot(x = features[3], y= 'MEDV', data = df_boston, ax=ax4)
sns.scatterplot(x = features[4], y= 'MEDV', data = df_boston, ax=ax5)
sns.scatterplot(x = features[5], y= 'MEDV', data = df_boston, ax=ax6)
sns.scatterplot(x = features[6], y= 'MEDV', data = df_boston, ax=ax7)
sns.scatterplot(x = features[7], y= 'MEDV', data = df_boston, ax=ax8)
sns.scatterplot(x = features[8], y= 'MEDV', data = df_boston, ax=ax9)
sns.scatterplot(x = features[9], y= 'MEDV', data = df_boston, ax=ax10)
sns.scatterplot(x = features[10], y= 'MEDV', data = df_boston, ax=ax11)
sns.scatterplot(x = features[11], y= 'MEDV', data = df_boston, ax=ax12)
sns.scatterplot(x = features[12], y= 'MEDV', data = df_boston, ax=ax13)
sns.scatterplot(x = features[13], y= 'MEDV', data = df_boston, ax=ax14)

figure, ((ax1,ax2,ax3,ax4,ax5,ax6,ax7), (ax8,ax9,ax10,ax11,ax12,ax13,ax14)) = plt.subplots(nrows=2, ncols=7, figsize=(35,10))
sns.boxplot(df_boston[features[0]], ax=ax1)
sns.boxplot(df_boston[features[1]], ax=ax2)
sns.boxplot(df_boston[features[2]], ax=ax3)
sns.boxplot(df_boston[features[3]], ax=ax4)
sns.boxplot(df_boston[features[5]], ax=ax6)
sns.boxplot(df_boston[features[6]], ax=ax7)
sns.boxplot(df_boston[features[7]], ax=ax8)
sns.boxplot(df_boston[features[8]], ax=ax9)
sns.boxplot(df_boston[features[9]], ax=ax10)
sns.boxplot(df_boston[features[10]], ax=ax11)
sns.boxplot(df_boston[features[11]], ax=ax12)
sns.boxplot(df_boston[features[12]], ax=ax13)
sns.boxplot(df_boston[features[13]], ax=ax14)

df_boston.RAD.value_counts()
df_boston.TAX.value_counts()
df_boston.INDUS.value_counts()
# -----연속형 -> 범주형 변환(4개(1,2,3,4) 구분)
bins_rad = []
RAD_small = df_boston.RAD[df_boston.RAD<20]
describe = df_boston.describe()

interval1 = np.quantile(RAD_small,0)
interval2 = np.quantile(RAD_small,0.33)
interval3 = np.quantile(RAD_small,0.66)
interval4 = np.quantile(RAD_small,1)
bins =[0,interval2,interval3,interval4,max(df_boston.RAD)]
# new_RAD = pd.cut(df_boston.RAD, bins=bins, labels = [1,2,3,4])
df_boston.RAD = pd.cut(df_boston.RAD, bins=bins, labels = [1,2,3,4])
# TAX
TAX_small = df_boston.TAX[df_boston.TAX<600]

interval2 = np.quantile(TAX_small,0.33)
interval3 = np.quantile(TAX_small,0.66)
interval4 = np.quantile(TAX_small,1)
bins_TAX = [0,interval2,interval3,interval4,max(df_boston.TAX)]
# new_TAX = pd.cut(df_boston.TAX, bins=bins_TAX, labels = [1,2,3,4])
df_boston.TAX = pd.cut(df_boston.TAX, bins=bins_TAX, labels = [1,2,3,4])

# 연속형 -> 범주형 변환(RAD, TAX외 변수용)
def con2cat(feature_name):
    i2 = np.quantile(df_boston[feature_name],0.25)
    i3 = np.quantile(df_boston[feature_name],0.50)
    i4 = np.quantile(df_boston[feature_name],0.75)
    i5 = np.quantile(df_boston[feature_name],1)
    bins = [min(df_boston[feature_name])-1,i2,i3,i4,i5]
    print(bins)
    new_feature = pd.cut(df_boston[feature_name], bins=bins, labels = [1,2,3,4])
    return new_feature

df_boston.INDUS = con2cat('INDUS')
df_boston.INDUS.value_counts()
# a = con2cat('PTRATIO')
# con2cat('ZN')

df_boston.info()

# 이상치 처리 함수 정의
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

df_boston.columns

outliers_idx = detect_outliers(df_boston, 2, ['MEDV', 'CRIM', 'ZN', 'INDUS','NOX', 'RM', 'AGE', 'DIS','PTRATIO', 'B', 'LSTAT'])
df_boston.loc[outliers_idx]
df_boston = df_boston.drop(outliers_idx, axis=0).reset_index(drop=True)

df_boston.CHAS = df_boston.CHAS.astype('object')
df_boston.RAD = df_boston.RAD.astype('object')
df_boston.TAX = df_boston.TAX.astype('object')
df_boston.INDUS = df_boston.INDUS.astype('object')

des = df_boston.describe()

df_boston.to_csv('C:/Users/moon/Documents/github/posco_academy/boston.csv', encoding='utf-8-sig', index= False)

df_boston.RAD.value_counts()
df_boston.TAX.value_counts()

# 범주화 체크
df_boston['RAD'].value_counts()
df_boston['TAX'].value_counts()
df_boston['INDUS'].value_counts()
sns.distplot(df_boston.RAD)
