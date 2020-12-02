import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

path = 'C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/'
# df_card = pd.read_csv(path + '000_Card_Data.csv')
# df_tele = pd.read_csv(path + '000_Telemarketing_Data.csv')
df_spend = pd.read_csv(path + '000_Card_Spanding.csv')

df_spend.rename(columns={'사용일자':'date','지역':'gu','소비처':'dong','주소':'address',\
                         '소비처업종':'business','성별':'sex','연령':'age','사용횟수':'no_use',\
                             '사용금액':'amount','사용카드':'card_type'}, inplace = True)

    
df_spend.age.describe()
bins = []
    
df_spend.card_type = df_spend.card_type.astype('object')
df_spend['amount'] = df_spend['amount'] * 1000
df_spend['amount_one'] = df_spend.amount / df_spend.no_use

# 지역+소비처=주소임을 확인, 주소 컬럼 삭제
sum((df_spend.gu + ' ' + df_spend.dong) == df_spend.address) == df_spend.shape[0]
df_spend.drop('address', axis=1, inplace = True)

# gu, dong, address 드랍하기
df_spend.drop(['gu','dong','address'], axis=1, inplace = True)

# date컬럼 날짜형태로 변환
df_spend.date = pd.to_datetime(df_spend.date, format='%Y%m%d')
df_spend['year'] = df_spend.date.map(lambda x: x.year)
df_spend['month'] = df_spend.date.map(lambda x: x.month)
df_spend['day'] = df_spend.date.map(lambda x: x.day)
df_spend['weekday'] = df_spend['date'].map(lambda x: x.weekday())

df_spend.drop('date', inplace=True, axis=1)

# EDA 시작
# 연속형 변수부터 탐색해본다.
def get_outliers(df_col):
    q1 = np.quantile(df_col, 0.25)
    q3 = np.quantile(df_col, 0.75)
    step = (q3 - q1) * 1.5

    idx = df_col[(df_col < q1-step) | (df_col > q3+step)].index
    # print('펜스를 넘어가는 데이터의 개수: ',idx)
    return idx

def get_hist_box(df_col):
    count, bin_edges = np.histogram(df_spend.age,10)
    fig = plt.figure(figsize = (12,8))
    ax0 = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)
    df_col.plot(kind='hist', xticks=bin_edges, ax = ax0)
    df_col.plot(kind='box', vert = True, ax = ax1)
    plt.show()
    
df_spend.isnull().sum() # 결측치 없음.

# # ['gu.', 'dong.', 'business.', 'sex', 'age', 'no_use', 'amount', 'card_type', 'year', 'month', 'day']
# # gu
# df_spend.gu.unique() # 종로구, 노원구만 존재

# # dong
# df_spend.dong.unique() # 27개 동 존재

# business
len(df_spend.business.unique()) # 21개 비지니스 존재

# age
df_spend.age.describe() # 최소20, 최대 105, 중위수 45 평균 45
count, bin_edges = np.histogram(df_spend.age,10)
len(get_outliers(df_spend.age)) # 467개 존재

get_hist_box(df_spend.age)

# no_use, 사용 횟수
df_spend.no_use.describe() # 최소 -1 최대 20082 평균 82, 중위수 24
df_spend.drop(df_spend[df_spend['no_use'] < 0].index, inplace = True) # -1인 값은 삭제하도록 함

get_hist_box(df_spend.no_use)
len(get_outliers(df_spend.no_use)) # 22904 개의 펜스 넘는 값 존재 
len(get_outliers(df_spend.no_use)) / df_spend.shape[0] # 12퍼센트의 값이 펜스를 넘는다.

# amount # 단위: 1000원
df_spend.amount.describe() # 평균: 1702, 최소 2, 최대 2210348 (22억)
get_hist_box(df_spend.amount)
len(get_outliers(df_spend.amount)) / df_spend.shape[0]

## 성별에 따른 사용 카드 분포를 확인한다.
male_card_type = df_spend[df_spend['sex'] == '남성']['card_type'] # 남성의 카드 타입
female_card_type = df_spend[df_spend['sex'] == '여성']['card_type'] # 여성의 카드 타입

male_card_type.value_counts().plot(kind = 'bar')
female_card_type.value_counts().plot(kind = 'bar')

sns.catplot(x='card_type',hue='sex' ,data = df_spend)

## 보류
df_spend.sort_values(by='age', ascending=False).reset_index(drop=True, inplace = True)
