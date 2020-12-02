import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

df_spend = pd.read_csv('C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/spending_no_outlier.csv')
sns.barplot(data = df_spend, x = 'age', y = 'no_use')

sns.barplot(data = df_spend, x = 'age', y = 'amount_one')

sns.barplot(data = df_spend, x = 'age', y = 'amount')

# 40대 비지니스별 사용금액
g = sns.barplot(data = df_spend[df_spend['age'] == '40대'], x = 'business', y = 'amount')
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.set_title('40')

# 40대 여성 비지니스별 사용금액
g = sns.barplot(data = df_spend[(df_spend['age'] == '40대')&(df_spend['sex'] == '여성')], x = 'business', y = 'amount')
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.set_title('40_female')

# 40대 남성 비지니스별 사용금액
g = sns.barplot(data = df_spend[(df_spend['age'] == '40대')&(df_spend['sex'] == '남성')], x = 'business', y = 'amount')
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.set_title('40_male')

# 40대 **남녀** 비지니스별 사용금액
g = sns.barplot(data = df_spend[(df_spend['age'] == '40대')], x = 'business', y = 'amount', hue='sex')
g.set_xticklabels(g.get_xticklabels(), rotation=45)
g.set_title('40')

g = sns.barplot(data = df_spend[(df_spend['age'] == '40대')], x = 'month', y= 'amount', hue = 'sex')
g.set_title('40대 월별 결제액')

g = sns.barplot(data = df_spend[(df_spend['age'] == '40대')], x = 'month', y= 'amount_one', hue = 'sex')
g.set_title('40대 월별 1회당 결제액')

df_spend['month'].unique()

sns.boxplot(data = df_spend, x = 'no_use')

# 정규분포 3시그마로 이상치 탐지시 945부터 이상치로 간주
no_use_mean = (df_spend.no_use).mean()
no_use_std = np.std(df_spend.no_use)
df_spend[df_spend['no_use'] > no_use_mean + 3 * no_use_std].sort_values(by='no_use',ascending=False)['no_use']

# 박스플랏으로 no_use 이상치 탐지시 120부터 이상치로 간주
df_spend.no_use.describe()
q3_no_use = np.quantile(df_spend.no_use, 0.75)
q1_no_use = np.quantile(df_spend.no_use, 0.25)
iqr_no_use = q3_no_use - q1_no_use

df_spend[df_spend['no_use'] > q3_no_use + 1.5 * iqr_no_use].sort_values(by='no_use',ascending=False)['no_use']

# 정규분포 3시그마 데이터 처리 시작
no_use_mean = (df_spend.no_use).mean()
no_use_std = np.std(df_spend.no_use)
df_spend[(df_spend['no_use'] > no_use_mean + 3 * no_use_std)|(df_spend['no_use'] < no_use_mean - 3 * no_use_std)]

def get_index(df, col):
    mean_ = df[col].mean()
    std_ = np.std(df[col])
    return df[(df[col] > mean_ + 3*std_)|(df[col] < mean_ - 3* std_)].index

df_spend.drop(get_index(df_spend,'no_use'), inplace = True)
df_spend.drop(get_index(df_spend,'amount'), inplace = True)

# df_spend.loc[get_index(df_spend,'no_use'),:]

# csv로 저장
# df_spend.to_csv('C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/12.2_am_spending.csv', index = False, encoding = 'utf-8-sig')

df_spend = pd.read_csv('C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/spending_no_outlier(drop_no_use).csv')

# 위 데이터로 다시 시작, 플랏을 그려보기로 한다.

df_spend['card_type'] = df_spend['card_type'].astype('object')
df_spend.info()

# 40대 **남녀** 비지니스별 사용금액
def get_bar_business(age_group, x = 'business', y='amount', hue='sex'):
    g = sns.barplot(data = df_spend[(df_spend['age'] == age_group)], x = x, y = y, hue=hue)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    g.set_title(age_group)

# get_bar_business('40대')

def get_age_business(age_group):
    return df_spend[(df_spend['age'] == age_group)].groupby('business').amount.sum().sort_values(ascending=False)[:5].index

def get_age_business_cards(age_group, business):
    print(business)
    return df_spend[(df_spend['age'] == age_group)&(df_spend['business'] == business)].card_type.value_counts()

# 40대가 자주 소비하는 소비처 5곳 각각에서 자주 사용되는 5개의 카드
get_bar_business('40대')
get_age_business('40대')
get_age_business_cards('40대',get_age_business('40대')[0])[:5]
get_age_business_cards('40대',get_age_business('40대')[1])[:5]
get_age_business_cards('40대',get_age_business('40대')[2])[:5]

# 50대가 자주 소비하는 소비처 5곳 각각에서 자주 사용되는 5개의 카드
get_bar_business('50대')
get_age_business('50대')
get_age_business_cards('50대',get_age_business('50대')[0])[:5]
get_age_business_cards('50대',get_age_business('50대')[1])[:5]
get_age_business_cards('50대',get_age_business('50대')[2])[:5]

# 30대가 자주 소비하는 소비처 5곳 각각에서 자주 사용되는 5개의 카드
get_bar_business('30대')
get_age_business('30대')
get_age_business_cards('30대',get_age_business('30대')[0])[:5]
get_age_business_cards('30대',get_age_business('30대')[1])[:5]
get_age_business_cards('30대',get_age_business('30대')[2])[:5]

#####
list_age = df_spend['age'].unique().tolist()
list_business = df_spend['business'].unique().tolist()
df_result = pd.DataFrame()
for i in list_business:
    cond1 = (df_spend['business']==i)
    table1 = df_spend.loc[cond1]
    
    for j in list_age:
        cond2 = (table1['age']==j)
        result1 = pd.pivot_table(data=table1,index='card_type',values='amount',aggfunc='sum').reset_index()
        result1['business'] = i
        result1['age'] = j
        df_n = result1.sort_values(by='amount',ascending=False).head(5)
        df_result = pd.concat([df_result, df_n])

df_result['Card_ID'] =df_result['card_type'] 
df_card = pd.read_csv('C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/000_Card_Data.csv')

df_result2 = pd.merge(df_result,df_card,on='Card_ID', how='left')

dict1 = {'자동차정비':'oil', '유통업':'comm', '레저업소':'travel', '음료식품':'shopping', '서적문구':'culture',
         '수리서비스':'oil', '요식업소':'food', '의복':'shopping',
       '보건위생':'life', '광학제품':'public', '신변잡화':'shopping', '연료판매':'oil',
       '의료기관':'life', '주방용품':'life', '직물':'shopping', '사무통신':'public',
       '문화취미':'culture', '가전':'life', '자동차판매':'oil', '가구':'life', '전기':'life'}
df_card['혜택분야'].unique()
df_result2['type1'] = df_result2['business'].replace(dict1)
cond3 = (df_result2['type1']==df_result2['혜택분야'])
df_result2['혜택매칭여부']= cond3
df_result2.to_excel('result3.xlsx')

cond3.value_counts()[1]/cond3.value_counts().sum()


labels = ['일치','불일치'] ## 라벨
frequency = [366,264] ## 빈도
 
fig = plt.figure(figsize=(8,8)) ## 캔버스 생성
fig.set_facecolor('white') ## 캔버스 배경색을 하얀색으로 설정
ax = fig.add_subplot() ## 프레임 생성
 
pie = ax.pie(frequency, ## 파이차트 출력
       startangle=90, ## 시작점을 90도(degree)로 지정
       counterclock=False, ## 시계 방향으로 그린다.
       autopct=lambda p : '{:.2f}%'.format(p), ## 퍼센티지 출력
       wedgeprops=dict(width=0.5) ## 중간의 반지름 0.5만큼 구멍을 뚫어준다.
       )
