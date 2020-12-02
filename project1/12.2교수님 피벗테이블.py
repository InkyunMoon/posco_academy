import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

df_spend = pd.read_csv('C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/spending_no_outlier.csv')

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
        df_n = result1.sort_values(by='amount',ascending=False).head(5) # 상위 5개 amount를 추출
        # 상위 n개로 설정하지 않으면 모든 카드 선택 -> 정확도가 너무 낮음
        df_result = pd.concat([df_result, df_n])

df_result['Card_ID'] =df_result['card_type'] 
df_card = pd.read_csv('C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/000_Card_Data.csv')

df_result2 = pd.merge(df_result,df_card,on='Card_ID', how='left')

dict1 = {'자동차정비':'oil', '유통업':'comm', '레저업소':'travel', '음료식품':'food', '서적문구':'culture',
         '수리서비스':'oil', '요식업소':'food', '의복':'shopping',
       '보건위생':'life', '광학제품':'public', '신변잡화':'shopping', '연료판매':'oil',
       '의료기관':'life', '주방용품':'life', '직물':'shopping', '사무통신':'public',
       '문화취미':'culture', '가전':'life', '자동차판매':'oil', '가구':'life', '전기':'life'}

df_result2['type1'] = df_result2['business'].replace(dict1)
cond3 = (df_result2['type1']==df_result2['혜택분야'])
df_result2['혜택매칭여부']= cond3
df_result2.to_excel('result3.xlsx')

cond3.value_counts()[1]/cond3.value_counts().sum()

df_result2['type1'].value_counts()/ sum(df_result2['type1'].value_counts())

# df_result2[['age','business','card_type']]
# for business in list_business:
# cond1 = (df_result2['business'] == business)
#     for age in list_age:

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


df_result2['매칭 수'] = df_result2['혜택매칭여부'].replace({True:1, False:0})
pd.pivot_table(data=df_result2, index='business',values='매칭 수',aggfunc='sum').reset_index()
len(df_spend.business.unique())
