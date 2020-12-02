import pandas as pd
import seaborn as sns
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


path = 'C:/Users/moon/Documents/github/posco_academy/project1/B4_카드_DataSet/'
df_spend = pd.read_csv(path + '12.2_am_spending.csv')

# 소비처 업종의 특성에 따른 요일별 소비패턴
''' Business 종류 확인
['자동차정비', '유통업', '레저업소', '음료식품', '서적문구', '수리서비스', '요식업소', '의복',
       '보건위생', '광학제품', '신변잡화', '연료판매', '의료기관', '주방용품', '직물', '사무통신',
       '문화취미', '가전', '자동차판매', '가구', '전기']
'''
b_list = df_spend.business.unique()

def by_weekday(business):
    plt.figure()
    g = sns.countplot(data = df_spend[df_spend['business'] == business], x = 'weekday')
    g.set_title(business)
    plt.show()

# temp = df_spend[df_spend.business == '요식업소']
# sns.barplot(data = temp, x = 'weekday', y = 'no_use')

# df_spend[(df_spend['business'] == '자동차판매')].weekday.value_counts()
# temp = df_spend[(df_spend['business'] == '자동차판매')&(df_spend['weekday'] == 6)]
# countplot은 행 개수를 센다. -> 카드 종류 등에 따라 Count 개수 변경됨. 단순히 판매가 줄거나 늘었다고 할 수 없을 듯...

# 21개의 그래프 생성
for business in b_list:
    by_weekday(business)
    print(business)
# 카운트 해주는 그래프 다시 그려보기
    
def by_weekday_bar(business):
    plt.figure()
    g = sns.barplot(data = df_spend[df_spend['business'] == business], x = 'weekday', y = 'no_use')
    g.set_title(business)
    plt.show()
    
for business in b_list:
    by_weekday_bar(business)

# 동별 분석
def by_dong_bar(dong):
    plt.figure()
    g = sns.barplot(data = df_spend[df_spend['dong'] == dong], x = 'business', y = 'no_use')
    g.set_title(dong)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    plt.show()
    
dong_list = df_spend.dong.unique()

for dong in dong_list:
    by_dong_bar(dong)

g = sns.countplot(data = df_spend[(df_spend['dong'] == '공릉1동')], x = 'business')
g.set_xticklabels(g.get_xticklabels(), rotation=45)

def by_age_bar_no_use(age):
    plt.figure()
    g = sns.barplot(data = df_spend[df_spend['age'] == age], x = 'business', y = 'no_use')
    g.set_title(age)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    plt.show()
    
age_list = df_spend.age.unique()

for age in age_list:
    by_age_bar_no_use(age)
    
def by_age_bar_amount(age):
    plt.figure()
    g = sns.barplot(data = df_spend[df_spend['age'] == age], x = 'business', y = 'amount')
    g.set_title(age)
    g.set_xticklabels(g.get_xticklabels(), rotation=45)
    plt.show()
    
for age in age_list:
    by_age_bar_amount(age)

len(age_list)
