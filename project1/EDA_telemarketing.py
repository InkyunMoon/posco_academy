import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib import font_manager, rc
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


path = 'C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/'
df_tele = pd.read_csv(path + '000_Telemarketing_Data.csv')
df_tele.rename(columns={'연령':'age','직업군':'group_job','결혼여부':'is_married',\
                        '학력':'edu','신용카드소지여부':'is_holder','주택대출여부':'is_mortgage',\
                            '개인대출여부':'is_personal_loan',\
                        '통화시간':'call_duration','연락시도횟수':'no_call_trial','과거통화횟수':'no_call_past',\
                            '계약여부':'is_contract','연락일자':'call_date','카드종류':'card_type'}, inplace = True)
#
df_tele.call_date = pd.to_datetime(df_tele.call_date, format='%Y-%m-%d')
df_tele['year'] = df_tele.call_date.map(lambda x: x.year)
df_tele['month'] = df_tele.call_date.map(lambda x: x.month)
df_tele['day'] = df_tele.call_date.map(lambda x: x.day)
df_tele['weekday'] = df_tele['call_date'].map(lambda x: x.weekday())
df_tele.drop(['call_date','p_days'], axis = 1, inplace = True)

df_tele.card_type = df_tele.card_type.astype('object')
df_tele.year = df_tele.year.astype('object')
df_tele.month = df_tele.month.astype('object')
df_tele.day = df_tele.day.astype('object')

df_tele.info()

# 직업군, 결혼여부, 학력, 신용카드 소지 여부, 주택 대출여부, 개인대출여부
# ['연령', '직업군', '결혼여부', '학력', '신용카드소지여부', '주택대출여부', '개인대출여부', 'contact',
#       '통화시간', '연락시도횟수', 'p_days', '과거통화횟수', '계약여부', '연락일자', '카드종류']

# 연령
plt.figure(figsize=(20,15))
ax = sns.countplot(df_tele['연령'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

# 직업군
df_tele.group_job.value_counts()

# 결혼여부 is_married
df_tele.is_married.value_counts()

# 학력
df_tele.edu.value_counts()

# 신용카드 소지 여부 
df_tele.is_holder.value_counts()

# 주택대출여부
df_tele.is_mortgage.value_counts()

# 개인대출여부
df_tele.is_personal_loan.value_counts()

df_tele[(df_tele['is_mortgage']=='미확인') & (df_tele['is_personal_loan']=='미확인')] # 대출여부는 두 피쳐 모두 동시에 미확인임을 알 수 있다.

df_temp = df_tele[['group_job','is_married','edu','is_holder','is_mortgage','is_personal_loan']]

def con(num, equal = True):
    if (num == 1):     
        if (equal == True):
            return (df_temp['group_job'] == '미확인')
        else:
            return (df_temp['group_job'] != '미확인')
        
    elif (num == 2):     
        if (equal == True):
            return (df_temp['is_married'] == '미확인')
        else:
            return (df_temp['is_married'] != '미확인')
        
    elif (num == 3):
        if (equal == True):
            return (df_temp['edu'] == '미확인')
        else:
            return (df_temp['edu'] != '미확인')
        
    elif (num == 4):
        if (equal == True):
            return (df_temp['is_holder'] == '미확인')
        else:
            return (df_temp['is_holder'] != '미확인')
    elif (num == 5):
        if (equal == True):
            return (df_temp['is_mortgage'] == '미확인')
        else:
            return (df_temp['is_mortgage'] != '미확인')
        
    elif num == 6:
        if (equal == True):
            return (df_temp['is_personal_loan'] == '미확인')
        else:
            return (df_temp['is_personal_loan'] != '미확인')

con(1, False)
df_temp[df_temp['group_job'] != '미확인']

df_temp[con(6, False)]
df_temp[con(6, True)]

df_edu_mi = df_tele[con(1,False)&con(2,False)&con(3,True)&con(4,False)&con(1,False)&con(5,False)&con(6,False)][['group_job','is_married','edu','is_holder','is_mortgage','is_personal_loan']] # 결혼여부만 미확인인 경우
df_edu_   = df_tele[con(1,False)&con(2,False)&con(3,False)&con(4,False)&con(5,False)&con(6,False)][['group_job','is_married','edu','is_holder','is_mortgage','is_personal_loan']] # 모든 데이터가 미확인이 아닌 경우

df_edu_Y = df_edu_['edu']
df_edu_X = df_edu_.drop('edu', axis=1)

compare_edu_dropped = df_edu_mi.drop('edu',axis=1).drop_duplicates()

# for row in range(len(compare_edu_dropped)):
#     count = 0
#     for row2 in range(len((compare_edu_dropped.iloc[row,:] == df_edu_X))):
#         if (compare_edu_dropped.iloc[row,:] == df_edu_X).iloc[row2:,].sum() == 5:
#             count +=1


(compare_edu_dropped.iloc[0,:] == df_edu_X).iloc[0,:]
(compare_edu_dropped.iloc[0,:] == df_edu_X).iloc[0:]


# in df_edu_[['group_job','is_married','is_holder','is_mortgage','is_personal_loan']]





Y_married = data['is_married']
X_married = data.drop('is_married', axis=1)

Y_married = Y_married.map({'결혼':0,'미혼':1,'이혼':2})

X_married_dummy = pd.get_dummies(X_married)

trainX, testX, trainY, testY = train_test_split(X_married_dummy, Y_married, test_size = 0.2, stratify = Y_married)

kfold = StratifiedKFold(n_splits=5)

RFC = RandomForestClassifier()
# RFC.fit(trainX, trainY)
# accuracy_score(RFC.predict(testX),testY) # 68%
rf_param_grid = {'max_depth':[None],
                'max_features':[1,3,10],
                'min_samples_split':[2,3,10],
                'min_samples_leaf':[1,3,10],
                'bootstrap':[False],
                'n_estimators':[100,300],
                'criterion':['gini']}

gsRFC = GridSearchCV(RFC, param_grid = rf_param_grid, cv = kfold, scoring = 'accuracy', n_jobs = 4, verbose = 1)
gsRFC.fit(trainX, trainY)
RFC_best = gsRFC.best_estimator_
gsRFC.best_score_

gsRFC.predict(testX)

data.columns

### 결혼여부 모델로 예측해본 결과 정확도 70% 미만...

sns.boxplot(df_tele.no_call_trial)
count_trial, bin_edges_trial = np.histogram(df_tele.no_call_trial, 25)
df_tele.no_call_trial.plot(kind='hist', bins=bin_edges_trial)

sns.boxplot(df_tele.no_call_past)
count_past, bin_edges_past = np.histogram(df_tele.no_call_past, 25)
df_tele.no_call_past.plot(kind='hist', bins=bin_edges_past)

len(get_outliers(df_tele.no_call_trial)) # 2406개의 펜스 벗어나는 데이터
len(get_outliers(df_tele.no_call_past)) # 5625개의 펜스 벗어나는 데이터
# card_type은 이상치가 없음 카드 종류는 균일한 사용자수로 분리되어있다.

### 위 이상치 처리 보류......
df_tele.edu.value_counts()
df_tele[(df_tele['age'] < 20)&(df_tele['edu'] == '고졸')]
df_tele[(df_tele['age'] < 21)&(df_tele['edu'] == '대졸학사')]
df_tele[(df_tele['age'] < 24)]['group_job'].value_counts()
df_tele[(df_tele['group_job']=='은퇴')&(df_tele['age'] < 30)]

#
df_tele[['no_call_trial','is_contract']].groupby('is_contract').mean()
df_tele[['no_call_trial','month']].groupby('month').median()
# 겨울에 전화시도를 많이하는 경향 확인


###########################################
# 직업에 따른 전화 날짜(요일)에 대한 차이?
# 1) 직업에 따른 요일 차이 확인
df = df_tele[['group_job','is_contract','weekday']]
df = df.drop(df[df.group_job == '미확인'].index).reset_index(drop=True)
df.weekday = df.weekday.astype('object')

df_success = df[df.is_contract == 'yes'].reset_index(drop=True)
df_fail = df[df.is_contract == 'no'].reset_index(drop=True)

df[df.is_contract == 'yes']

df_success
df_fail
ct_success = pd.crosstab(df_success.group_job, df_success.weekday, margins=False).T
ct_fail = pd.crosstab(df_fail.group_job, df_fail.weekday, margins=False).T
ct = pd.crosstab(df.group_job, df.weekday, margins=False).T

ct_success_rate = ct_success / ct

job_list = ct_success.columns

ct_success['공무원'].plot(kind='bar')

for job in job_list: # 계약 성공한 사람들 요일별 표
    ct_success[job].plot(kind='bar')
    plt.title(job+' yes', fontsize = 40)
    plt.show()

for job in job_list: # 계약 실패한 사람들 요일별 표
    ct_fail[job].plot(kind='bar')
    plt.title(job+' no', fontsize = 40)
    plt.show()

###
for job in job_list: # 계약 실패한 사람들 요일별 표
    ct_success_rate[job].plot(kind='bar')
    plt.title(job, fontsize = 40)
    plt.yticks(fontsize = 25)
    plt.show()


df_tele.is_holder.value_counts()
is_holder_yes = df_tele[df_tele['is_holder'] == '있음']
