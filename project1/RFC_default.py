# !pip install catboost
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from scipy.stats import chi2, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.svm import SVC
# from sklearn.
# from catboost import CatBoostClassifier

df_tele = pd.read_csv('C:/Users/moon/Documents/github/posco_academy/project1/B4_카드_DataSet/tele_for_modele.csv')

df_tele.rename(columns={'연령':'age','직업군':'group_job','결혼여부':'is_married',\
                        '학력':'edu','신용카드소지여부':'is_holder','주택대출여부':'is_mortgage',\
                            '개인대출여부':'is_personal_loan',\
                        '통화시간':'call_duration','연락시도횟수':'no_call_trial','과거통화횟수':'no_call_past',\
                            '계약여부':'is_contract','연락일자':'call_date','카드종류':'card_type'}, inplace = True)
temp_age = df_tele[df_tele['group_job'] == '은퇴'][['age','group_job']].sort_values('age').reset_index(drop=True)

temp_age.group_job.value_counts() 
df_tele.group_job.unique()
df_tele.age.describe()

# before
df_tele.edu.value_counts().plot(kind='bar', title = '학력')
df_tele.group_job.value_counts().plot(kind='bar', title = '직업군')
df_tele.is_married.value_counts().plot(kind='bar', title = '결혼여부')
df_tele.is_mortgage.value_counts().plot(kind='bar', title = '주택대출')
df_tele.is_personal_loan.value_counts().plot(kind='bar', title = '개인대출')


# df_tele = df_tele.drop(df_tele[df_tele.edu == '미확인'].index)

# # 학력 연관성
# contin_edu = pd.crosstab(df_tele.is_contract, df_tele.edu)
# chi, p, dof, expected = chi2_contingency(contin_edu)
# print(f"chi 스퀘어 값: {chi}",
#       f"p-value (0.05): {p}",
#       f"자유도 수: {dof}",
#       f"기대값: \n{pd.DataFrame(expected)}",
#       f"측정값: \n{contin_edu}", sep = "\n" )

# # 직업군 연관성
# contin_group_job = pd.crosstab(df_tele.is_contract, df_tele.group_job)
# chi, p, dof, expected = chi2_contingency(contin_group_job)
# print(f"chi 스퀘어 값: {chi}",
#       f"p-value (0.05): {p}",
#       f"자유도 수: {dof}",
#       f"기대값: \n{pd.DataFrame(expected)}",
#       f"측정값: \n{contin_group_job}", sep = "\n" )

# # 결혼여부 연관성
# contin_is_married = pd.crosstab(df_tele.is_contract, df_tele.is_married)
# chi, p, dof, expected = chi2_contingency(contin_is_married)
# print(f"chi 스퀘어 값: {chi}",
#       f"p-value (0.05): {p}",
#       f"자유도 수: {dof}",
#       f"기대값: \n{pd.DataFrame(expected)}",
#       f"측정값: \n{contin_is_married}", sep = "\n" )

# # 대출여부(주택) 연관성
# contin_is_mortgage = pd.crosstab(df_tele.is_contract, df_tele.is_mortgage)
# chi, p, dof, expected = chi2_contingency(contin_is_mortgage)
# print(f"chi 스퀘어 값: {chi}",
#       f"p-value (0.05): {p}",
#       f"자유도 수: {dof}",
#       f"기대값: \n{pd.DataFrame(expected)}",
#       f"측정값: \n{contin_is_mortgage}", sep = "\n" )

# # 대출여부(개인) 연관성
# contin_is_personal_loan = pd.crosstab(df_tele.is_contract, df_tele.is_personal_loan)
# chi, p, dof, expected = chi2_contingency(contin_is_personal_loan)
# print(f"chi 스퀘어 값: {chi}",
#       f"p-value (0.05): {p}",
#       f"자유도 수: {dof}",
#       f"기대값: \n{pd.DataFrame(expected)}",
#       f"측정값: \n{contin_is_personal_loan}", sep = "\n" )

drop_index = df_tele[(df_tele['group_job'] == '미확인')|(df_tele['is_married'] == '미확인')|\
        (df_tele['edu'] == '미확인')|(df_tele['is_mortgage'] == '미확인')].index
df_temp = df_tele.drop(drop_index)
df_tele.edu.value_counts().plot(kind='bar', title = '학력')

df_tele = df_tele.drop(df_tele[df_tele.is_personal_loan == '미확인'].index)

df_tele.edu.value_counts().plot(kind='bar', title = '학력')
df_tele.group_job.value_counts().plot(kind='bar', title = '직업군')
df_tele.is_married.value_counts().plot(kind='bar', title = '결혼여부')
df_tele.is_mortgage.value_counts().plot(kind='bar', title = '주택대출')
df_tele.is_personal_loan.value_counts().plot(kind='bar', title = '개인대출')

# 분석에 사용될 데이터프레임의 shape:(38138, 10)\
df_tele.drop('card_type', axis=1, inplace = True)
# df_temp.to_csv(r'C:\Users\moon\Documents\posco_academy\project1\B4_카드_DataSet\tele_for_model.csv', index = False, encoding = 'utf-8-sig')

scaler = StandardScaler()
scaled_2 = scaler.fit_transform(df_tele[['no_call_past','call_duration']])
df_scaled_2 = pd.DataFrame({'no_call_past':scaled_2[:,0],'call_duration':scaled_2[:,1]})
Y = df_tele['is_contract']
Y = Y.map({'yes':1,'no':0})
X = df_tele.drop(['is_contract','no_call_past','call_duration'], axis=1)
X = pd.concat([X,df_scaled_2], axis=1)
X = pd.get_dummies(X)

trainX,testX,trainY,testY = train_test_split(X,Y,test_size = 0.2, random_state = 1768, stratify=Y)

RFC = RandomForestClassifier()
RFC.fit(trainX,trainY)
GB = GradientBoostingClassifier()
GB.fit(trainX,trainY)
BAG = BaggingClassifier()
BAG.fit(trainX,trainY)

conf_RFC = confusion_matrix(RFC.predict(testX), testY)
conf_GB = confusion_matrix(GB.predict(testX), testY)
conf_BAG = confusion_matrix(BAG.predict(testX), testY)
con

def get_accu(conf_RFC):
    accu = (conf_RFC[0][0] + conf_RFC[1][1]) / (conf_RFC[0][0]+conf_RFC[1][1]+conf_RFC[0][1]+conf_RFC[1][0])
    return accu

get_accu(conf_RFC)
get_accu(conf_GB)
get_accu(conf_BAG)


# 붙여넣기
kfold = StratifiedKFold(n_splits = 5)
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, trainX, y = trainY, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

cv_res.sort_values(by='CrossValMeans', axis = 0, ascending=False)
g = sns.barplot('CrossValMeans','Algorithm',data=cv_res,palette='Set3',orient='h',xerr=cv_std)
g.set_xlabel('Mean Accuracy')
g = g.set_title('Cross validation scores')
cv_res.to_csv('C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/default_result.csv')
