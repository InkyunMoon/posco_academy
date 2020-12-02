import pandas as pd
from scipy.stats import chi2, chi2_contingency
import seaborn as sns

df = pd.read_csv('C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/tele_no_outlier.csv')
df.drop(df[df['group_job']=='미확인'].index, inplace = True)   
df.drop(df[df['is_married']=='미확인'].index, inplace = True)
df.drop(df[df['edu']=='미확인'].index, inplace = True)
# df.drop(df[df['is_holder']=='미확인'].index, inplace = True)
df.drop(df[df['is_mortgage']=='미확인'].index, inplace = True)


df.loc[df['age']>70,'age'] = 70
df.age = df.age.astype('object')
df.info()

# 직업별과거 전화횟수에 차이가 있는지?
df_left = df[['group_job','no_call_past']].groupby(by='group_job').sum()
df_right = df['group_job'].value_counts()

df_past_by_job = pd.merge(df_left, df_right, left_index = True, right_index = True, how = 'left')
(df_past_by_job['no_call_past'] / df_past_by_job['group_job'])

######### 코드가 날아갔다...(14:39)
# 직업과 대출 관련성 확인 - 연관 있다.
contin_job_loan = pd.crosstab(df.group_job, df.is_mortgage)
chi, p, dof, expected = chi2_contingency(contin_job_loan)
print(f"chi 스퀘어 값: {chi}",
      f"p-value (0.05): {p}",
      f"자유도 수: {dof}",
      f"기대값: \n{pd.DataFrame(expected)}",
      f"측정값: \n{contin_job_loan}", sep = "\n" )

# 결혼과 대출 관련성 확인 - 연관 있다.
contin_married_loan = pd.crosstab(df.is_married, df.is_mortgage)
chi, p, dof, expected = chi2_contingency(contin_married_loan)
print(f"chi 스퀘어 값: {chi}",
      f"p-value (0.05): {p}",
      f"자유도 수: {dof}",
      f"기대값: \n{pd.DataFrame(expected)}",
      f"측정값: \n{contin_married_loan}", sep = "\n" )

contin_married_loan = pd.crosstab(df.is_married, df.is_mortgage)
chi, p, dof, expected = chi2_contingency(contin_married_loan)
print(f"chi 스퀘어 값: {chi}",
      f"p-value (0.05): {p}",
      f"자유도 수: {dof}",
      f"기대값: \n{pd.DataFrame(expected)}",
      f"측정값: \n{contin_married_loan}", sep = "\n" )

# 직업&결혼과 대출 관련성 확인 - ?
df.columns

sns.barplot(data = df, x = age, y = ''