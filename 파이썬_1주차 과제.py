import pandas as pd
import numpy as np
# import sys

path = 'C:/Users/moon/Documents/github/posco_academy/'
# file_name = sys.argv[1]
# if file_name == '':
    # file_name == 'students.txt'
stu_df = pd.read_csv(path + 'students.txt', sep='\t', header=None)

#학생이름 리스트
names = [x.split()[1].lower() for x in stu_df.iloc[:,1]]

#학번, 시험점수 리스트
id_scores = []
for i in range(len(stu_df)):
    id_scores.append(stu_df.iloc[i,[0,2,3]].tolist())

#이름 = 학번, 시험점수 리스트 할당
for idx, name in enumerate(names):
    globals()[name] = id_scores[idx]
    
stu_list = [globals()[name] for name in names]

#Average 컬럼 생성
stu_df['Average'] = (stu_df.iloc[:,2] + stu_df.iloc[:,3]) / 2

def get_grade(score):
    if score >= 90:
        return 'A'
    elif (score >= 80) & (score < 90):
        return 'B'
    elif (score >= 70) & (score < 80):
        return 'C'
    elif (score >= 60) & (score < 70):
        return 'D'
    else:
        return 'F'

# stu_df에 Grade
stu_df['Grade'] = stu_df.Average.apply(lambda x : get_grade(x))
stu_df.sort_values(by='Average', ascending=False, inplace = True)
stu_df.columns = ['Student','Name','Midterm','Final','Average','Grade']
stu_df.reset_index(drop=True, inplace = True)
stu_df

def func_show(idx=None):
    stu_df.Average = stu_df.Average.astype('float')
    result = stu_df.sort_values(by='Average', ascending=False)
    if idx == None:
        print(result)
    elif len(idx) > 0:
        print(result.loc[idx,:])
    
func_list = ['show','search','changescore','searchgrade','add','remove','quit']
while True:
    parameter = input('# ').lower()
    # show 기능
    if (parameter == 'show') & (parameter in func_list):
        # 소수점 첫째자리 명시 추후 추가
        # stu_df.Average = stu_df.Average.astype('float')
        # print(stu_df.sort_values(by='Average', ascending=False))
        func_show()
        
    elif (parameter == 'search') & (parameter in func_list):
        ID = int(input('Student ID: '))
        if ID not in list(stu_df.Student):
            print('NO SUCH PERSON.')
            break
        else:
            idx = stu_df.Student[stu_df.Student == ID].index[0]
            print(stu_df.iloc[idx,:])
    elif (parameter == 'changescore') & (parameter in func_list):
       ID = int(input('Student ID: '))
       if ID not in list(stu_df.Student):
           print('NO SUCH PERSON')
       exam = input('Mid/Final?')
       if exam not in ['mid','final']:
           exam = input('Mid/Final?')
       to_what = int(input('Input new score: '))
       if abs(to_what) > 100:
           pass
       else:
           idx = stu_df.Student[stu_df.Student == ID].index[0]
           print(stu_df.loc[idx,:].to_frame().T.reset_index(drop=True))
           if exam == 'mid':
               stu_df.loc[idx,'Midterm'] = to_what
               stu_df.loc[idx, 'Average'] = ((stu_df.loc[idx, 'Midterm'] + stu_df.loc[idx, 'Final']) / 2)
               stu_df.loc[idx, 'Grade'] = get_grade(stu_df.loc[idx, 'Average'])
           elif exam == 'final':
               stu_df.loc[idx, 'Final'] = to_what
               stu_df.loc[idx, 'Average'] = ((stu_df.loc[idx, 'Midterm'] + stu_df.loc[idx, 'Final']) / 2)
               stu_df.loc[idx, 'Grade'] = get_grade(stu_df.loc[idx, 'Average']) 
           print('Score changed')
           # print(stu_df.loc[idx,:].to_frame().T.reset_index(drop=True))
           func_show()
           
    elif (parameter == 'add') & (parameter in func_list):
        ID = int(input('Student ID: '))
        if ID in stu_df.Student:
            print('ALREADY EXISTS.')
        else:
            NAME = input('Name: ')
            MID = int(input('Midterm Score: '))
            FINAL = int(input('Final Score: '))
            AVERAGE = (MID + FINAL) / 2
            temp_df = pd.DataFrame({'Student':[ID],'Name':[NAME],'Midterm':[MID],'Final':[FINAL],'Average':[AVERAGE], 'Grade':[get_grade(AVERAGE)]})
            stu_df = pd.concat([stu_df,temp_df])
            stu_df.sort_values(by='Average', ascending=False, inplace = True)
            print('Student added.')
            
    elif (parameter == 'searchgrade') & (parameter in func_list):
        GRADE = input('Grade to search: ')
        if GRADE not in ['A','B','C','D','F']:
            print('NO RESULTS.')
        else:
            print(stu_df[stu_df.Grade == GRADE])
         
    elif (parameter == 'remove') & (parameter in func_list):
        ID = input('Student ID: ')
        if len(stu_df) == 0:
            print('List is empty')
        elif ID not in stu_df.Student:
            print('NO SUCH PERSON.')
        else:
            print('Student removed.')
            stu_df.drop(stu_df.Student[stu_df.Student == ID].index[0]).reset_index(drop=True,inplace=True)    
    
    elif (parameter == 'quit') & (parameter in func_list):
        answer = input('Save data?[yes/no]')
        if answer == 'yes':
            stu_df.sort_values(by='Average', inplace=True)
            np.savetxt('newStudents.txt', stu_df, fmt='%s', delimiter="\t")
            print('File name: %s' %('newStudents.txt'))
            break
        elif answer == 'no':
            break