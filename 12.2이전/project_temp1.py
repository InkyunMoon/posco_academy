import sys

if len(sys.argv) == 1:
    file_name = 'students.txt'
else:
    file_name = input('enter filename: ')

with open(file_name, 'r') as f:
    students = f.readlines()

student_list = []
for student in students:
    student = student.split()
    name = student[1] + ' ' + student[2]
    student_list.append([int(student[0]),name,int(student[3]),int(student[4])])

def mean_grade(score_list,get=True): # 리스트 요소 하나씩(리스트형태)을 넣어준다.
    mean = (score_list[2] + score_list[3])/2
    if mean >= 90:
        grade = 'A'
    elif (mean >= 80) & (mean < 90):
        grade = 'B'
    elif (mean >= 70) & (mean < 80):
        grade = 'C'
    elif (mean >= 60) & (mean < 70):
        grade = 'D'
    else:
        grade = 'F'

    score_list.append(mean)
    score_list.append(grade)
    if get==True:
        return score_list

result_list = []
for student in student_list:
    result_list.append(mean_grade(student))

# 정렬
result_list.sort(key = lambda x: x[3], reverse=True)
# result_list

result_list.sort(key=lambda x:x[4], reverse=True)

# 함수 정의

def func_show():
    result_list.sort(key=lambda x:x[4], reverse=True)
    print('Student', 'Name','Midterm','Final','Average','Grade',sep='\t')
    print('----------------------------------------------------------')
    for row in result_list:
        
        print(row[0],row[1],row[2],row[3],row[4],row[5], sep='\t')

def func_search(ID,get=False, print=True):
    id_list = []
    for row in result_list:
        id_list.append(row[0])
    if ID not in id_list:
        print('NO SUCH PERSON.')
    else:
        idx =id_list.index(ID)
        chosen_row = result_list[idx]
#         print(chosen_row[0],chosen_row[1],chosen_row[2],chosen_row[3],chosen_row[4],chosen_row[5], sep='\t')
        if get == True:
            return [chosen_row[0],chosen_row[1],chosen_row[2],chosen_row[3],chosen_row[4],chosen_row[5]]

def func_searchgrade(GRADE):
    if GRADE not in ['A','B','C','D','F']:
        pass
    else:
        idx_list = []
        for idx, row in enumerate(result_list):
            if GRADE == row[5]:
                idx_list.append(idx)
                if len(idx_list) == 0:
                    print('NO RESULTS')
        for i in idx_list:
            func_search(result_list[i][0])

def func_remove(ID):
    if ID not in [id[0] for id in result_list]:
        print('NO SUCH PERSON')
    else:
        for idx, row in enumerate(result_list):
            if ID == row[0]:
                del result_list[idx]
                print('Student removed')

def func_add(ID):
    if ID in [id[0] for id in result_list]:
        print('ALREADY EXISTS.')
    else:
#         ID = int(input('Student ID: '))
        NAME = input('Name: ')
        MID = int(input('Midterm Score: '))
        FINAL = int(input('Final Score: '))
        
        result_list.append(mean_grade([ID,NAME,MID,FINAL]))
        print('Student added.')

def func_changescore(ID):
    if ID not in [id[0] for id in result_list]:
        print('NO SUCH PERSON')
    else:
        idx = [id[0] for id in result_list].index(ID)
    exam = input('Mid/Final?').lower()
    if exam not in ['mid','final']:
        pass
    else:
        new_score = int(input('Input new score: '))
    if abs(new_score) < 100:
        pass
    
    changed = mean_grade(func_search(ID, True))
    print(changed[0], changed[1], changed[2], changed[3], changed[4], changed[5], sep='\t')

    if exam == 'mid':
        changed = [changed[0], changed[1], new_score, changed[3], changed[4], changed[5]]
        
    elif exam == 'final':
        changed = [changed[0], changed[1], changed[2], new_score ,changed[4], changed[5]]
        
    result_list[idx] = mean_grade(changed[:4])
    print('Score changed')
    print(result_list[idx][0],result_list[idx][1],result_list[idx][2],result_list[idx][3],\
          result_list[idx][4],result_list[idx][5],sep='\t')

def quit():
    answer = input('Save data?[yes/no]')
    if answer == 'yes':
        result_list.sort(key=lambda x:x[4], reverse=True)
        temp = ''
        with open(path+'newStudent', 'w') as f:
            for line in result_list:
                for word in line:
                    temp += str(word) + '\t'
                temp += '\n'
            f.write(temp)
        print('File Saved.')
    else:
        print('Have a nice day.')
    pass

func_list = ['show','search','changescore','searchgrade','add','remove','quit']
while True:
    param = input('# ').lower()
# 1번 함수    
    if param == 'show':
        func_show()
  
    elif param == 'quit':
        quit()
        break

# 2번 함수
    elif param == 'search':
        ID = int(input('Student ID: '))
        func_search(ID)
    
# 3번 함수
    elif param == 'changescore':
        ID = int(input('Student ID: '))
        func_changescore(ID)

# 4번 함수
    elif param == 'add':
        ID = int(input('Student ID: '))
        func_add(ID)
        
# 5번 함수
    elif param == 'searchgrade':
        GRADE = input('Grade to search:: ').upper()
        func_searchgrade(GRADE)
        
# 6번 함수
    elif param == 'remove':
        ID = int(input('Student ID: '))
        func_remove(ID)
        
# 7번 함수

