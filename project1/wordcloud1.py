# !pip install wordcloud
import pandas as pd
from konlpy.tag import Okt, Mecab, Hannanum
# from ekonlpy.tag import Mecab
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

df_card = pd.read_csv('C:/Users/moon/Documents/posco_academy/project1/B4_카드_DataSet/000_Card_Data.csv')
credit_info = df_card[df_card['분류'] == '신용카드']['소개'].reset_index(drop=True)
debit_info = df_card[df_card['분류'] == '체크카드']['소개'].reset_index(drop=True)

credit_info.shape
debit_info.shape

# mecab = Mecab()
# mecab.morphs(credit_info[0])

okt= Okt()

hannanum = Hannanum()
hannanum.morphs(credit_info[0])

tokenizer = okt = Okt()
okt.morphs(credit_info[0])

# reg_ex = '[]'
# re.sub(pattern=reg_ex, repl=' ', string=' '.join(okt.morphs(df_credit.loc[row,'소개'])))

# re.sub(r'[.,!?"\':;~()]', '',  df_credit.loc[0,'소개']).split(' ')

# re.match

# ' '.join(okt.morphs(df_credit.loc[row,'소개']))


credit_tokens = []
df_credit = credit_info.to_frame()
for row in range(len(credit_info)):
    sent = re.sub(r'[.,!?"\':;~()&]', '', df_credit.loc[row,'소개']).replace('\n',' ')
    credit_tokens.append(sent.split(' '))
df_credit['tokens'] = credit_tokens

#### 예전 코딩 가져오기

temp = []
df_credit['tokens'] = 0
for idx, row in enumerate(df_credit['소개']):
    print(idx)
    temp.append(tokenizer.morphs(row))

def temp(list_):
    temp_list = []
    for x in list_:
        if len(x) < 2:
            continue
        else:
            temp_list.append(x)
    return temp_list

tokenizer.nouns(' '.join(df_credit['tokens'].loc[0,'소개']))
df_credit['tokens2'] = 0

temp = []
for idx, row in enumerate(df_credit['소개']):
    temp.append(tokenizer.nouns(' '.join(df_credit['tokens'][idx])))
df_credit['tokens2'] = temp

temp3 = []
for idx in range(1, len(df_credit['소개'])):
    df_credit.extend(df_credit['tokens2'][idx]) ####

common_words = Counter(temp3).most_common(300)

data = common_words
tmp_data = dict(data)
wordcloud = WordCloud(font_path = 'c:Windows/Fonts/malgun.ttf',
                      relative_scaling=0.2,
                      ).generate_from_frequencies(tmp_data)
plt.figure(figsize=(16,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()