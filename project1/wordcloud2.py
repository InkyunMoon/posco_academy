from konlpy.tag import Twitter
twitter = Twitter()

sentences_tag = []
for sentence in df_credit['소개']:
    morph = twitter.pos(sentence)
    sentences_tag.append(morph)
    print(morph)
    print('-' * 30)
 
noun_adj_list = []
for sentence1 in sentences_tag:
    for word, tag in sentence1:
        if tag in ['Noun','Adjective']:
            noun_adj_list.append(word)
                
counts = Counter(noun_adj_list)
tags = counts.most_common(30)
print(tags)

data = counts
tmp_data = dict(data)
wordcloud = WordCloud(font_path = 'c:Windows/Fonts/malgun.ttf',
                      relative_scaling=0.2,
                      ).generate_from_frequencies(tmp_data)
plt.figure(figsize=(16,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

from konlpy.tag import Twitter
twitter = Twitter()

sentences_tag = []
for sentence in df_credit['소개']:
    morph = twitter.pos(sentence)
    sentences_tag.append(morph)
    print(morph)
    print('-' * 30)
 
noun_adj_list = []
for sentence1 in sentences_tag:
    for word, tag in sentence1:
        if tag in ['Noun','Adjective']:
            noun_adj_list.append(word)
                
counts = Counter(noun_adj_list)
tags = counts.most_common(30)
print(tags)

data = counts
tmp_data = dict(data)
wordcloud = WordCloud(font_path = 'c:Windows/Fonts/malgun.ttf',
                      relative_scaling=0.2,
                      ).generate_from_frequencies(tmp_data)
plt.figure(figsize=(16,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

##################################
twitter = Twitter()

sentences_tag = []
for sentence in df_debit['소개']:
    morph = twitter.pos(sentence)
    sentences_tag.append(morph)
    print(morph)
    print('-' * 30)
 
noun_adj_list = []
for sentence1 in sentences_tag:
    for word, tag in sentence1:
        if tag in ['Noun','Adjective']:
            noun_adj_list.append(word)
                
counts = Counter(noun_adj_list)
tags = counts.most_common(30)
print(tags)

data = counts
tmp_data = dict(data)
wordcloud = WordCloud(font_path = 'c:Windows/Fonts/malgun.ttf',
                      relative_scaling=0.2,
                      ).generate_from_frequencies(tmp_data)
plt.figure(figsize=(16,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()