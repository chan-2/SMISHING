'''
Train 데이터셋에서
스팸 문장들 중에 가장 많이 나온 단어 100개
정상 문장들 중에 가장 많이 나온 단어 100개 뽑기
'''
import pandas as pd
from collections import Counter
import csv

def save_list_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


train_data = pd.read_csv("../data/mecab_train_x.csv")
#print(train_data)

train_data.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
train_data.rename(columns={'0': 'text'}, inplace=True)

word_list = [word.strip() for text in train_data['text'] for word in text.split(',')]

spam_words = word_list[:1650]
spam_words = [word for word in spam_words if len(word) > 1]
word_counts_spam = Counter(spam_words)

one_hunnit_spam_words = word_counts_spam.most_common(100)
spam = []
for i in range(len(one_hunnit_spam_words)):
    spam.append(one_hunnit_spam_words[i][0])
print("spam", spam)
save_list_to_csv(spam, 'spam_100.csv')

normal_words = word_list[1650:]
normal_words = [word for word in normal_words if len(word) > 1]
word_counts_normal = Counter(normal_words)

one_hunnit_normal_words = word_counts_normal.most_common(100)
normal = []
for i in range(len(one_hunnit_normal_words)):
    normal.append(one_hunnit_normal_words[i][0])
print("normal", normal)
save_list_to_csv(normal, 'normal_100.csv')