'''
Train 데이터셋에서
스팸 문장들 중에 가장 많이 나온 단어 100개
정상 문장들 중에 가장 많이 나온 단어 100개 뽑기
&
컨셉넷 돌려서 신조어인지 아닌지 미리 판별 해놓기
'''
import pandas as pd
from collections import Counter
import csv
import requests

def save_list_to_csv(data, filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def is_neologism(word):
    api_url = f"http://api.conceptnet.io/c/ko/{word}"
    response = requests.get(api_url)
    
    flag = 0
    if response.status_code == 200:
        data = response.json()
        if len(data['edges']) == 0:
            print(f"{word}는 신조어입니다.")
            flag = 1
        else:
            print(f"{word}는 신조어가 아닙니다.")
            flag = 0
    else:
        print("ConceptNet API 요청에 실패했습니다.")
        flag = -1
    return flag

train_data = pd.read_csv("../data/mecab_train_x.csv")

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

spam_neo = []; spam_not_neo = []; spam_api_error = []
for i in range(len(spam)):
    tmp = is_neologism(spam[i])
    if tmp == 1: #신조어
        spam_neo.append(spam[i])
    elif tmp == 0: #신조어 아님
        spam_not_neo.append(spam[i])
    else: #api 오류
        spam_api_error.append(spam[i])

save_list_to_csv(spam_neo, "spam_neo.csv")
save_list_to_csv(spam_not_neo, "spam_not_neo.csv")
save_list_to_csv(spam_api_error, "spam_api_error.csv")



normal_words = word_list[1650:]
normal_words = [word for word in normal_words if len(word) > 1]
word_counts_normal = Counter(normal_words)

one_hunnit_normal_words = word_counts_normal.most_common(100)
normal = []
for i in range(len(one_hunnit_normal_words)):
    normal.append(one_hunnit_normal_words[i][0])

normal_neo = []; normal_not_neo = []; normal_api_error = []
for i in range(len(normal)):
    tmp = is_neologism(normal[i])
    if tmp == 1: #신조어
        normal_neo.append(normal[i])
    elif tmp == 0: #신조어 아님
        normal_not_neo.append(normal[i])
    else: #api 오류
        normal_api_error.append(normal[i])

save_list_to_csv(normal_neo, "normal_neo.csv")
save_list_to_csv(normal_not_neo, "normal_not_neo.csv")
save_list_to_csv(normal_api_error, "normal_api_error.csv")