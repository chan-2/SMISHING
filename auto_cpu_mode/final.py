import pandas as pd
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
import warnings

# 경고메세지 끄기
warnings.filterwarnings(action='ignore')

################################### prepare #############################
hi = input("스팸 검사를 실시할 메세지를 입력하세요 : ")
hi=hi.replace(',','')

temp=[]
temp.append(hi)
df=pd.DataFrame(temp, columns=["text"])
df.to_csv("./test_data.csv", index=True, index_label="id")

print("검사중입니다 ...")
time.sleep(2)

################################### test #############################
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text

text=[]
with open('./test_data.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for i in reader:
        if i[1]=="text":
            continue
        text.append(i[1])

test_dataset = TextDataset(text)
test_data_loader = DataLoader(test_dataset, batch_size=1,shuffle=False)

model=torch.load("./test_model.pt", map_location=torch.device('cpu'))
model.eval()

#texts=[]
y_pred=[]
with torch.no_grad():
    for x in test_data_loader:
        #texts.append(x[0])
        x=model.tokenize_texts(x[0])
        x=torch.tensor(x)
        if torch.round(model(x))==1:
            y_pred.append("삐빅! 스팸입니다.")
        else:
            y_pred.append("정상입니다.")

for i in range(len(y_pred)):
    print(str(y_pred[i]))