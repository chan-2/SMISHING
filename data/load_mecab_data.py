import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


def load_mecab_dataset(test_ratio=0.2, erase_comma=True, random_state=None):

    with open('data/mecab_train_x.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        if erase_comma:
            texts = [(row[1] if row[1][:2] != "광고" else row[1][2:]).replace(",", "") for row in reader]
        else:
            texts = [row[1].replace("광고", "", 1) for row in reader]
    with open('data/mecab_train_y.csv', 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        labels = torch.FloatTensor(np.array([int(row[1]) for row in reader]).reshape(-1, 1)).detach()
    if random_state is None:
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_ratio)
    else:
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_ratio, random_state=random_state)
    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)
    return train_dataset, test_dataset

