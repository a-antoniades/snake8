import numpy as np
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


# create dataset
def create_dataset(directory):
    counter = 0
    df = np.empty([0, 6])
    for file in os.listdir(directory):
        if file.endswith(".csv") and df is not None:
            df_ = pd.read_csv(os.path.join(directory, file))
            df_ = df_.iloc[1:].to_numpy()
            df = np.vstack((df, df_))
        elif file.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, file))
            df = df.iloc[1:].to_numpy()
        counter += 1
        if counter >= 10000:
            break
    return df


# create dataset
df_ = create_dataset("/Users/antonis/Desktop/PycharmProjects/deep_snake/snake_data/df_snake_test")
df = pd.DataFrame(df_)
df_1 = df.loc[df[5] == 1]
df_2 = df.loc[df[5] == 0]

ratio_0_1 = len(df_2)/len(df_1)
n_remove = int(ratio_0_1*len(df_1))

df_1_remove = df_1[0:n_remove]

df = df_1_remove.append(df_2)
df = df.iloc[:, 1:6]
df_train = df.sample(frac=0.75)
df_test = df.sample(frac=0.25)
# df_encoded_labels = pd.get_dummies(df[5], prefix='l')
# df = df.iloc[:, 0:5].join(df_encoded_labels)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 3)  # input transformation
        self.fc2 = nn.Linear(3, 2)  # output transformation

    def forward(self, x):
        x = self.fc1(x)  # output of the first layer
        # x = F.tanh(x)
        # x = self.fc2(x)
        return x

    def predict(self, x):
        pred = F.softmax(self.forward(x), dim=1)
        ans = []
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(0)
        return torch.tensor(ans)


X = np.array(df_train.iloc[:, 0:4])
X = torch.from_numpy(X).type(torch.FloatTensor)
y = np.array(df_train.iloc[:, 4])
y = torch.from_numpy(y).type(torch.LongTensor)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
losses = []
for epoch in range(num_epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # if epoch % 10 == 0:
    print(loss)

def predict(x):
    x = torch.from_numpy()












