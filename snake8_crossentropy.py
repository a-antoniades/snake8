import random
import curses
import time
from collections import namedtuple
import numpy as np
import shutil
import statistics
import math
import sys
import pandas as pd
import glob
import os.path
import sys
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from snake8_observations import SnakeGame as sn
from numpy import genfromtxt
model_path = []


def play_snake(n_mod, no_games=1000):
    snake8 = sn()
    if n_mod >= 1:
        model = Net()
        model_paths = glob.glob("/Users/antonis/Desktop/PycharmProjects/deep_snake/model/rnn/new/*")
        latest_model = max(model_paths, key=os.path.getctime)
        model.load_state_dict(torch.load(str(latest_model))['model_state_dict'])
    for _ in range(no_games):
        if snake8.collision() is True:
            break
        snake8.collision()
        if n_mod == 0:
            snake8.rand_keyz(random.randint(0, 3))
        else:
            snake8.torch_keyz(model)
        snake8.direction()
        snake8.snake_length()


def create_dataset(directory=
                   "/Users/antonis/Desktop/PycharmProjects/deep_snake/snake_data/snake8_observations_new"):
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, file))
            df = df.drop(columns=df.columns[0])
            df = df.to_numpy()
            yield df


def threshold(n=97.5):
    rewards = []
    for table in create_dataset():
        value = np.sum(table[:, -1], axis=0)
        rewards.append(value)
    threshold = np.nanpercentile(rewards, n, axis=0)
    thresh_mean = np.nanmean(rewards)
    return threshold, thresh_mean


def filter_batch(sum_reward):
    for table in create_dataset():
        value = np.sum(table[:, -1], axis=0)
        if value >= sum_reward:
            yield table
0

class Net(nn.Module):   # 8, 128, 4
    def __init__(self, obs_size=8, hidden_size=20, n_actions=4, p=0.75):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Dropout(p),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return F.softmax(x, dim=0)
        """for m in pred:
            max_m, max_m_index = torch.max(m, 0)
            move.append(max_m_index)  # use index to append actual one-hotencoded move
        return torch.tensor(move)"""

#   df: snake_pos: 0,1,2,3,4,5 apple_pos: 6,7 move: 8,9,10,11 reward: 12

def normalized(df, x_columns=8):
    mean = np.mean(df[:, 0:8])
    std = np.std(df[:, 0:8])
    dfXnormal = np.add(df[:, 0:8], ((-1)*mean))
    dfXnormal = np.dot(df[:, 0:8], 1/std)
    return dfXnormal

def train(n_mod):
    global model_path
    directory = "/Users/antonis/Desktop/PycharmProjects/deep_snake/model/rnn/new"
    files = glob.glob("/Users/antonis/Desktop/PycharmProjects/deep_snake/model/rnn/new/*")
    model = Net()
    model.train()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    if len(os.listdir(directory)) != 0:
        latest_model = max(files, key=os.path.getctime)
        checkpoint = torch.load(latest_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
    threshold_value, mean_value = threshold()
    filtered_batch = filter_batch(threshold_value)
    model_path = []
    losses = []
    loss = []
    number = 0
    for df in filtered_batch:
        X = normalized(df)
        X = np.random.shuffle(X)
        X = torch.from_numpy(X).type(torch.FloatTensor)
        Y = torch.from_numpy(df[:, 8]).type(torch.LongTensor)
        df_train = TensorDataset(X, Y)
        train_loader = DataLoader(df_train, len(X[:, 1]), shuffle=False)
        train_iter = iter(train_loader)
        observations, moves = train_iter.next()
        output = model(observations)
        loss = loss_fn(output, moves)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        number += 1
        print(loss, number)
        model_path = "/Users/antonis/Desktop/PycharmProjects/deep_snake/model/rnn/new/model_%d" % n_mod
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'loss': loss},
              model_path)
    return "/Users/antonis/Desktop/PycharmProjects/deep_snake/model/rnn/new/model_%d" % n_mod


def move_files(n_mod, new_directory, data_directory):
    files = os.listdir(new_directory)
    directory = data_directory + '_' + str(n_mod)
    os.mkdir(directory)
    for file in files:
        if file.endswith(".csv"):
            source = os.path.join(new_directory, file)
            shutil.move(source, directory)


def reinforcement(no_games=1000, no_trainings=10):
    new_model = "/Users/antonis/Desktop/PycharmProjects/deep_snake/model/rnn/new"
    move_model = "/Users/antonis/Desktop/PycharmProjects/deep_snake/model/rnn/old"
    new_snake = "/Users/antonis/Desktop/PycharmProjects/deep_snake/snake_data/snake8_observations_new"
    move_snake = "/Users/antonis/Desktop/PycharmProjects/deep_snake/snake_data/snake8_observations"
    for i in range(no_trainings):
        for n in range(no_games):
            play_snake(i)
        train(i)
        move_files(i, new_snake, move_snake)


"""
model = Net()
agent, loss = train(model)
plt.plot(loss)


pred1 = agent(torch.from_numpy(df3[1, 0:8]).type(torch.FloatTensor))
pred = agent.predict(agent(pred3))

pred3 = pred1.view(pred1.size(0), -1)
pred = agent.predict(torch.from_numpy(df3[1, 0:8]).type(torch.FloatTensor))
pred1 = agent(torch.from_numpy(df3[1, 0:8]).type(torch.FloatTensor))
pred2 = F.softmax(pred1, dim=0)

pred = agent.predict(agent(pred1))

#  python3 /Users/antonis/Desktop/PycharmProjects/deep_snake/snake8_crossentropy.py
"""

2
