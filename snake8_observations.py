import random
import curses
import time
import numpy as np
import math
import sys
import pandas as pd
import os.path
import glob
import shutil
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class SnakeGame:
    def __init__(self):
        # render
        self.height = 21
        self.width = 21
        self.apple_pos = [random.randint(2, self.height - 2), random.randint(2, self.width - 2)]
        self.snake_head = [random.randint(5, self.height - 5), random.randint(5, self.width - 5)]
        self.snake_position = [(self.snake_head[1] + 1, self.snake_head[0]),
                               (self.snake_head[1] + 2, self.snake_head[0]),
                               (self.snake_head[1] + 3, self.snake_head[0])]
        self.prev_snake_position = []
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.button_suggested = 1
        self.key = curses.KEY_RIGHT
        self.model_key = 1
        self.key2 = 1
        self.mass = []

        # pixels
        self.snake_map = np.empty([0, 10])
        self.reward = 1
        # left [1,0,0,0], right [0,1,0,0], up [0,0,1,0], down [0,0,0,1]
        self.rand_move = 1
        self.new_row = np.hstack((
            np.array(self.snake_position[0][0]), np.array(self.snake_position[0][1]),
            np.array(self.snake_position[1][0]), np.array(self.snake_position[1][1]),
            np.array(self.snake_position[2][0]), np.array(self.snake_position[2][1]),
            np.array(self.apple_pos[0]), np.array(self.apple_pos[1]),
            np.array(self.button_direction), np.array(self.reward)
        ))

        # input
        self.data = np.array([[1, 2, 3, 4, 5]])
        # self.obstacles_detect = np.array([[0,1], [0,1], [0,1]])
        self.obstacles_detect = np.array([0, 0, 0])  # 1, 2, 3
        self.angle_detect = []  # 4
        self.sugg_direct = []  # 5
        self.distance_detect = []
        self.survive_detect = []  # 6

        # model load

        # mode vectors
        self.choices = [-1, 0, 1]
        self.vertical_vector = np.array([[0, -1], [1, 0], [0, 1]])
        self.inverse_vertical_vector = np.dot(self.vertical_vector, -1)
        self.horizontal_vector = np.array([[1, 0], [0, 1], [-1, 0]])
        self.inverse_horizontal_vector = np.dot(self.horizontal_vector, -1)

        # 1 obstacle on left, 2 obstacle in front, 3 obstacle on right
        # 4 angle, 5 sugg_directectory

    def collision_apple(self):
        self.reward += 25
        allowed_values_x = list(range(1, self.height - 1))
        allowed_values_x.remove(self.snake_head[0])
        allowed_values_y = list(range(1, self.width - 1))
        allowed_values_y.remove(self.snake_head[1])
        for i in range(len(self.snake_position)):
            num_x = (self.snake_position[i][0])
            num_y = (self.snake_position[i][1])
            try:
                allowed_values_x.remove(num_x)
                allowed_values_y.remove(num_y)
            except ValueError:
                pass
        self.apple_pos = [random.choice(allowed_values_x), random.choice(allowed_values_y)]
        self.score = self.score + 1
        return self.apple_pos, self.score

    def collision_boundaries(self):
        if self.snake_head[0] >= self.height or self.snake_head[0] <= 1 or self.snake_head[1] >= self.width or \
                self.snake_head[1] <= 1:
            return 1
        else:
            return 0

    def collision_self(self):
        self.snake_head = self.snake_position[0]
        if self.snake_head in self.snake_position[1:-1]:
            return 1
        else:
            return 0

    def keyz(self):
        next_key = win.getch()

        if next_key == -1:
            self.key = self.key
        else:
            self.key = next_key

        if self.key == curses.KEY_LEFT and self.prev_button_direction != 1:
            self.button_direction = 0
        elif self.key == curses.KEY_RIGHT and self.prev_button_direction != 0:
            self.button_direction = 1
        elif self.key == curses.KEY_UP and self.prev_button_direction != 2:
            self.button_direction = 3
        elif self.key == curses.KEY_DOWN and self.prev_button_direction != 3:
            self.button_direction = 2
        else:
            pass

    def rand_keyz(self, button):  # direction 1=right, 0=left, 2=down, 3=up
        next_key = button
        self.reward = 1

        if next_key == -5:
            self.key2 = self.key2
        else:
            self.key2 = next_key

        if self.key2 == 1 and self.prev_button_direction != 1:
            self.button_direction = 0
            self.rand_move = 0
        elif self.key2 == 0 and self.prev_button_direction != 0:
            self.button_direction = 1
            self.rand_move = 1
        elif self.key2 == 2 and self.prev_button_direction != 2:
            self.button_direction = 3
            self.rand_move = 3
        elif self.key2 == 3 and self.prev_button_direction != 3:
            self.button_direction = 2
            self.rand_move = 2
        else:
            pass

    def torch_keyz(self, model):
        self.reward = 1
        # ouptut
        torch.no_grad()
        model.eval()
        input = torch.from_numpy(self.new_row[0:8]).type(torch.FloatTensor)
        output = model(input)
        pred = F.softmax(output, dim=0)
        pred = pred.data.numpy()
        # keys
        self.model_key = np.random.choice(len(pred), p=pred)
        '''max_val = max(pred)
        max_index = list(pred).index(max_val)
        self.model_key = max_index'''

        if self.model_key == 0 and self.prev_button_direction != 1:
            self.button_direction = 0
        elif self.model_key == 1 and self.prev_button_direction != 0:
            self.button_direction = 1
        elif self.model_key == 3 and self.prev_button_direction != 2:
            self.button_direction = 3
        elif self.model_key == 2 and self.prev_button_direction != 3:
            self.button_direction = 2
        else:
            pass

    def add_readings(self):
        # input
        self.suggested_direction()  # self.sugg_direct
        self.obstacle_detection()  # self.obstacles_detect
        # self.angle_detection()  # self.angle_detect
        # output
        # self.distance_detection()   # self.distance_detect
        # save table
        new_row = np.hstack((self.obstacles_detect, self.sugg_direct, self.survive_detect))
        self.data = np.vstack((self.data, new_row))


    def direction(self):  # direction 1=right, 0=left, 3=down, 2=up

        self.prev_button_direction = self.button_direction
        self.prev_snake_position = self.snake_position

        if self.button_direction == 1:
            self.snake_head[1] += 1
        elif self.button_direction == 0:
            self.snake_head[1] -= 1
        elif self.button_direction == 2:
            self.snake_head[0] += 1
        elif self.button_direction == 3:
            self.snake_head[0] -= 1

    def snake_length(self):
        self.prev_snake_position = self.snake_position
        if self.snake_head == self.apple_pos:
            self.apple_pos, self.score = self.collision_apple()
            self.snake_position.insert(0, list(self.snake_head))
            self.mass.append(self.apple_pos)
        else:
            self.snake_position.insert(0, list(self.snake_head))
            last = self.snake_position.pop()

    def obstacle_detection(self):  # direction 1=right, 0=left, 2=down, 3=up
        self.obstacles_detect = np.array([0, 0, 0])

        if self.button_direction == 0:
            if self.snake_position[0][1] >= self.width - 2:
                self.obstacles_detect[1] = 1
            if self.snake_position[0][0] >= self.height - 2:
                self.obstacles_detect[0] = 1
            if self.snake_position[0][0] == 1:
                self.obstacles_detect[2] = 1
        if self.button_direction == 1:
            if self.snake_position[0][0] >= self.height - 2:
                self.obstacles_detect[2] = 1
            if self.snake_position[0][1] >= self.width - 2:
                self.obstacles_detect[1] = 1
            if self.snake_position[0][0] <= 1:
                self.obstacles_detect[0] = 1
        if self.button_direction == 3:
            if self.snake_position[0][0] >= self.height - 2:
                self.obstacles_detect[1] = 1
            if self.snake_position[0][0] <= 1:
                self.obstacles_detect[1] = 1
            if self.snake_position[0][1] >= self.width - 2:
                self.obstacles_detect[2] = 1
            if self.snake_position[0][1] <= 1:
                self.obstacles_detect[0] = 1
        if self.button_direction == 2:
            if self.snake_position[0][0] <= 1:
                self.obstacles_detect[1] = 1
            if self.snake_position[0][1] >= self.width - 2:
                self.obstacles_detect[0] = 1
            if self.snake_position[0][0] >= self.height - 2:
                self.obstacles_detect[1] = 1
            if self.snake_position[0][1] <= 1:
                self.obstacles_detect[2] = 1

    def angle_detection(self):
        try:
            opp = self.snake_position[0][0] - self.apple_pos[0]
            adj = self.snake_position[0][1] - self.apple_pos[0]
            opp_adj = math.tan(opp / adj)
            self.angle_detect = opp_adj / 180
        except ZeroDivisionError:
            self.angle_detect = 0

    def suggested_direction(self):  # direction 1=right, 0=left, 2=down, 3=up
        if self.button_direction == self.button_suggested:
            self.sugg_direct = 0
        elif self.button_direction == 0:
            if self.button_suggested == 2:
                self.sugg_direct = -1
            elif self.button_suggested == 3:
                self.sugg_direct = 1
            else:
                self.sugg_direct = 0
        elif self.button_direction == 1:
            if self.button_suggested == 2:
                self.sugg_direct = 1
            elif self.button_suggested == 3:
                self.sugg_direct = -1
            else:
                self.sugg_direct = 0
        elif self.button_direction == 2:
            if self.button_suggested == 0:
                self.sugg_direct = -1
            elif self.button_suggested == 1:
                self.sugg_direct = 1
            else:
                self.sugg_direct = 0
        elif self.button_direction == 3:
            if self.button_suggested == 1:
                self.sugg_direct = 1
            elif self.button_suggested == 0:
                self.sugg_direct = -1
            else:
                self.sugg_direct = 0

        self.button_suggested = self.button_direction

    def distance_detection(self):
        prev_distance = ((self.prev_snake_position[0][0] ** 2 + self.prev_snake_position[0][1] ** 2) ** (1 / 2))
        current_distance = ((self.snake_position[0][0] ** 2 + self.snake_position[0][1] ** 2) ** (1 / 2))
        displacement = current_distance - prev_distance
        if displacement <= 0:
            self.distance_detect = 1
        if displacement > 0:
            self.distance_detect = 0
        if self.collision() is True:
            self.distance_detect = -1

    def add_readings(self):
        # input
        self.suggested_direction()  # self.sugg_direct
        self.obstacle_detection()  # self.obstacles_detect
        # self.angle_detection()  # self.angle_detect
        # output
        # self.distance_detection()   # self.distance_detect
        # save table
        new_row = np.hstack((self.obstacles_detect, self.sugg_direct, self.survive_detect))
        self.data = np.vstack((self.data, new_row))

    def collision(self):
        if (self.snake_position[0][0] == 0 or
                self.snake_position[0][0] == self.height - 1 or
                self.snake_position[0][1] == 0 or
                self.snake_position[0][1] == self.width - 1 or
                self.snake_position[0] in self.snake_position[1:]):
            # sc.addstr(5, 5, 'your score is: ' * self.score)
            # save df
            df = pd.DataFrame(self.snake_map)
            for i in range(100000):
                file = \
                    ('/Users/antonis/Desktop/PycharmProjects/deep_snake/snake_data/snake8_observations_new/snake%d.csv' % i)
                if os.path.isfile(file):
                    continue
                else:
                    df.to_csv(file, index=True)
                    break
            return True
                    # exit
            # print(self.mass)
            # print(self.height, self.width)
        else:
            self.survive_detect = 1
            self.pixels()

    def pixels(self):
        self.new_row = np.hstack((
                             np.array(self.snake_position[0][0]), np.array(self.snake_position[0][1]),
                             np.array(self.snake_position[1][0]), np.array(self.snake_position[1][1]),
                             np.array(self.snake_position[2][0]), np.array(self.snake_position[2][1]),
                             np.array(self.apple_pos[0]), np.array(self.apple_pos[1]),
                             np.array(self.button_direction), np.array(self.reward)
                            ))
        self.snake_map = np.vstack((self.snake_map, self.new_row))

    # minimizing loss functiom
    # loss = (required score) - (actual score) -> calculate using cross entropy or MSE


'''
if __name__ == "__main__":

    game = SnakeGame()

    for _ in range(1000000):
        if game.collision() is True:
            break
        game.collision()
        game.torch_keyz()
        game.direction()
        game.snake_length()
'''

# for i in {1..100}; do python3 /Users/antonis/Desktop/PycharmProjects/deep_snake/snake8_observations.py; done