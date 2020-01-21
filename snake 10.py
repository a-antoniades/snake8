import random
import curses
import time
import numpy as np
import math
import sys
import pandas as pd
import os.path


class SnakeGame:
    def __init__(self):
        self.height = 20
        self.width = 20
        self.apple_pos = [random.randint(2, self.height - 2), random.randint(2, self.width - 2)]
        self.snake_head = [random.randint(5, self.height - 5), random.randint(5, self.width - 5)]
        self.snake_position = [[self.snake_head[1] + 1, self.snake_head[0]],
                               [self.snake_head[1] + 2, self.snake_head[0]],
                               [self.snake_head[1] + 3, self.snake_head[0]]]
        self.prev_snake_position = []
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.key = curses.KEY_RIGHT
        self.key2 = 0
        self.mass = []

        # input
        self.data = np.array([[1, 2, 3, 4, 5]])
        # self.obstacles_detect = np.array([[0,1], [0,1], [0,1]])
        self.obstacles_detect = np.array([0, 0, 0])  # 1, 2, 3
        self.angle_detect = []  # 4
        self.sugg_dir = []  # 5
        self.distance_detect = []   # 6

        # 1 obstacle on left, 2 obstacle in front, 3 obstacle on right
        # 4 angle, 5 sugg_directory

    # initializing screenxx
    def render_init(self):
        global win, sc
        sc = curses.initscr()
        curses.curs_set(0)
        curses.noecho()
        win = curses.newwin(self.height, self.width, 0, 0)
        win.keypad(1)
        win.addch(self.apple_pos[0], self.apple_pos[1], curses.ACS_DIAMOND)
        win.border(0)
        win.timeout(100)

    def collision_apple(self):
        allowed_values_x = list(range(1, self.height - 1))
        allowed_values_x.remove(self.snake_head[0])
        allowed_values_y = list(range(1, self.width - 1))
        allowed_values_y.remove(self.snake_head[1])
        for i in range(3):
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
        if self.snake_head[0] >= self.height-1 or self.snake_head[0] <= 0 or self.snake_head[1] >= self.width-1 or self.snake_head[1] <= 0:
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

    def rand_keyz(self):
        next_key = random.randint(0, 3)

        if next_key == -5:
            self.key2 = self.key2
        else:
            self.key2 = next_key

        if self.key2 == 1 and self.prev_button_direction != 1:
            self.button_direction = 0
        elif self.key2 == 0 and self.prev_button_direction != 0:
            self.button_direction = 1
        elif self.key2 == 2 and self.prev_button_direction != 2:
            self.button_direction = 3
        elif self.key2 == 3 and self.prev_button_direction != 3:
            self.button_direction = 2
        else:
            pass

    def direction(self): # direction 0=right, 1=left, 2=up, 3=down

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
            win.addch(self.apple_pos[0], self.apple_pos[1], curses.ACS_DIAMOND)

        else:
            self.snake_position.insert(0, list(self.snake_head))
            last = self.snake_position.pop()
            win.addch(last[0], last[1], ' ')

        win.addch(self.snake_position[0][0], self.snake_position[0][1], 'Α')

    def collision(self):
        if (self.snake_position[0][0] == 0 or
                self.snake_position[0][0] == self.width + 1 or
                self.snake_position[0][1] == 0 or
                self.snake_position[0][1] == self.height + 1 or
                self.snake_position[0] in self.snake_position[1:]):
            # sc.addstr(5, 5, 'your score is: ' * self.score)
            # save df
            df = pd.DataFrame(self.data)
            file = "/Users/antonis/Desktop/PycharmProjects/deep_snake/snake_data/snake_game.csv"
            for i in range(1000):
                if os.path.isfile(file):
                    i += 1
                else 
                    
                
            df.to_csv("/Users/antonis/Desktop/PycharmProjects/deep_snake/snake_data/snake_game*.csv", index=True)
            # exit
            sc.refresh()
            time.sleep(2)
            curses.endwin()
            sys.exit()
            # print(self.mass)
            # print(self.height, self.width)
        else:
            return False

    def obstacle_detection(self):
        self.obstacles_detect = np.array([0, 0, 0])

        if self.button_direction == 0:
            if self.snake_position[0][1] == self.width:
                self.obstacles_detect[1] = 1
            if self.snake_position[0][0] == self.height:
                self.obstacles_detect[0] = 1
            if self.snake_position[0][0] == 0:
                self.obstacles_detect[2] = 1
        if self.button_direction == 1:
            if self.snake_position[0][0] == self.height:
                self.obstacles_detect[2] = 1
            if self.snake_position[0][1] == self.width:
                self.obstacles_detect[1] = 1
            if self.snake_position[0][0] == 0:
                self.obstacles_detect[0] = 1
        if self.button_direction == 2:
            if self.snake_position[0][0] == self.height:
                self.obstacles_detect[1] = 1
            if self.snake_position[0][1] == self.width:
                self.obstacles_detect[2] = 1
            if self.snake_position[0][1] == 0:
                self.obstacles_detect[0] = 1
        if self.button_direction == 3:
            if self.snake_position[0][0] == 0:
                self.obstacles_detect[1] = 1
            if self.snake_position[0][1] == self.width:
                self.obstacles_detect[0] = 1
            if self.snake_position[0][0] == 0:
                self.obstacles_detect[2] = 1

    def angle_detection(self):
        try:
            opp = self.snake_position[0][0] - self.apple_pos[0]
            adj = self.snake_position[0][1] - self.apple_pos[0]
            opp_adj = math.tan(opp/adj)
            self.angle_detect = opp_adj/180
        except ZeroDivisionError:
            self.angle_detect = 0

    def suggested_direction(self):
        if self.prev_button_direction == self.button_direction:
            self.sugg_dir = 0
        elif self.button_direction == 0:
            if self.prev_button_direction == 2:
                self.sugg_dir = 1
            elif self.prev_button_direction == 3:
                self.sugg_dir = -1
            else: pass
        elif self.button_direction == 1:
            if self.prev_button_direction == 2:
                self.sugg_dir = -1
            elif self.prev_button_direction == 3:
                self.sugg_dir = 1
            else: pass
        elif self.button_direction == 2:
            if self.prev_button_direction == 0:
                self.sugg_dir = -1
            elif self.prev_button_direction == 1:
                self.sugg_dir = 1
            else: pass
        elif self.button_direction == 3:
            if self.prev_button_direction == 1:
                self.sugg_dir = -1
            elif self.prev_button_direction == 0:
                self.sugg_dir = 1
            else: pass

    def distance_detection(self):
        prev_distance = ((self.prev_snake_position[0][0] ** 2 + self.prev_snake_position[0][1] ** 2) ** (1/2))
        current_distance = ((self.snake_position[0][0] ** 2 + self.snake_position[0][1] ** 2) ** (1/2))
        displacement = current_distance - prev_distance
        if displacement <= 0:
            self.distance_detect = 1
        if displacement > 0:
            self.distance_detect = 0
        if self.collision() is True:
            self.distance_detect = -1

    def add_readings(self):
        # input
        self.obstacle_detection()   # self.obstacles_detect
        self.angle_detection()  # self.angle_detect
        self.suggested_direction()   # self.sugg_dir
        # output
        self.distance_detection()   # self.distance_detect
        # save table
        new_row = np.hstack((self.obstacles_detect, self.angle_detect, self.sugg_dir))
        self.data = np.vstack((self.data, new_row))


if __name__ == "__main__":
    game = SnakeGame()
    game.render_init()

    for _ in range(1000000):
        game.keyz()
        game.direction()
        game.snake_length()
        game.add_readings()
        game.collision()



# build dataset
# next step: train neural network to learn how to survive. use regression. need new 4 column
# dataframe
