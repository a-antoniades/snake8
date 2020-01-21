import random
import curses
import time
import numpy as np


class SnakeGame:
    def __init__(self):
        self.height = 20
        self.width = 20
        self.apple_pos = [random.randint(2, self.height - 2), random.randint(2, self.width - 2)]
        self.snake_head = [random.randint(5, self.height - 5), random.randint(5, self.width - 5)]
        self.snake_position = [[self.snake_head[1] + 1, self.snake_head[0]],
                               [self.snake_head[1] + 2, self.snake_head[0]],
                               [self.snake_head[1] + 3, self.snake_head[0]]]
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        self.key = curses.KEY_RIGHT
        self.mass = []
        self.data =

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

    def direction(self):

        self.prev_button_direction = self.button_direction

        if self.button_direction == 1:
            self.snake_head[1] += 1
        elif self.button_direction == 0:
            self.snake_head[1] -= 1
        elif self.button_direction == 2:
            self.snake_head[0] += 1
        elif self.button_direction == 3:
            self.snake_head[0] -= 1

    def snake_length(self):

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
            sc.addstr(5, 5, 'your score is: ' * self.score)
            sc.refresh()
            time.sleep(2)
            curses.endwin()
            print(self.mass)
            print(self.height, self.width)
            return

if __name__ == "__main__":
    game = SnakeGame()
    game.render_init()

    for _ in range(1000000):
        game.keyz()
        game.direction()
        game.snake_length()
        game.collision()






