# snake8
Cross Entropy vs Deep Q-Learning VS Evolutionary algorithms for solving the game of Snake

Snake_8 = snake game with curses gui

Snake8_observations = snake game class that gathers data by playing randomly or using a trained model. saves each game's data as a numbered csv_file.

Snake8_survive_nn_binary = linear models that predict wether the next move will be succesful using obstacles (walls) as features.

Snake8_crossentropy = reinforcement learning algorithm - calls the Snake8_observations class, gathers episode data in real time about snake position, apple position, move, and policy reward. Trains algorithm and then uses first model to replay the game recursively. Utilizes yield functions to minimize memory overhead and enable assymatrical batching, where one batch is a single game.

