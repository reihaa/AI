import numpy as np
import random
from tqdm import tqdm


initial_state = ('a', 'a', 'a', 'b', 'a', 'b', 'b', '.', 'b')
final_state = ('.', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a')


def is_neighbor(i, j):
    x1, y1 = i // 3, i % 3
    x2, y2 = j // 3, j % 3
    return (x1 == x2 and abs(y1 - y2) == 1) or (y1 == y2 and abs(x1 - x2) == 1)


class Solver:
    def __init__(self):

        self.exploration_rate = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.01

        self.num_episodes = 10000
        self.max_steps_per_episodes = 100
        self.learning_rate = 0.5
        self.discount_rate = 0.99

        self.q_table = np.array(np.zeros([9, 9]))

        self.rewards_all_episodes = list()

    def run(self):
        for episode in tqdm(range(self.num_episodes)):
            state = list(initial_state)
            route = [7]
            current_position = route[0]
            current_episode_reward = 0
            for step in range(self.max_steps_per_episodes):
                if random.uniform(0, 1) > self.exploration_rate:
                    next_position = np.argmax(self.q_table[current_position])
                else:
                    playable_actions = []
                    for j in range(9):
                        if is_neighbor(current_position, j) and not (len(route) >= 2 and j == route[-2]):
                            playable_actions.append(j)
                    next_position = np.random.choice(playable_actions)

                state[current_position], state[next_position] = state[next_position], state[current_position]
                done = tuple(state) == final_state
                reward = 1 if done else 0  # whether if we are done or not

                self.q_table[current_position, next_position] = self.q_table[current_position, next_position] * (
                        1 - self.learning_rate) + self.learning_rate * (reward + self.discount_rate * np.max(
                    self.q_table[next_position]))

                route.append(next_position)
                current_position = next_position
                current_episode_reward += reward

                if done:
                    break

            self.exploration_rate = self.min_exploration_rate + (
                    self.max_exploration_rate - self.min_exploration_rate) * np.exp(
                -self.exploration_rate * episode)

            self.rewards_all_episodes.append(current_episode_reward)

    def result(self):
        partition_size = 1000
        rewards_per_thousand_episodes = \
            np.split(np.array(self.rewards_all_episodes), self.num_episodes / partition_size)
        for index, r in enumerate(rewards_per_thousand_episodes):
            print("{} : {} \n".format(index * partition_size, np.average(r)))
        for t,i in enumerate(self.q_table):
            print('\n',t,':')
            for j in range(3):
                for p in range(j * 3, (j + 1) * 3):
                    print(round(i[p]*10,4), end=" ,")
                print()



solver = Solver()
solver.run()
solver.result()
