import torch
import random
import numpy as np
from collections import deque
from game import Game
from model import Linear_QNet, QTrainer
from utils import plot
from car import Car

class Agent:
    def __init__(self, input_shape) -> None:
        self.n_games = 0
        self.lr = 0.0005
        self.gamma = 0.99 # discount rate
        self.n_actions = 5
        self.action_space = [i for i in range(self.n_actions)]
        self.epsilon = 1.00 # randomness
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.10
        self.batch_size = 512
        self.mem_size = 25000
        self.memory = deque(maxlen=self.mem_size) # popleft()
        print("INITIALIZING MODEL", input_shape)

        self.model_eval = Linear_QNet(input_shape, self.n_actions, self.batch_size)
        self.model_target = Linear_QNet(input_shape, self.n_actions, self.batch_size)


        self.trainer = QTrainer(self.model_target, self.model_eval, self.lr, self.gamma)

    def get_state(self, game, car):
        return game.get_state(car)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor([state], dtype=torch.float32)
            actions = self.model_eval(state)
            action = torch.argmax(actions).item()
        print("ACTION", action)
        return action
    
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = float("-inf")
    car = Car()
    game = Game()
    agent = Agent(input_shape = game.get_state(car).shape[0])

    while True:
        game.run(car)

        state_old = agent.get_state(game, car)
        # get move
        final_move = agent.get_action(state_old)
        # get new state
        reward, done, score = game.update(final_move, car)
        car.draw_coin_vector(game.screen, game.coins)
        car.generate_sensors(game.screen, game.lines)
        car.draw(game.screen)
        state_new = agent.get_state(game, car)
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        game.flip()
        
        if car.dead:
            # train long memory
            print("RESETTING")
            car.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model_eval.save()
            
            print("Game", agent.n_games, "Score", score, "Record", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            print("ALL RESET")
            
if __name__ == "__main__":
    train()