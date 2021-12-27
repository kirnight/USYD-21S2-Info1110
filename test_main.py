from game_engine import Engine
from gui import GUI
from player import Player

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define parameters
BATCH_SIZE = 32             # Training volume per batch
LR = 0.01                   # Learning rate
EPSILON = 0.9               # Epsilon Greedy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
N_ACTIONS = 5
ENV_A_SHAPE = 0      # to confirm the shape
game = Engine('examples/complexb.txt', Player, GUI)
s = game.state_tran()
N_STATES = len(s)


# Building neural network model，returning possible action
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization

        self.dense1 = nn.Linear(50, 50)
        self.dense1.weight.data.normal_(0, 0.1)

        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        for _ in range(5):
            x = self.dense1(x)
            x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
 
 
# Building Q-Learning model
class DQN(object):
    def __init__(self):
        # Both nets are the same but one updates every time and one updates every 100 time(if not changed)
        self.eval_net, self.target_net = Net(), Net()
 
        self.learn_step_counter = 0
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # Initializing memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
 
    # 选择动作
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        return action

    # Load model
    def load_model(self):
        self.eval_net.load_state_dict(torch.load("train_model.pth"))
        self.eval_net.eval()
    
 
dqn = DQN()



# Test
dqn.load_model()
game = Engine('examples/complexb.txt', Player, GUI)
s = game.state_tran()
ep_r = 0
score = 0
sum_reward = 0
while True:
    a = dqn.choose_action(s)

    if a == 0:
        action = [True, False, False, False]
    if a == 1:
        action = [True, False, False, True]
    if a == 2:
        action = [True, True, False, False]
    if a == 3:
        action = [True, True, False, True]
    if a == 4:
        action = [True, False, True, False]
    if a == 5:
        action = [True, False, True, True]

    bullet_hitted_asteroids , collided_asteroids , fuel , score , time = game.data_tran()

    done = game.test(action)

    s_ = game.state_tran()

    bullet_hitted_asteroids , collided_asteroids , fuel_ , score_ , time = game.data_tran()

    # modify the reward
    r_now = score_
    r_change = score_ - score
    r = 0.3 * r_now + r_change


    ep_r += r

    if done:
        break

    score = score_
    s = s_
