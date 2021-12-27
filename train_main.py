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


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()


# Building neural network model，returning possible action
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50).to(device)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization

        self.dense1 = nn.Linear(50, 50).to(device)
        self.dense1.weight.data.normal_(0, 0.1)

        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS).to(device)
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
    def __init__(self, use_premodel):
        # Both nets are the same but one updates every time and one updates every 100 time(if not changed)
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        if use_premodel:
            self.eval_net = self.eval_net.load_state_dict(torch.load("train_model.pth")).to(device)
            self.target_net = self.target_net.load_state_dict(torch.load("train_model.pth")).to(device)

        self.learn_step_counter = 0
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # Initializing memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss().to(device)
 
    # Choosing ation
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # input only one sample
        if np.random.uniform() < EPSILON:   # Epsilon Greedy
            actions_value = self.eval_net.forward(x).to(device)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action
 
    # Memory storing
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)) # packing the trasition
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # learn
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)
 
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # The function of detach is to update without backpropagation, because the update of the target is defined earlier
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Save model
    def save_model(self):
        torch.save(self.eval_net.state_dict(), "train_model.pth")
 
dqn = DQN(use_premodel=False)



# Train
print('\nCollecting experience...')
for i_episode in range(2000):
    game = Engine('examples/complexb.txt', Player, GUI)
    s = game.state_tran()
    ep_r = 0
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

        done = game.train(action)

        s_ = game.state_tran()

        bullet_hitted_asteroids , collided_asteroids , fuel_ , score_ , time = game.data_tran()

        # modify the reward
        r_now = score_
        r_change = score_ - score
        r = 0.3 * r_now + r_change

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:  # it will learn when the memory_counter is larger than MEMORY_CAPACITY
            for i in range(100):
                dqn.learn()
            if done:
                print('Ep: ', i_episode,
                    '| Ep_r: ', round(ep_r, 2),
                    '| score: ', score_)

        if r > sum_reward:
            dqn.save_model()
            sum_reward = r


        if done:
            break

        try:
            dqn.store_transition(s, a, r, s_)
        except ValueError:
            break

        score = score_
        s = s_

