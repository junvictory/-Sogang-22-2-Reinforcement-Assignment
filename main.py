import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import collections

# pytorch library is used for deep learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# HYPER PARAM
LEARNING_RATE =0.0005
GAMMA = 0.98
BUFFER_LIMIT = 50000 #Replay Buffer SIZE
BATCH_SIZE = 32
EPISODE = 10000
TRAIN_START = 2000
scoreSum = []
scoreTimeInterval=[]

epsilonList = []
rewardAveList = []


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_LIMIT)    # double-ended queue
    
    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self,env):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 128) #State 2
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)#State 3
        self.env = env

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return self.env.action_space.sample()
        else : 
            return out.argmax().item()   

def train(q, q_target, memory, optimizer):
    for i in range(10):
        state,action,reward,new_state,done_mask = memory.sample(BATCH_SIZE)
        q_out = q(state)
        q_action = q_out.gather(1,action)
        # DQN
        max_q_new_state = q_target(new_state).max(1)[0].unsqueeze(1)
        target = reward + GAMMA * max_q_new_state * done_mask 
        # MSE Loss
        loss = F.mse_loss(q_action, target)

        #Pytorch Code
        optimizer.zero_grad()#Gradient
        loss.backward()#Backpropagation
        optimizer.step()#Update Parameter


def main():
    env = gym.make("MountainCar-v0")
    q_network = QNetwork(env) 
    q_network_target = QNetwork(env) 
    q_network_target.load_state_dict(q_network.state_dict()) 
    rp_buffer = ReplayBuffer() 

    success = 0.0
    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

    for c_episode in range(EPISODE):        
        state = env.reset()
        done = False
        epsilon = max(1-c_episode/(EPISODE*0.8),0.01) #Linear annealing from 8% to 1%

        while not done:
            action = q_network.sample_action(torch.from_numpy(state).float(),epsilon)
            new_state, reward, done, _ = env.step(action) 
            done_mask = 0.0 if done else 1.0

            if action == 2 and new_state[0]-state[0]>0:
              reward = 1
            if action == 0 and new_state[0]-state[0]<0:
              reward = 1

            rp_buffer.put((state,action,reward,new_state,done_mask))
            
            state = new_state.copy()
            score += reward

            if done:
                epsilonList.append(epsilon)
                scoreSum.append(score)  
                break

        if rp_buffer.size()>TRAIN_START:#Training Start
            train(q_network, q_network_target, rp_buffer, optimizer)
        
        

        if c_episode%print_interval==0 and c_episode!=0:
            q_network_target.load_state_dict(q_network.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(c_episode, scoreSum[c_episode]/print_interval, rp_buffer.size(), epsilon*100))
            scoreTimeInterval.append(c_episode)
            rewardAveList.append(scoreSum[c_episode]/print_interval)  
            score = 0.0
            
    env.close()
    print(success)

    plt.plot(scoreTimeInterval, rewardAveList)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()

if __name__ == '__main__':
  print("start")
  main()