import random
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from collections import deque
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Player:
    def __init__(self, name: str, turn : int) -> None:
        self.name = name
        self.turn = turn

    def move(self, board_arr) -> int :
        raise NotImplementedError("Implemented by child class")
    
class Bot(Player):
    def __init__(self, name: str, turn : int) -> None:
        super().__init__(name,turn)
        self.name = 'bot'

    def move(self, board_arr) -> int :
        available_cols = board_arr.sum(axis = 0) < board_arr.shape[0]
        available_cols = [c for c, i in zip(list(range(board_arr.shape[1])), available_cols) if i]
        col = random.choice(available_cols)
        return col
    

class RLBot(Player):
    def __init__(self, name: str, turn:int) -> None:
        super().__init__(name,turn)
        self.loss_fn = nn.MSELoss()
        self.activation = nn.ReLU  # define as class
        self.lr = 1e-3
        self.model = self.initialize_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        self.gamma = 0.9
        self.epsilon = 0.2 if self.turn!= -1 else 0.0

        self.stop_training = False
        self.min_loss_dict = {'current_min': np.inf, 'num_steps': 0}
        self.patience = 200


        self.losses = []
        self.rewards = []
        self.n_moves = 0
        #sqars : state_t, q_vals_t , action_t, reward_t, state_t+1
        self.current_sqars = [None, None, None, None, None]
        self.reward_vals = {
            'draw': 5,
            'win' : 10,
            'loss' : -10,
            'move' : 0,
        }

    def initialize_model(self):
        input_n = 84
        hidden_n = 150
        hidden_n_2 = 100
        output_n = 7
        model = nn.Sequential(
            nn.Linear(input_n, hidden_n),
            self.activation(),
            nn.Linear(hidden_n, hidden_n_2),
            self.activation(),
            nn.Linear(hidden_n_2, output_n),
        )
        model.to(device)
        return model

    def get_state_array(self, piece_arrays : Dict):
        state_array = np.stack([piece_arrays[self.turn], piece_arrays[-self.turn]])  #6*7*2
        return state_array
    
    def process_state(self, curr_state: np.array):
        num_cells = curr_state[0].shape[0]*curr_state[0].shape[1]
        curr_state = curr_state.reshape(1, num_cells*2)

        if isinstance(self.activation, nn.ReLU):
            curr_state += np.random.rand(1, num_cells*2)/10.0
        state = torch.from_numpy(curr_state).float()
        return state

    def move(self, piece_arrays: Dict) -> int:
        curr_state = self.get_state_array(piece_arrays)
        state = self.process_state(curr_state)
        q_vals = self.model(state)
        q_vals_ = q_vals.data.numpy().squeeze()

        board_arr = piece_arrays[self.turn] + piece_arrays[-self.turn]
        available_cols = board_arr.sum(axis = 0) < board_arr.shape[0]
        available_indices = [i for i, valid in enumerate(available_cols) if valid]
        if len(available_indices) == 0:
            raise ValueError("No legal moves available. Board is full or in an invalid state.")

        if random.random() < self.epsilon:
            action_ = random.choice(available_indices)
        else:
            masked_q_vals = np.where(available_cols, q_vals_, -1e9)
            action_ = int(np.argmax(masked_q_vals))
        self.current_sqars[0] = state
        self.current_sqars[1] = q_vals
        self.current_sqars[2] = action_
        self.n_moves +=1
        return action_
    
    def get_reward(self, move_result):
        if move_result is None:
            reward = self.reward_vals['move']
        elif 'draw' in move_result:
            reward = self.reward_vals['draw']
        else:
            win_side = int(move_result.split('_')[1])
            reward = self.reward_vals['win'] if win_side == self.turn else self.reward_vals['loss']
        
        self.current_sqars[3] = reward
        self.rewards.append(reward)
        return reward
    
    
    def reset_vars(self):
        self.current_sqars = [None, None, None, None, None]

    def update_early_stopping(self):
        check_window_steps = 50
        if len(self.losses) >= check_window_steps:
            current_avg_loss = np.mean(self.losses[-check_window_steps:])
            if current_avg_loss < self.min_loss_dict['current_min']:
                self.min_loss_dict['current_min'] = current_avg_loss
                self.min_loss_dict['num_steps'] = 0
            else:
                self.min_loss_dict['num_steps']+=1
            if self.min_loss_dict['num_steps'] >= self.patience:
                self.stop_training = True


    def train(self, new_piece_arrays, result):
        new_state = self.get_state_array(new_piece_arrays)     #new state s' after we make an action
        new_state = self.process_state(new_state)

        self.current_sqars[4] = new_state

        reward = self.get_reward(result)

        with torch.no_grad():
            new_q = self.model(new_state)
        max_q = torch.max(new_q)     #get the value of next best action

        #Q_learning target value
        Y = reward if result is not None else reward + (self.gamma * max_q)
        # Y = torch.tensor([Y]).detach().squeeze().to(device)
        Y = torch.tensor([Y], dtype=torch.float32).detach().squeeze().to(device)

        # X = self.current_sqars[1].squeeze()[self.current_sqars[2]]
        X = self.current_sqars[1].squeeze()[self.current_sqars[2]]


        loss = self.loss_fn(X,Y) 
        self.optimizer.zero_grad()   #reset gradients
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()     #update network paramters

        self.update_early_stopping()
        if result is not None:
            self.reset_vars()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=device))

    def plot_results(self, show = False, save_path = None):
        fig,ax1 = plt.subplots()
        ax1.set_xlabel("Moves")
        ax1.set_ylabel("Rewards")
        ax1.plot(np.arange(len(self.rewards)), self.rewards, color = 'r')

        ax2 = ax1.twinx()
        ax2.set_ylabel("loss")
        ax2.plot(np.arange(len(self.losses)), self.losses, color='b')

        fig.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()
    
    def save_model_and_results(self, model_path: str):
        torch.save(self.model.state_dict(), model_path + 'model.pth')
        #results : losses, win, avg_win_rate
        np.savetxt(model_path+'losses.csv', np.array(self.losses), delimiter=',')
        self.plot_results(save_path = model_path + 'results.png')




class Human(Player):
    def __init__(self, name: str, turn : int) -> None:
        super().__init__(name, turn)

class RLBotDDQN(RLBot):
    def __init__(self, name: str, turn: int):
        super().__init__(name, turn)
        self.memory = deque(maxlen = 1000)
        self.batch_size = 200
        self.target_sync_freq = 350
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())
    
    def train(self, new_piece_arrays, result):
        new_state = self.get_state_array(new_piece_arrays)     #new state s' after we make an action
        new_state = self.process_state(new_state)
        self.current_sqars[4] = new_state
        reward = self.get_reward(result)
        curr_experience = (
            self.current_sqars[0],  # state_t
            self.current_sqars[2],  # action_t
            reward,     # reward_t
            self.current_sqars[4],  # next_state
            int(result is not None) # done
        )


        self.memory.append(curr_experience)

        if len(self.memory) <= self.batch_size:
            return None

        minibatch = random.sample(self.memory, self.batch_size)
        s_batch = torch.cat([s for (s,a,r,s2,d) in minibatch]).to(device)
        a_batch = torch.Tensor([a for (s,a,r,s2,d) in minibatch]).to(device)
        r_batch = torch.Tensor([r for (s,a,r,s2,d) in minibatch]).to(device)
        s2_batch = torch.cat([s2 for (s,a,r,s2,d) in minibatch]).to(device)
        d_batch = torch.Tensor([d for (s,a,r,s2,d) in minibatch]).to(device)

        q1 = self.model(s_batch)

        #get Q values of new state to update last state's Q value
        with torch.no_grad():
            new_q = self.target_model(s2_batch)
        max_q = torch.max(new_q, dim=1)     #get the value of next best action

        #Q_learning target value
        Y = r_batch + self.gamma*((1-d_batch)*max_q[0]) 
        # Y = torch.tensor([Y]).detach().squeeze().to(device)
        X = q1.gather(dim=1 , index = a_batch.long().unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(X,Y.detach()) 
        self.optimizer.zero_grad()   #reset gradients
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()     #update network paramters

        if self.n_moves % self.target_sync_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        if result is not None:
            self.reset_vars()






