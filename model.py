import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Linear_QNet(nn.Module):
    def __init__(self, input_size, output_size, batch_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,256)
        self.linear2 = nn.Linear(256,output_size)
        self.batch_size = batch_size
        self.model_folder_path = f'./model/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
        file_name = os.path.join(self.model_folder_path,file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model_target, model_eval, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model_target = model_target
        self.model_eval = model_eval
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model_eval.parameters(), lr=lr)


    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        q_next = self.model_target(next_state)
        q_eval = self.model_eval(next_state) 
        q_pred = self.model_eval(state)

        max_actions = torch.argmax(q_eval, axis=1)

        q_target = q_pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * q_next[idx, max_actions[idx]].item()  # Ensure Q_new is a scalar

            action_idx = action[idx].item()
            q_target[idx][action_idx] = Q_new  # Q_new should be a scalar here


        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        loss = self.criterion(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()