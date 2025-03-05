import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(self, inp, hidden_state):
        inp = F.relu(self.input(inp))
        inp = F.relu(self.hidden(inp)) #

        hidden_state = self.hidden2(hidden_state)
        hidden_state = torch.tanh(inp + hidden_state)

        out = F.relu(self.hidden3(hidden_state))
        out = self.output(out)

        return out, hidden_state
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            self.load_state_dict(torch.load(file_name))
        else:
            print(f"No saved model found at {file_name}")

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, hidden_state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

            if hidden_state.dim() == 1:
                hidden_state = torch.unsqueeze(hidden_state, 0)

        if isinstance(hidden_state, tuple):
            hidden_state = torch.stack(hidden_state)

        hidden_state = hidden_state.detach()

        # 1: predicted Q values with current state
        pred, next_hidden = self.model(state, hidden_state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                next_state_single = next_state[idx].unsqueeze(0)
                next_hidden_single = next_hidden[idx].unsqueeze(0).detach()

                with torch.no_grad():
                    next_pred = self.model(next_state_single, next_hidden_single)[0]
                    Q_new = reward[idx] + self.gamma * torch.max(next_pred)

            if action.dim() > 1:
                target[idx][torch.argmax(action[idx]).item()] = Q_new
            else:
                target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

        return hidden_state.detach()




        