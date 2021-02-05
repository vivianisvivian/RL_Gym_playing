import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
writer = SummaryWriter('./log')


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))  # 输出【-1,1】的确定action
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)  # 按维度1拼接起来
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():   # 传入超参数
            setattr(self, key, value)

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        # 定义四个网络
        self.actor = Actor(s_dim, 256, a_dim)
        self.actor_target = Actor(s_dim, 256, a_dim)
        self.critic = Critic(s_dim + a_dim, 256, a_dim)
        self.critic_target = Critic(s_dim + a_dim, 256, a_dim)

        # 初始化参数
        if not self.Reset_parameters:
            print('Loading model')
            self.actor.load_state_dict(torch.load(self.actor_model_path))
            self.critic.load_state_dict(torch.load(self.critic_model_path))

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 定义参数更新方式
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.buffer = []
        self.episode_rewards = []
        self.learn_i = 0

    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()  # 输出action
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:  # 容量满时，第一个出栈
            self.buffer.pop(0)
        self.buffer.append(transition)

    def put_reward(self, *r):
        if len(self.episode_rewards) == 10:  # 容量满时，第一个出栈
            self.episode_rewards.pop(0)
        self.episode_rewards.append(r)

    def learn(self):
        if len(self.buffer) < self.batch_size:  # 还没存够batch_size, 不学习
            return

        samples = random.sample(self.buffer, self.batch_size)

        s0, a0, r1, s1 = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)

        def critic_learn():  # 使用TD_error进行更新，注意区分目标网络和实际网络
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)

            loss1 = loss.detach()
            writer.add_scalar('Critic_LOSS', loss1, self.learn_i)

            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():  # 使用Q值进行梯度上升
            loss = -torch.mean(self.critic(s0, self.actor(s0)))

            loss2 = loss.detach()
            writer.add_scalar('Actor_LOSS', loss2, self.learn_i)

            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):  # 软更新目标网络参数
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        self.learn_i += 1
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

    def train_model(self, num):
        for episode in range(num):
            s0 = self.env.reset()
            episode_reward = 0

            for step in range(500):
                if self.train_with_render:
                    self.env.render()

                a0 = self.act(s0)
                s1, r1, done, _ = self.env.step(a0)
                self.put(s0, a0, r1, s1)

                episode_reward += r1
                s0 = s1

                self.learn()     

            self.put_reward(episode_reward)
            print('Episode:', episode, 'Reward: ', episode_reward)
            if np.mean(self.episode_rewards) >= self.save_reward:
                torch.save(self.actor.state_dict(), self.actor_model_path)
                torch.save(self.critic.state_dict(), self.critic_model_path)
                print("收敛成功，保存模型")
                break
        writer.close()

    def test_model(self, num):
        model = self.actor
        model.load_state_dict(torch.load(self.actor_model_path))
        model.eval()

        for step in range(num):
            s0 = self.env.reset()
            episode_reward = 0

            for step in range(500):
                self.env.render()
                s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
                a0 = model(s0).squeeze(0).detach().numpy()
                s1, r1, done, _ = self.env.step(a0)
                episode_reward += r1
                s0 = s1
            print('Reward: ', episode_reward)

