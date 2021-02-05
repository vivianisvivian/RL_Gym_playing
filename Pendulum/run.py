from ddpg import Agent
import gym
import time

env = gym.make('Pendulum-v0')
env.seed(1)
start = time.time()

# 超参数
params = {
    'env': env,
    'gamma': 0.99,  # 增益折损
    'actor_lr': 0.001,  # 学习率
    'critic_lr': 0.001,
    'tau': 0.02,  # 软更新参数
    'capacity': 10000,  # 经验池容量
    'batch_size': 32,  # 随机梯度下降，经验池回放
    'train_with_render': True,  # 是否在训练时开启渲染
    'save_reward': -800,  # 在episode reward达到多少时，停止训练，储存模型
    'actor_model_path': 'model/DDPG_actor.pt',  # 模型储存位置
    'critic_model_path': 'model/DDPG_critic.pt',
    'Reset_parameters': False,   # 是否从0开始训练
}
agent = Agent(**params)
agent.train_model(200)  # 训练num幕
# agent.test_model(3)   # 测试num幕


train_time = time.time() - start
env.close()
print("Time: %.4f" % train_time)


# 查看LOSS变化，终端输入：
# tensorboard --logdir=./log --port=6007
