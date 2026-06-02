import time

# import gym
# import pybullet_envs
import numpy as np
from collections import deque
import torch

import argparse
from buffer import ReplayBuffer
import glob
from utils import collect_random, evaluate
import random

import gym
# from gym import Wrapper
# import collections
import os
import cv2
torch.autograd.set_detect_anomaly = True
from agent_CNN_LSTM import CQLSAC_CNN_LSTM, CQLSAC_CNN
from logger import Logger

# os.environ['WANDB_MODE']='offline'

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL-SAC-active-tracking_agent", help="Run name, default: CQL-SAC")
    parser.add_argument("--buffer_path", type=str,
                        default=None)
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes, default: 200")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=50, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    parser.add_argument("--max_distractor", type=int, default=0, help="")
    parser.add_argument("--lstm_seq_len", type=int, default=20, help="")
    parser.add_argument("--lstm_out", type=int, default=64, help="")
    parser.add_argument("--lstm_layer", type=int, default=1, help="")
    parser.add_argument("--input_type", type=str, default='deva_cnn_lstm', help="")
    
    # 日志配置
    parser.add_argument("--use_wandb", type=int, default=0, help="是否使用 wandb (0/1), default: 0")
    parser.add_argument("--use_tensorboard", type=int, default=1, help="是否使用 tensorboard (0/1), default: 1")
    parser.add_argument("--log_dir", type=str, default="./logs", help="TensorBoard 日志保存目录")
    parser.add_argument("--project_name", type=str, default="CQL", help="项目名称")
    parser.add_argument("--save_dir", type=str, default="./trained_models", help="模型保存目录")


    args = parser.parse_args()

    return args
def load_Buffer(buffer ,path ,config):
    data_list = os.listdir(path)
    data_list.sort()
    print('loading dataset buffer...')
    for d in range(0 ,len(data_list)):
    # for d in range(0,1):
        print('loading :', data_list[d])
        dict_tmp = torch.load(os.path.join(path, data_list[d]))
        if ('deva' in config.input_type.lower() or 'image' in config.input_type.lower()) or 'mask' in config.input_type.lower():
        # states
            state_tmp = np.array([np.array(x[:, :, 0:3]) for x in dict_tmp['image']])[:-1]  # .transpose(0 ,3 ,1 ,2)
            next_state_tmp = np.array([np.array(x[:, :, 0:3]) for x in dict_tmp['image']])[1:]  # .transpose(0 ,3 ,1 ,2)
        if 'devadepth' in config.input_type.lower() or 'rgbd' in config.input_type.lower():
            state_tmp = np.array([np.array(x[:, :, 0:4]) for x in dict_tmp['image']])[:-1]  # .transpose(0 ,3 ,1 ,2)
            next_state_tmp = np.array([np.array(x[:, :, 0:4]) for x in dict_tmp['image']])[1:]  # .transpose(0 ,3 ,1 ,2)
        # actions
        act_tmp = np.array([np.array(x) for x in dict_tmp['action']])[:-1].squeeze(axis=1)
        # rewards
        re_tmp = np.array([np.array(x) for x in dict_tmp['reward']]).squeeze(axis=1)[:-1]  # done
        assert state_tmp.shape[0] == next_state_tmp.shape[0] and re_tmp.shape[0] == next_state_tmp.shape[0] and \
               next_state_tmp.shape[0] == act_tmp.shape[0]
        for i in range(0, state_tmp.shape[0]):
            if i % state_tmp.shape[0] == 0 and i > 0:
                done = True
            else:
                done = False
            buffer.add(
                torch.from_numpy(np.array(cv2.resize(state_tmp[i], (64, 64)).transpose(2, 0, 1))).float().cuda(),
                torch.from_numpy(act_tmp[i]).float().cuda(),
                torch.from_numpy(np.array(re_tmp[i])).float().cuda(),
                torch.from_numpy(
                    np.array(cv2.resize(next_state_tmp[i], (64, 64)).transpose(2, 0, 1))).float().cuda(),
                torch.from_numpy(np.array(done)).float().cuda())

    print('loading dataset buffer finished.')
    return buffer


def train(config):

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device,
                          lstm_seq_len=config.lstm_seq_len,config=config)
    buffer_path = config.buffer_path
    buffer = load_Buffer(buffer, buffer_path, config)

    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    # 初始化日志记录器
    logger = Logger(
        project_name=config.project_name,
        run_name=config.run_name,
        config=config,
        use_wandb=bool(config.use_wandb),
        use_tensorboard=bool(config.use_tensorboard),
        log_dir=config.log_dir
    )

    with logger:
        if 'deva' in config.input_type.lower() or 'image' in config.input_type.lower() or 'Mask' in config.input_type.lower():
            if 'cnn' in config.input_type.lower() and 'lstm' in config.input_type.lower():
                agent = CQLSAC_CNN_LSTM(state_size=(3, 64, 64),
                                        action_size=2,
                                        tau=config.tau,
                                        hidden_size=config.hidden_size,
                                        learning_rate=config.learning_rate,
                                        temp=config.temperature,
                                        with_lagrange=config.with_lagrange,
                                        cql_weight=config.cql_weight,
                                        target_action_gap=config.target_action_gap,
                                        device=device,
                                        stack_frames=1,
                                        lstm_seq_len=config.lstm_seq_len,
                                        lstm_layer=config.lstm_layer,
                                        lstm_out=config.lstm_out)
            elif 'cnn' in config.input_type.lower():
                agent = CQLSAC_CNN(state_size=(3, 64, 64),
                                        action_size=2,
                                        tau=config.tau,
                                        hidden_size=config.hidden_size,
                                        learning_rate=config.learning_rate,
                                        temp=config.temperature,
                                        with_lagrange=config.with_lagrange,
                                        cql_weight=config.cql_weight,
                                        target_action_gap=config.target_action_gap,
                                        device=device,
                                       )
            elif 'mlp' in config.input_type.lower():
                agent = CQLSAC_MLP_LSTM(state_size=(5, 4),
                                        action_size=2,
                                        tau=config.tau,
                                        hidden_size=config.hidden_size,
                                        learning_rate=config.learning_rate,
                                        temp=config.temperature,
                                        with_lagrange=config.with_lagrange,
                                        cql_weight=config.cql_weight,
                                        target_action_gap=config.target_action_gap,
                                        device=device,
                                        stack_frames=1,
                                        lstm_seq_len=config.lstm_seq_len,
                                        lstm_layer=config.lstm_layer,
                                        lstm_out=config.lstm_out)


        if 'devadepth' in config.input_type.lower() or 'rgbd' in config.input_type.lower():
            if 'cnn' in config.input_type.lower() and 'lstm' in config.input_type.lower():
                agent = CQLSAC_CNN_LSTM(state_size=(4, 64, 64),
                                        action_size=2,
                                        tau=config.tau,
                                        hidden_size=config.hidden_size,
                                        learning_rate=config.learning_rate,
                                        temp=config.temperature,
                                        with_lagrange=config.with_lagrange,
                                        cql_weight=config.cql_weight,
                                        target_action_gap=config.target_action_gap,
                                        device=device,
                                        stack_frames=1,
                                        lstm_seq_len=config.lstm_seq_len,
                                        lstm_layer=config.lstm_layer,
                                        lstm_out=config.lstm_out)
            elif 'cnn' in config.input_type.lower():
                agent = CQLSAC_CNN(state_size=(4, 64, 64),
                                   action_size=2,
                                   tau=config.tau,
                                   hidden_size=config.hidden_size,
                                   learning_rate=config.learning_rate,
                                   temp=config.temperature,
                                   with_lagrange=config.with_lagrange,
                                   cql_weight=config.cql_weight,
                                   target_action_gap=config.target_action_gap,
                                   device=device,
                                   )



        # 监控模型梯度
        logger.watch(agent, log="gradients", log_freq=10)

        for i in range(1, config.episodes + 1):
            episode_steps = 0
            rewards = 0
            while True:
                train_q1, train_q2, policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn(
                    buffer.sample())
                steps += 1
                if steps >= 200:
                    episode_steps += 1
                    steps = 0
                    break


            average10.append(rewards)
            total_steps += episode_steps
            print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps, ))

            logger.log({
                # "Train/Reward": rewards,
                # "Train/Average10": np.mean(average10),
                "Train/Steps": total_steps,
                # "Train/total Loss": total_loss,
                "Train/train_q1": train_q1,
                "Train/train_q2": train_q2,
                "Train/Policy Loss": policy_loss,
                "Train/Alpha Loss": alpha_loss,
                "Train/Lagrange Alpha Loss": lagrange_alpha_loss,
                "Train/CQL1 Loss": cql1_loss,
                "Train/CQL2 Loss": cql2_loss,
                "Train/Bellman error 1": bellmann_error1,
                "Train/Bellman error 2": bellmann_error2,
                "Train/Alpha": current_alpha,
                "Train/Lagrange Alpha": lagrange_alpha,
                "Train/Steps": steps,
                "Train/Episode": i,
                "Train/Buffer size": buffer.__len__(),

            }, step=i)

            if i % config.save_every == 0:
                logger.save_model(agent, config.save_dir, "CQL-SAC", ep=i)


if __name__ == "__main__":
    config = get_config()
    train(config)

