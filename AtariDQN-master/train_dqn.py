#!/usr/bin/env python3

import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

COLAB = False
CUDA = True

if not COLAB:
    from lib import wrappers
    from lib import dqn_model

    import argparse
    from tensorboardX import SummaryWriter

ENV_NAME = "PongNoFrameskip-v4"
ENV_NAME = "BreakoutNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10 ** 4 * 2
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 1000
LEARNING_STARTS = 10000

EPSILON_DECAY = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

MODEL = "PretrainedModels/PongNoFrameskip-v4-407.dat"
LOAD_MODEL = True

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.bool), np.array(next_states)


class Agent:
    def __init__(self, env, replay_memory):
        self.env = env
        self.replay_memory = replay_memory
        self._reset()
        self.last_action = 0

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        """
        Select action
        Execute action and step environment
        Add state/action/reward to experience replay
        """
        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.replay_memory.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calculate_loss(batch, net, target_net, device="cpu"):
    """
    Calculate MSE between actual state action values,
    and expected state action values from DQN
    """
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done = torch.BoolTensor(dones).to(device)

    # print('next_states_v', actions_v )
    state_action_values = net(states_v).gather(1, actions_v.long().unsqueeze(-1)).squeeze(-1)
    # print('state_action_values:', state_action_values)
    next_state_values = target_net(next_states_v).max(1)[0]
    # print('next_state_values: ', next_state_values)
    next_state_values[done] = 0.0
    # print('done:', done)
    next_state_values = next_state_values.detach()

    # print(type(next_state_values))
    # print(type(GAMMA))
    # print(type(rewards_v))

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

print("ReplayMemory will require {}gb of GPU RAM".format(round(REPLAY_SIZE * 32 * 84 * 84 / 1e+9, 2)))

if __name__ == "__main__":
    if COLAB:
        """Default argparse does not work on colab"""
        class ColabArgParse():
            def __init__(self, cuda, env, reward, model):
                self.cuda = cuda
                self.env = env
                self.reward = reward
                self.model = model

        args = ColabArgParse(CUDA, ENV_NAME, MEAN_REWARD_BOUND, MODEL)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
        parser.add_argument("--env", default=ENV_NAME,
                            help="Name of the environment, default=" + ENV_NAME)
        parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
                            help="Mean reward to stop training, default={}".format(round(MEAN_REWARD_BOUND, 2)))
        parser.add_argument("-m", "--model", help="Model file to load")
        args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    # Make Gym environement and DQNs
    if COLAB:
        env = make_env(args.env)
        net = DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    else:
        env = wrappers.make_env(args.env)
        net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
        writer = SummaryWriter(comment="-" + args.env)

    print(net)

    replay_memory = ExperienceReplay(REPLAY_SIZE)
    agent = Agent(env, replay_memory)
    epsilon = EPSILON_START

    if LOAD_MODEL:
        net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
        target_net.load_state_dict(net.state_dict())
        print("Models loaded from disk!")
        # Lower exploration rate
        EPSILON_START = EPSILON_FINAL

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    best_mean_reward = None
    frame_idx = 0
    timestep_frame = 0
    timestep = time.time()
    iepisode = 0

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - timestep_frame) / (time.time() - timestep)
            timestep_frame = frame_idx
            timestep = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            # print("{} frames: done {} games, mean reward {}, eps {}, speed {} f/s".format(
                # frame_idx, len(total_rewards), round(mean_reward, 3), round(epsilon,2), round(speed, 2)))
            if not COLAB:
                writer.add_scalar("epsilon", epsilon, frame_idx)
                writer.add_scalar("speed", speed, frame_idx)
                writer.add_scalar("reward_100", mean_reward, frame_idx)
                writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-" + str(len(total_rewards)) + ".dat")
                if COLAB:
                    gsync.update_file_to_folder(args.env + "-" + str(len(total_rewards)) + ".dat")
                if best_mean_reward is not None:
                    print("New best mean reward {} -> {}, model saved".format(round(best_mean_reward, 3), round(mean_reward, 3)))
                best_mean_reward = mean_reward
            if mean_reward > args.reward and len(total_rewards) > 10:
                print("Game solved in {} frames! Average score of {}".format(frame_idx, mean_reward))
                break
            print(reward)
            print('Episode:', iepisode, ' score:', reward, ' Avg Score:', mean_reward, ' epsilon:', epsilon)

        if len(replay_memory) < LEARNING_STARTS:
            continue

        if frame_idx % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = replay_memory.sample(BATCH_SIZE)
        loss_t = calculate_loss(batch, net, target_net, device=device)
        loss_t.backward()
        optimizer.step()
        iepisode += 1
    env.close()
    if not COLAB:
        writer.close()
