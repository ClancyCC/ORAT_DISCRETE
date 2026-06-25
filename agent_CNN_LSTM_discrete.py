"""
Discrete action version of CQL-SAC agents.
Uses DiscreteActor (Categorical policy with Gumbel-Softmax) instead of continuous Gaussian Actor.
Actions are scalar 0~(action_size-1), one-hot encoded before being passed to Critic.
CQL enumerates all discrete actions instead of sampling random continuous actions.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, DiscreteActor, CNN_LSTM, CNN_simple
import numpy as np
import math
import copy
from torch.autograd import Variable
from utils_basic import weights_init


class CQLSAC_CNN_LSTM_Discrete(nn.Module):
    """Discrete-action version of CQLSAC_CNN_LSTM."""

    def __init__(self,
                 state_size,
                 action_size,
                 tau,
                 hidden_size,
                 learning_rate,
                 temp,
                 with_lagrange,
                 cql_weight,
                 target_action_gap,
                 device,
                 stack_frames,
                 lstm_seq_len,
                 lstm_layer,
                 lstm_out
                 ):
        super(CQLSAC_CNN_LSTM_Discrete, self).__init__()
        self.state_size = state_size
        self.action_size = action_size  # number of discrete actions (e.g., 9)
        self.stack_frames = stack_frames
        self.device = device
        self.lstm_seq_len = lstm_seq_len
        self.gamma = torch.FloatTensor([0.99]).to(device)

        self.tau = tau
        hidden_size = hidden_size
        learning_rate = learning_rate
        self.clip_grad_param = 1

        # Discrete SAC: target entropy = -ln(|A|)
        self.target_entropy = -np.log(action_size)

        self.log_alpha = torch.tensor([0.0], requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)

        # CQL params
        self.with_lagrange = with_lagrange
        self.temp = temp
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate)

        # image processing network
        self.lstm_layer = lstm_layer
        self.CNN_LSTM = CNN_LSTM(state_size=self.state_size,
                                 action_size=self.action_size,
                                 hidden_size=hidden_size,
                                 stack_frames=self.stack_frames,
                                 lstm_out=lstm_out,
                                 lstm_layer=self.lstm_layer
                                 ).to(self.device)
        self.CNN_LSTM_optimizer = optim.Adam(self.CNN_LSTM.parameters(), lr=learning_rate)

        # --- Discrete Actor ---
        self.actor_local = DiscreteActor(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # --- Critic Network (w/ Target Network) ---
        # Critic receives state + one-hot action (action_size-dim)
        self.critic1 = Critic(self.CNN_LSTM.outdim, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(self.CNN_LSTM.outdim, action_size, hidden_size, 1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()
        self.critic1_target = Critic(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(self.CNN_LSTM.outdim, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

    def get_action(self, state, eval=False):
        """Returns action index (0~action_size-1) for given state."""
        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
                self.actor_local.train()
            else:
                action = self.actor_local.get_action(state)
        return action.cpu().numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)
        q1 = self.critic1(states, actions_pred)
        q2 = self.critic2(states, actions_pred)
        min_Q = torch.min(q1, q2)
        actor_loss = ((alpha * log_pis - min_Q)).mean()
        return actor_loss, log_pis

    def learn(self, experiences):
        """Updates actor, critics and entropy_alpha using batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences

        batch_size = states.shape[0]

        # ---- one-hot encode scalar actions (0~action_size-1) ----
        # actions from buffer: [batch, lstm_seq_len] (float scalars)
        actions = actions.long()  # convert to long for one_hot
        actions_onehot = F.one_hot(actions, num_classes=self.action_size).float()
        # actions_onehot: [batch, lstm_seq_len, action_size]

        # ---- Image processing ----
        states = self.CNN_LSTM(states)
        states = states.reshape(batch_size * self.lstm_seq_len, -1)

        actions_onehot = actions_onehot.reshape(batch_size * self.lstm_seq_len, -1)
        # actions_onehot: [batch*seq_len, action_size]

        with torch.no_grad():
            next_states = self.CNN_LSTM(next_states)
            next_states = next_states.reshape(batch_size * self.lstm_seq_len, -1)

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)

        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            next_action, new_log_pi = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states, next_action)
            Q_target2_next = self.critic2_target(next_states, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * new_log_pi
            Q_targets = rewards.reshape(batch_size * self.lstm_seq_len, 1) + \
                        (self.gamma * (1 - dones.reshape(batch_size * self.lstm_seq_len, 1)) * Q_target_next)

        q1 = self.critic1(states, actions_onehot)
        q2 = self.critic2(states, actions_onehot)
        q1_train = q1.mean()
        q2_train = q2.mean()

        critic1_loss = F.mse_loss(q1, Q_targets)
        critic2_loss = F.mse_loss(q2, Q_targets)

        # ---------------------------- CQL for discrete actions ---------------------------- #
        # Enumerate ALL discrete actions for CQL (instead of sampling random continuous actions)
        num_actions = self.action_size
        all_actions = torch.eye(num_actions).to(self.device)  # [num_actions, num_actions] one-hot

        # Repeat states for each action: [batch*seq_len * num_actions, lstm_out]
        all_states = states.unsqueeze(1).repeat(1, num_actions, 1).view(-1, states.shape[-1])
        all_actions_expanded = all_actions.unsqueeze(0).repeat(states.shape[0], 1, 1).view(-1, num_actions)

        # Compute Q(s, a) for all actions
        q1_all = self.critic1(all_states, all_actions_expanded).view(-1, num_actions)  # [batch*seq_len, 9]
        q2_all = self.critic2(all_states, all_actions_expanded).view(-1, num_actions)

        # CQL: logsumexp(Q) - Q(s, a_data)
        cql1_scaled_loss = ((torch.logsumexp(q1_all / self.temp, dim=1).mean() * self.temp) - q1.mean()) * self.cql_weight
        cql2_scaled_loss = ((torch.logsumexp(q2_all / self.temp, dim=1).mean() * self.temp) - q2.mean()) * self.cql_weight

        cql_alpha_loss = torch.FloatTensor([0.0]).to(self.device)
        cql_alpha = torch.FloatTensor([0.0]).to(self.device)
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss

        # Update critics
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        self.CNN_LSTM_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
        self.CNN_LSTM_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return (q1_train.item(), q2_train.item(), actor_loss.item(), alpha_loss.item(),
                critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(),
                cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item())

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class CQLSAC_CNN_Discrete(nn.Module):
    """Discrete-action version of CQLSAC_CNN (no LSTM)."""

    def __init__(self,
                 state_size,
                 action_size,
                 tau,
                 hidden_size,
                 learning_rate,
                 temp,
                 with_lagrange,
                 cql_weight,
                 target_action_gap,
                 device
                 ):
        super(CQLSAC_CNN_Discrete, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.device = device

        self.gamma = torch.FloatTensor([0.99]).to(device)
        self.tau = tau
        hidden_size = hidden_size
        learning_rate = learning_rate
        self.clip_grad_param = 1

        # Discrete SAC: target entropy = -ln(|A|)
        self.target_entropy = -np.log(action_size)

        self.log_alpha = torch.tensor([0.1], requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach()
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=learning_rate)

        # CQL params
        self.with_lagrange = with_lagrange
        self.temp = temp
        self.cql_weight = cql_weight
        self.target_action_gap = target_action_gap
        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.cql_alpha_optimizer = optim.Adam(params=[self.cql_log_alpha], lr=learning_rate)

        # image processing network
        self.CNN_Simple = CNN_simple(self.state_size, 1).to(device)
        self.CNN_optimizer = optim.Adam(self.CNN_Simple.parameters(), lr=learning_rate)

        # --- Discrete Actor ---
        self.actor_local = DiscreteActor(self.CNN_Simple.outdim, action_size, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate)

        # --- Critic Network (w/ Target Network) ---
        self.critic1 = Critic(self.CNN_Simple.outdim, action_size, hidden_size, 2).to(device)
        self.critic2 = Critic(self.CNN_Simple.outdim, action_size, hidden_size, 1).to(device)

        assert self.critic1.parameters() != self.critic2.parameters()

        self.critic1_target = Critic(self.CNN_Simple.outdim, action_size, hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(self.CNN_Simple.outdim, action_size, hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)

    def get_action(self, state, eval=False):
        """Returns action index (0~action_size-1) for given state."""
        with torch.no_grad():
            if eval:
                action = self.actor_local.get_det_action(state)
                self.actor_local.train()
            else:
                action = self.actor_local.get_action(state)
        return action.cpu().numpy()

    def calc_policy_loss(self, states, alpha):
        actions_pred, log_pis = self.actor_local.evaluate(states)
        q1 = self.critic1(states, actions_pred)
        q2 = self.critic2(states, actions_pred)
        min_Q = torch.min(q1, q2)
        actor_loss = ((alpha * log_pis - min_Q)).mean()
        return actor_loss, log_pis

    def learn(self, experiences):
        """Updates actor, critics and entropy_alpha using batch of experience tuples."""
        states, actions, rewards, next_states, dones = experiences
        batch_size = states.shape[0]

        # ---- one-hot encode scalar actions (0~action_size-1) ----
        # actions from buffer: [batch] (float scalars), or [batch, 1]
        if actions.dim() > 1 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        actions = actions.long()
        actions_onehot = F.one_hot(actions, num_classes=self.action_size).float()
        # actions_onehot: [batch, action_size]

        # ---- Image processing ----
        states = self.CNN_Simple(states)
        states = states.reshape(batch_size, -1)
        with torch.no_grad():
            next_states = self.CNN_Simple(next_states)
            next_states = next_states.reshape(batch_size, -1)

        # ---------------------------- update actor ---------------------------- #
        current_alpha = copy.deepcopy(self.alpha)

        actor_loss, log_pis = self.calc_policy_loss(states, current_alpha)
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # Compute alpha loss
        alpha_loss = - (self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward(retain_graph=True)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        # ---------------------------- update critic ---------------------------- #
        with torch.no_grad():
            next_action, new_log_pi = self.actor_local.evaluate(next_states)
            Q_target1_next = self.critic1_target(next_states, next_action)
            Q_target2_next = self.critic2_target(next_states, next_action)
            Q_target_next = torch.min(Q_target1_next, Q_target2_next) - self.alpha.to(self.device) * new_log_pi
            Q_targets = rewards + (self.gamma * (1 - dones) * Q_target_next.squeeze())

        q1 = self.critic1(states, actions_onehot)
        q2 = self.critic2(states, actions_onehot)
        q1_train = q1.mean()
        q2_train = q2.mean()

        critic1_loss = F.mse_loss(q1, Q_targets)
        critic2_loss = F.mse_loss(q2, Q_targets)

        # ---------------------------- CQL for discrete actions ---------------------------- #
        num_actions = self.action_size
        all_actions = torch.eye(num_actions).to(self.device)  # [num_actions, num_actions]

        all_states = states.unsqueeze(1).repeat(1, num_actions, 1).view(-1, states.shape[-1])
        all_actions_expanded = all_actions.unsqueeze(0).repeat(states.shape[0], 1, 1).view(-1, num_actions)

        q1_all = self.critic1(all_states, all_actions_expanded).view(-1, num_actions)
        q2_all = self.critic2(all_states, all_actions_expanded).view(-1, num_actions)

        cql1_scaled_loss = ((torch.logsumexp(q1_all / self.temp, dim=1).mean() * self.temp) - q1.mean()) * self.cql_weight
        cql2_scaled_loss = ((torch.logsumexp(q2_all / self.temp, dim=1).mean() * self.temp) - q2.mean()) * self.cql_weight

        cql_alpha_loss = torch.FloatTensor([0.0]).to(self.device)
        cql_alpha = torch.FloatTensor([0.0]).to(self.device)
        if self.with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), min=0.0, max=1000000.0).to(self.device)
            cql1_scaled_loss = cql_alpha * (cql1_scaled_loss - self.target_action_gap)
            cql2_scaled_loss = cql_alpha * (cql2_scaled_loss - self.target_action_gap)

            self.cql_alpha_optimizer.zero_grad()
            cql_alpha_loss = (- cql1_scaled_loss - cql2_scaled_loss) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimizer.step()

        total_c1_loss = critic1_loss + cql1_scaled_loss
        total_c2_loss = critic2_loss + cql2_scaled_loss

        # Update critics
        self.critic1_optimizer.zero_grad()
        total_c1_loss.backward(retain_graph=True)
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_optimizer.step()

        self.CNN_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        total_c2_loss.backward()
        clip_grad_norm_(self.critic2.parameters(), self.clip_grad_param)
        self.critic2_optimizer.step()
        self.CNN_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target)
        self.soft_update(self.critic2, self.critic2_target)

        return (q1_train.item(), q2_train.item(), actor_loss.item(), alpha_loss.item(),
                critic1_loss.item(), critic2_loss.item(), cql1_scaled_loss.item(),
                cql2_scaled_loss.item(), current_alpha, cql_alpha_loss.item(), cql_alpha.item())

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)