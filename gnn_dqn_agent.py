import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


import collections
import math
import os
import os.path as osp
from tqdm import tqdm
from typing import List
import random
import time
import zipfile

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.display.max_rows = 10
from sklearn import metrics
from tensorly import decomposition

import torch
from torch.functional import tensordot
from torch import nn, optim, Tensor
import torch_geometric
from torch_geometric.data import Dataset, Data, download_url, extract_zip
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj

import random

def getEmbedding(model, users, pos, data):
    """
    INPUT:
        model: the LightGCN model you are training on
        users: this is the user index (note: use 0-indexed and not user number,
            which is 1-indexed)
        pos: positive index corresponding to an item that the user like
        neg: negative index corresponding to an item that the user doesn't like
        data: the entire data, used to fetch all users and all items
        mask: Masking matrix indicating edges present in the current
            train / validation / test set.
    """
    # assuming we always search for users and items by their indices (instead of
    # user/item number)
    all_users_items = model(model.embedding_user_item.weight.clone(),
                            data["edge_index"])
    all_users = all_users_items[:len(data["users"])]
    all_items = all_users_items[len(data["users"]):]
    users_emb = all_users[users]
    pos_emb = all_items[pos]
    # neg_emb = all_items[neg]
    n_user = len(data["users"])
    users_emb_ego = model.embedding_user_item(users)
    # offset the index to fetch embedding from user_item
    pos_emb_ego = model.embedding_user_item(pos + n_user)
    # neg_emb_ego = model.embedding_user_item(neg + n_user)
    return users_emb, pos_emb, users_emb_ego, pos_emb_ego


def bpr_loss(model, users, pos, data):
    """ 
    INPUT:
        model: the LightGCN model you are training on
        users: this is the user index (note: use 0-indexed and not user number,
            which is 1-indexed)
        pos: positive index corresponding to an item that the user like
            (0-indexed, note to index items starting from 0)
        neg: negative index corresponding to an item that the user doesn't like
        data: the entire data, used to fetch all users and all items
        mask: Masking matrix indicating edges present in the current
            train / validation / test set.
    OUTPUT:
        loss, reg_loss
    """
    # assuming we always sample the same number of positive and negative sample
    # per user
    # assert len(users) == len(pos) and len(users) == len(neg)
    (users_emb, pos_emb, userEmb0,  posEmb0) = getEmbedding(model, users, pos, data)
    reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                        posEmb0.norm(2).pow(2))/float(len(users))
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    
    loss = torch.mean(torch.nn.functional.softplus(-pos_scores))
    
    return loss, reg_loss


def _sample_pos_neg(data, mask):
    """Samples (user, positive item, negative item) tuples per user.

    If a user does not have a postive (negative) item, we choose an item
    with unknown liking (an item without raw rating data).

    Args:
        data: Dataset object containing edge_index and raw ratings matrix.
        mask: Masking matrix indicating edges present in the current
            train / validation / test set.
        num_samples_per_user: Number of samples to generate for each user.

    Returns:
        torch.Tensor object of (user, positive item, negative item) samples.
    """
    start = time.time()
    samples = []
    all_items = set(range(len(data["items"])))
    for user_index, user in enumerate(data["users"]):
        pos_items = set(
            torch.nonzero(data["edge_index"][user_index])[:, 0].tolist())
        unknown_items = all_items.difference(
               set(
                   torch.nonzero(
                       data["raw_edge_index"][user_index])[:, 0].tolist()))
        neg_items = all_items.difference(
           set(pos_items)).difference(set(unknown_items))
        unmasked_items = set(torch.nonzero(mask[user_index])[:, 0].tolist())
        if len(unknown_items.union(pos_items)) == 0 or \
               len(unknown_items.union(neg_items)) == 0:
           continue
        for _ in range(1):
           if len(pos_items.intersection(unmasked_items)) == 0:
               pos_item_index = random.choice(
                   list(unknown_items.intersection(unmasked_items)))
           else:
               pos_item_index = random.choice(
                   list(pos_items.intersection(unmasked_items)))
        samples.append((user_index, pos_item_index))
    end = time.time()
    return torch.tensor(samples, dtype=torch.int32)

def sample_pos_neg(data, train_mask):
    train_samples = _sample_pos_neg(data, train_mask)
    return train_samples


class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims, fc3_dims,
                 n_actions, device, name='dqn_eval_', chkpt_dir='.'):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        # self.fc4_dims = fc4_dims
        # self.fc5_dims = fc5_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        # self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        # self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device=device
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        #observation = observation.view(-1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        actions = self.fc4(x)
        return actions
    
    def save_checkpoint(self, epoch):
        print('... saving checkpoint ...')
        self.checkpoint_file=self.checkpoint_file+str(epoch)+'.pth'
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

class LightGCNConv(MessagePassing):
    r"""The neighbor aggregation operator from the `"LightGCN: Simplifying and
    Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126#>`_ paper

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        num_users (int): Number of users for recommendation.
        num_items (int): Number of items to recommend.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self,  in_channels: int, out_channels: int,
                 num_users: int, num_items: int, **kwargs):
        super(LightGCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_users = num_users
        self.num_items = num_items
        # self.device=device
        self.reset_parameters()

    def reset_parameters(self):
        pass  # There are no layer parameters to learn.

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """Performs neighborhood aggregation for user/item embeddings."""
        user_item = \
                torch.zeros(self.num_users, self.num_items, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        user_item[edge_index[:, 0], edge_index[:, 1]] = 1
        user_neighbor_counts = torch.sum(user_item, axis=1)
        item_neightbor_counts = torch.sum(user_item, axis=0)
        # Compute weight for aggregation: 1 / sqrt(N_u * N_i)
        weights = user_item / torch.sqrt(
                user_neighbor_counts.repeat(self.num_items, 1).T \
                * item_neightbor_counts.repeat(self.num_users, 1))
        weights = torch.nan_to_num(weights, nan=0)
        out = torch.concat((weights.T @ x[:self.num_users],
                            weights @ x[self.num_users:]), 0)
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class LightGCN(nn.Module):
    def __init__(self, 
                 config: dict,
                 device=None,
                 **kwargs):
        super().__init__()

        self.num_users  = config["n_users"]
        self.num_items  = config["m_items"]
        self.embedding_size = config["embedding_size"]
        self.in_channels = self.embedding_size
        self.out_channels = self.embedding_size
        self.num_layers = config["num_layers"]
        

        # 0-th layer embedding.
        self.embedding_user_item = torch.nn.Embedding(
            num_embeddings=self.num_users + self.num_items,
            embedding_dim=self.embedding_size)
        self.alpha = None
        
        # random normal init seems to be a better choice when lightGCN actually
        # don't use any non-linear activation function
        nn.init.normal_(self.embedding_user_item.weight, std=0.1)
        print('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()

        self.convs = nn.ModuleList()
        self.convs.append(LightGCNConv(
                self.embedding_size, self.embedding_size,
                num_users=self.num_users, num_items=self.num_items,  **kwargs))

        for _ in range(1, self.num_layers):
            self.convs.append(
                LightGCNConv(
                        self.embedding_size, self.embedding_size, 
                        num_users=self.num_users, num_items=self.num_items,
                        **kwargs))
        self.optimizer = optim.Adam(self.parameters(), lr=config["lr"])
        self.device = None
        if device is not None:
            self.convs.to(device)
            self.device = device
        
        self.to(self.device)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        xs: List[Tensor] = []

        edge_index = torch.nonzero(edge_index)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.device is not None:
                x = x.to(self.device)
            xs.append(x)
        xs = torch.stack(xs)
        
        self.alpha = 1 / (1 + self.num_layers) * torch.ones(xs.shape)
        if self.device is not None:
            self.alpha = self.alpha.to(self.device)
            xs = xs.to(self.device)
        x = (xs * self.alpha).sum(dim=0)  # Sum along K layers.
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')


class GRRS_Warm_Agent(object):
    def __init__(self, gamma, epsilon, alpha, model_config, batch_size,
                    max_mem_size=100000, eps_end=0.01, eps_dec=0.05, device="cpu"):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_MIN = eps_end
        self.EPS_DEC = eps_dec
        self.ALPHA = alpha
        self.action_space = [i for i in range(model_config["m_items"])]
        self.n_actions = model_config["m_items"]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.model_config=model_config
        self.input_dims=model_config["embedding_size"]
        self.lightGCN = LightGCN(config=self.model_config, device=device)
        self.lightGCN = torch.load('lightgcn_pretrain_model.pth',
                                        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device=device

        self.Q_eval = DeepQNetwork(alpha, n_actions=self.n_actions,
                              input_dims=self.input_dims, fc1_dims=1024, fc2_dims=2048, fc3_dims=4096, 
                              device=self.device)
        self.state_memory = np.zeros((self.mem_size, self.input_dims))
        self.new_state_memory = np.zeros((self.mem_size, self.input_dims))
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
    

    def storeTransition(self, transition, terminal):
        state = transition[0] 
        action = transition[1]
        state_ = transition[2]
        reward = transition[3]
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state.cpu().detach()
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_.cpu().detach()
        self.terminal_memory[index] = 1 - terminal
        self.mem_cntr += 1

    def chooseAction(self, observation, testing, watched_movies_id, K=2, movie_ctr=None,popularity=None):
        
        if not testing:
            rand = np.random.random()
            actions = self.Q_eval.forward(observation)
            actions_list=actions.tolist()
            if rand > self.EPSILON or testing:
                temp = sorted(actions_list, reverse=True)[:K]
                temp_full = sorted(actions_list, reverse=True)
                act = []
                act_all = []
                for ele in temp:
                    act.append(actions_list.index(ele))
                for ele in temp_full:
                    act_all.append(actions_list.index(ele))
            else:
                act_all=[i for i in range(self.n_actions)]
                act = random.sample(act_all, K)
            return act,  act_all
        else:
            rand = np.random.random()
            actions = self.Q_eval.forward(observation)
            actions_list=actions.tolist()
            for i in watched_movies_id:
                actions_list[i]=-np.inf
            for m in range(self.n_actions):
                if movie_ctr[m]<popularity:
                    actions_list[m]=-np.inf 

            if rand > self.EPSILON or testing:
                temp = sorted(actions_list, reverse=True)[:K]
                temp_full = sorted(actions_list, reverse=True)
                act = []
                act_all = []
                for ele in temp:
                    act.append(actions_list.index(ele))
                for ele in temp_full:
                    act_all.append(actions_list.index(ele))
            else:
                act_all=[i for i in range(self.n_actions)]
                act = random.sample(act_all, K)
                
            return act,  act_all

    def learn_warm(self):
                
            if self.mem_cntr > self.batch_size:
                self.lightGCN.eval()
                self.Q_eval.optimizer.zero_grad()

                max_mem = self.mem_cntr if self.mem_cntr < self.mem_size \
                                        else self.mem_size

                batch = np.random.choice(max_mem, self.batch_size, replace=False)
                state_batch = self.state_memory[batch]
                action_batch = self.action_memory[batch]
                reward_batch = self.reward_memory[batch]
                new_state_batch = self.new_state_memory[batch]
                terminal_batch = self.terminal_memory[batch]

                reward_batch = T.Tensor(reward_batch).to(self.Q_eval.device)
                terminal_batch = T.Tensor(terminal_batch).to(self.Q_eval.device)

                batch_index = np.arange(self.batch_size, dtype=np.int32)
                q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch].to(self.Q_eval.device)
                q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)
                q_next[terminal_batch.long()] = 0.0

                
                q_target = reward_batch + self.GAMMA*T.max(q_next, dim=1)[0]
            
                self.EPSILON = self.EPSILON-self.EPS_DEC if self.EPSILON > \
                            self.EPS_MIN else self.EPS_MIN

                loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
                loss.backward()
                self.Q_eval.optimizer.step()


    def learn_cool(self, data, train_mask):
        if self.mem_cntr > self.batch_size:
            samples_train=sample_pos_neg(data, train_mask)
            samples_train=samples_train.to(self.device)
            train_mask=train_mask.to(self.device)
            data = data.to(self.device)
            
            self.lightGCN.train()
            loss_sum = 0 
            self.lightGCN.optimizer.zero_grad()
            current_batch = samples_train
            users = current_batch[:, 0:1]
            pos = current_batch[:, 1:2]
            
            
            gloss, reg_loss = bpr_loss(self.lightGCN, users, pos, data)
            weight_decay = 0.1
            reg_loss = reg_loss * weight_decay
            gloss = gloss + reg_loss
            loss_sum += gloss.detach()

            gloss.backward()
            self.lightGCN.optimizer.step()

            if self.mem_cntr > self.batch_size:
                self.Q_eval.optimizer.zero_grad()

                max_mem = self.mem_cntr if self.mem_cntr < self.mem_size \
                                        else self.mem_size

                batch = np.random.choice(max_mem, self.batch_size, replace=False)
                state_batch = self.state_memory[batch]
                action_batch = self.action_memory[batch]
                reward_batch = self.reward_memory[batch]
                new_state_batch = self.new_state_memory[batch]
                terminal_batch = self.terminal_memory[batch]

                reward_batch = T.Tensor(reward_batch).to(self.Q_eval.device)
                terminal_batch = T.Tensor(terminal_batch).to(self.Q_eval.device)

                batch_index = np.arange(self.batch_size, dtype=np.int32)
                q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch].to(self.Q_eval.device)
                q_next = self.Q_eval.forward(new_state_batch).to(self.Q_eval.device)
                q_next[terminal_batch.long()] = 0.0

                
                q_target = reward_batch + self.GAMMA*T.max(q_next, dim=1)[0]
            
                self.EPSILON = self.EPSILON-self.EPS_DEC if self.EPSILON > \
                            self.EPS_MIN else self.EPS_MIN

                loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
                loss.backward()
                self.Q_eval.optimizer.step()
