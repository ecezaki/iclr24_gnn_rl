import pandas as pd
import numpy as np
from gnn_dqn_agent import *
from feature_extractor import *

import collections
from collections import Counter
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
#from tensorly import decomposition

import torch
from torch.functional import tensordot
from torch import nn, optim, Tensor
import torch_geometric
from torch_geometric.data import Dataset, Data, download_url, extract_zip
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
import pickle



rating_threshold = 3  #@param {type: "integer"}: Ratings equal to or greater than 3 are positive items.

config_dict = {
    "num_samples_per_user": 500,
    "num_users": 670,
    "num_items": 3883,

    "epochs": 100,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.1,

    "embedding_size": 64,
    "num_layers": 100,
    "K": 10,
    "mf_rank": 8,

    "minibatch_per_print": 100,
    "epochs_per_print": 1,

    "val_frac": 0.2,
    "test_frac": 0.1,

    "model_name": "model.pth"
}
model_config = {
    "n_users": 670,
    "m_items": 3883,
    "embedding_size": config_dict["embedding_size"],
    "num_layers": config_dict["num_layers"],
    "lr": 0.001
}

DATA_PATH = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def trans_ml(dat, thres):
    """
    Transform function that assign non-negative entries >= thres 1, and non-
    negative entries <= thres 0. Keep other entries the same.
    """
    thres = thres[0]
    matrix = dat['edge_index']
    matrix[(matrix < thres) & (matrix > -1)] = 0
    matrix[(matrix >= thres)] = 1
    dat['edge_index'] = matrix
    return dat


class MovieLens(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,
            transform_args=None, pre_transform_args=None):
        """
        root = where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (process data).
        """
        super(MovieLens, self).__init__(root, transform, pre_transform)
        self.transform = transform
        self.pre_transform = pre_transform
        self.transform_args = transform_args
        self.pre_transform_args = pre_transform_args

    @property
    def raw_file_names(self):
        return "ml-1m.zip"

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        download_url(DATA_PATH, self.raw_dir)

    def _load(self):
        with zipfile.ZipFile(self.raw_paths[0], 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
        users = pd.read_table('users.dat', 
                              sep='::', header=None, names=unames,
                              engine='python', encoding='latin-1')
        rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
        ratings = pd.read_table('ratings.dat', sep='::', 
                                header=None, names=rnames, engine='python',
                                encoding='latin-1')
        mnames = ['movie_id', 'title', 'genres']
        movies = pd.read_table('movies.dat', sep='::', 
                               header=None, names=mnames, engine='python',
                               encoding='latin-1')
        dat = pd.merge(pd.merge(ratings, users), movies)

        return users, ratings, movies, dat

    def process(self):
        # print('run process')
        # load information from file
        users, ratings, movies, dat = self._load()

        users = users['user_id']
        movies = movies['movie_id']

        num_users = config_dict["num_users"]
        if num_users != -1:
            users = users[:num_users]


        # get adjacency info
        self.num_user = users.shape[0]
        self.num_item = movies.shape[0]

        # initialize the adjacency matrix
        rat = torch.zeros(self.num_user, self.num_item)


        # create Data object
        data = Data(edge_index = rat,
                    raw_edge_index = rat.clone(),
                    data = ratings,
                    users = users,
                    items = movies)

        # apply any pre-transformation
        if self.pre_transform is not None:
            data = self.pre_transform(data, self.pre_transform_args)

        data = self.transform(data, [rating_threshold])

        # save the processed data into .pt file
        torch.save(data, osp.join(self.processed_dir, f'data.pt'))
        # print('process finished')
      
    def len(self):
        """
        return the number of examples in your graph
        """
        # TODO: how to define number of examples
        return 

    def get(self):
        """
        The logic to load a single graph
        """
        data = torch.load(osp.join(self.processed_dir, 'data.pt'))
        return data




# episodes, train_set, test_set = preprocess()
with open('sessions.txt', 'rb') as f:
      episodes=pickle.load(f)
with open('train_sessions.txt', 'rb') as f:
      train_sessions=pickle.load(f)
with open('test_sessions.txt', 'rb') as f:
      test_sessions=pickle.load(f)
# episodes.to(device)
root = os.getcwd()
movielens = MovieLens(root=root, transform=trans_ml)
data = movielens.get()


users, ratings, movies, dat = movielens._load()
Movies_df=movies
users = users['user_id']
movies = movies['movie_id']
user_ids = range(len(users))
movie_ids = range(len(movies))

user_to_id = dict(zip(users, user_ids))
movie_to_id = dict(zip(movies, movie_ids))
id_to_movie = dict(zip(movies, movie_ids))



num_epochs_list=[100]
reward_test_hist = []
genre_reward_test_hist = []
tail_reward_hist = []
recall_avg_hist=[]

user_movies_all={}
for i in range(len(users)):
    user_movies_all[i]=[]

movie_ctr=np.zeros(len(movies))

for idx, episode in enumerate(episodes):
    user_movies_all[user_to_id[episode['user_id'].min()]].extend(episode['item_id'].values)



for num_epochs in num_epochs_list:
    brain = GRRS_Warm_Agent(model_config=model_config, device=device, gamma=0.9, epsilon=1.0, batch_size=512, alpha=0.001)
    ########################### Training########################################################
    # print(data['edge_index'])
    all_users_items = brain.lightGCN.embedding_user_item.weight.cpu().detach()
    rec10_list=[]
    rec15_list=[]
    rec20_list=[]
    for epoch in range(num_epochs):
        print(f"Epoch={epoch}/{num_epochs}")
        # print(epoch)
        reward_test=0
        genre_reward=0
        tail_reward =0
        recall =[]
        movielens = MovieLens(root=root, transform=trans_ml)
        data = movielens.get()
        users, ratings, movies, dat = movielens._load()
        users = users['user_id']
        movies = movies['movie_id']
        user_ids = range(len(users))
        movie_ids = range(len(movies))
        w_h_d={} # all users watched history
        r_h_d={} # all users rating history
        
        for i in range(len(users)):
            w_h_d[i]=[]
            r_h_d[i]=[]
        
        user_to_id = dict(zip(users, user_ids))
        movie_to_id = dict(zip(movies, movie_ids))
        id_to_movie = dict(zip(movie_ids, movies))
        # print(len(episodes))
        num_ep=0
        seq_reward_hist=[]
        actions_recommeded_hist=[]
        movie_ctr=np.zeros(config_dict["num_items"])
        for episode in episodes:
            if episode['session_id'].min() in train_sessions:
                num_ep += 1
                # print(f"Max epochs={num_epochs}, Epoch={epoch}, Episode={num_ep-1}")
                user_id=episode['user_id'].min()

                user_index_list=[user_to_id[user_id]]
                temp=state_selection(brain.lightGCN,all_users_items, data, user_to_id[user_id], w_h_d[user_to_id[user_id]], r_h_d[user_to_id[user_id]], config_dict, feature_dim=config_dict["embedding_size"])
                # print(temp)
                state=temp.cpu().detach()
                # state = torch.tensor(temp, dtype=torch.float32)

                ctr=0
            
                for t in range(episode.shape[0]):
                    # print("user_id={user_id}, t={t}")
                    testing=False
                    terminal=False

                    if t == episode.shape[0]-1:
                        terminal=True
                
    
                    actual_movie_seen=episode['item_id'].values[t]
                    movie_ctr[movie_to_id[actual_movie_seen]] += 1
                    m_id = movie_to_id[actual_movie_seen]
                    actual_genre=Movies_df['genres'][m_id].split('|') 
        
                    rating = episode['rating'].values[t]
                
                
                    action_list_K,  action_list_full = brain.chooseAction(state, testing, w_h_d[user_to_id[user_id]])
                
                    reward = -1
                    if movie_to_id[actual_movie_seen] in action_list_K:
                        reward = 1



                    w_h_d_true = w_h_d
                    r_h_d_true = r_h_d
                    recommended_action = action_list_K[0]

                    w_h_d[user_to_id[user_id]].append(recommended_action)
                    rew = 1 if recommended_action==movie_to_id[actual_movie_seen] and rating>=3 else -1
                    r_h_d[user_to_id[user_id]].append(rew)
                    temp=state_selection(brain.lightGCN, all_users_items, data, user_to_id[user_id], [w_h_d[user_to_id[user_id]][-1]], [r_h_d[user_to_id[user_id]][-1]], config_dict, feature_dim=config_dict["embedding_size"])
                    state_sim=temp.cpu().detach()
                    state_sim = temp

                    transition_sim=[state, recommended_action, state_sim, rew]

                    brain.storeTransition(transition=transition_sim, terminal=terminal)

                    

                    reward_orig= 1 if rating >= 3 else -1
                    
                    w_h_d = w_h_d_true
                    r_h_d = r_h_d_true
                    w_h_d[user_to_id[user_id]].append(movie_to_id[actual_movie_seen])
                    r_h_d[user_to_id[user_id]].append(reward_orig)
                    temp=state_selection(brain.lightGCN, all_users_items, data, user_to_id[user_id], [w_h_d[user_to_id[user_id]][-1]], [r_h_d[user_to_id[user_id]][-1]], config_dict, feature_dim=config_dict["embedding_size"])
                    # print(temp)
                    state_=temp.cpu().detach()
                    # state_=torch.tensor(temp, dtype=torch.float32)
                    transition=[state, m_id, state_, reward_orig]
                    data['edge_index'][user_to_id[user_id],movie_to_id[actual_movie_seen]] = 1
                    # brain.storeTransition(transition=transition, terminal=terminal)


                
                    ctr += 1
                    # if not testing:
                    brain.learn_warm() 
                    # print(f"counter={ctr}")
                    state = state_
                
                # all_users_items = brain.lightGCN.forward(brain.lightGCN.embedding_user_item.weight.clone(), data['edge_index'])


    ###################### Testing ###############################################
        

        
      
        for popularity in [5]:
                recall_per_user=[]
                for episode in episodes:
                    if episode['session_id'].min() in test_sessions:
                        testing=True
                        user_id=episode['user_id'].min()
                        state = all_users_items[user_to_id[user_id]]
                        # temp=state_selection(brain.lightGCN, all_users_items, data, user_to_id[user_id], [], [], config_dict, feature_dim=config_dict["embedding_size"])
                        # # print(temp)
                        # state=temp.detach().clone()

                        actual_movies_seen=episode['item_id'].values

                        m_ids=[]
                        for movie in list(actual_movies_seen):
                            m_ids.append(movie)
                        
                        m_ids = list(set(m_ids).union(set(user_movies_all[user_to_id[user_id]])))
                        m_ids_new=[]
                        for movie in m_ids:
                            m_ids_new.append(movie_to_id[movie])
                        m_ids = m_ids_new
                        # actual_genre=Movies_df['genres'][m_id].split('|') 
                        
                        # rating = episode['rating'].values[t]
                                
                                
                        action_list_K,  action_list_full = brain.chooseAction(state, testing, w_h_d[user_to_id[user_id]], K=10, movie_ctr=movie_ctr, popularity=popularity)
                        new_action_list=[]
                        for m in action_list_K:
                            if movie_ctr[m]>popularity-1:
                                new_action_list.append(m)
                        recommended_movies=new_action_list
                  
                        new_m_ids=[]
                        for m in m_ids:
                            if movie_ctr[m]>1:
                                new_m_ids.append(m)
                        L=len(new_m_ids)- len(set(w_h_d[user_to_id[user_id]]))
                        # if L>0:
                        if L>0:
                            K=10
                            recall_per_user.append(len(set(recommended_movies).intersection(set(m_ids)))/L)     
                # print(round(float(recall_per_user), 6))
                if len(recall_per_user)>0:
                    # print(f"Rec@10={round(float(np.mean(recall_per_user)), 6)}")
                    rec10_list.append(np.mean(recall_per_user))
                    # print(f"ALPHA={alpha} Max RecallK20={round(float(np.max(recall_per_user)), 6)}")
                    # print(f"ALPHA={alpha} Median RecallK20={round(float(np.median(recall_per_user)), 6)}")        


        for popularity in [5]:
                recall_per_user=[]
                for episode in episodes:
                    if episode['session_id'].min() in test_sessions:
                        testing=True
                        user_id=episode['user_id'].min()
                        state = all_users_items[user_to_id[user_id]]
                        # temp=state_selection(brain.lightGCN, all_users_items, data, user_to_id[user_id], [], [], config_dict, feature_dim=config_dict["embedding_size"])
                        # # print(temp)
                        # state=temp.detach().clone()

                        actual_movies_seen=episode['item_id'].values

                        m_ids=[]
                        for movie in list(actual_movies_seen):
                            m_ids.append(movie)
                        
                        m_ids = list(set(m_ids).union(set(user_movies_all[user_to_id[user_id]])))
                        m_ids_new=[]
                        for movie in m_ids:
                            m_ids_new.append(movie_to_id[movie])
                        m_ids = m_ids_new
                        # actual_genre=Movies_df['genres'][m_id].split('|') 
                        
                        # rating = episode['rating'].values[t]
                                
                                
                        action_list_K,  action_list_full = brain.chooseAction(state, testing, w_h_d[user_to_id[user_id]], K=15, movie_ctr=movie_ctr, popularity=popularity)
                        new_action_list=[]
                        for m in action_list_K:
                            if movie_ctr[m]>popularity-1:
                                new_action_list.append(m)
                        recommended_movies=new_action_list
                      
                        new_m_ids=[]
                        for m in m_ids:
                            if movie_ctr[m]>1:
                                new_m_ids.append(m)
                        L=len(new_m_ids)- len(set(w_h_d[user_to_id[user_id]]))
                        # if L>0:
                        if L>0:
                            K=15
                            recall_per_user.append(len(set(recommended_movies).intersection(set(m_ids)))/L)     
                # print(round(float(recall_per_user), 6))
                if len(recall_per_user)>0:
                    # print(f"Rec@15={round(float(np.mean(recall_per_user)), 6)}")
                    rec15_list.append(np.mean(recall_per_user))
                    # print(f"ALPHA={alpha} Max RecallK20={round(float(np.max(recall_per_user)), 6)}")
                    # print(f"ALPHA={alpha} Median RecallK20={round(float(np.median(recall_per_user)), 6)}")        


        for popularity in [5]:
                recall_per_user=[]
                for episode in episodes:
                    if episode['session_id'].min() in test_sessions:
                        testing=True
                        user_id=episode['user_id'].min()
                        state = all_users_items[user_to_id[user_id]]
                        # temp=state_selection(brain.lightGCN, all_users_items, data, user_to_id[user_id], [], [], config_dict, feature_dim=config_dict["embedding_size"])
                        # # print(temp)
                        # state=temp.detach().clone()

                        actual_movies_seen=episode['item_id'].values

                        m_ids=[]
                        for movie in list(actual_movies_seen):
                            m_ids.append(movie)
                        
                        m_ids = list(set(m_ids).union(set(user_movies_all[user_to_id[user_id]])))
                        m_ids_new=[]
                        for movie in m_ids:
                            m_ids_new.append(movie_to_id[movie])
                        m_ids = m_ids_new
                        # actual_genre=Movies_df['genres'][m_id].split('|') 
                        
                        # rating = episode['rating'].values[t]
                                
                                
                        action_list_K,  action_list_full = brain.chooseAction(state, testing, w_h_d[user_to_id[user_id]], K=20, movie_ctr=movie_ctr, popularity=popularity)
                        new_action_list=[]
                        for m in action_list_K:
                            if movie_ctr[m]>popularity-1:
                                new_action_list.append(m)
                        recommended_movies=new_action_list
                      
                        new_m_ids=[]
                        for m in m_ids:
                            if movie_ctr[m]>1:
                                new_m_ids.append(m)
                        L=len(new_m_ids)- len(set(w_h_d[user_to_id[user_id]]))
                        if L>0:
                            K=20
                            recall_per_user.append(len(set(recommended_movies).intersection(set(m_ids)))/L)     
                if len(recall_per_user)>0:
                    # print(f"Rec@20={round(float(np.mean(recall_per_user)), 6)}")
                    rec20_list.append(np.mean(recall_per_user))


    print(f'Recall@10={max(rec10_list)}')
    print(f'Recall@15={max(rec15_list)}')
    print(f'Recall@20={max(rec20_list)}')
