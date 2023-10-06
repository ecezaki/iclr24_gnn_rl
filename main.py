import pandas as pd
import numpy as np
from gnn_dqn_agent import *
from preprocessing_small import preprocess
from feature_extractor import *
import random

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


import torch
from torch.functional import tensordot
from torch import nn, optim, Tensor
import torch_geometric
from torch_geometric.data import Dataset, Data, download_url, extract_zip
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
import pickle


# episodes, train_set, test_set = preprocess()

rating_threshold = 3  #@param {type: "integer"}: Ratings equal to or greater than 3 are positive items.

rnames = ['user_id','book_id','rating','u_id','b_id','ts']
raw_ratings = pd.read_table('ratings.dat', sep=',', 
                                header=rnames, engine='python',
                                encoding='latin-1')

config_dict = {
    "num_samples_per_user": 500,
    "num_users": len(set(raw_ratings['u_id'].values)),
    "num_items": len(set(raw_ratings['b_id'].values)),

    "epochs": 25,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.1,

    "embedding_size": 64,
    "num_layers": 4,
    "K": 20,
    "mf_rank": 8,

    "minibatch_per_print": 100,
    "epochs_per_print": 1,

    "val_frac": 0.2,
    "test_frac": 0.1,

    "model_name": "model.pth"
}
model_config = {
    "n_users": config_dict["num_users"],
    "m_items": config_dict["num_items"],
    "embedding_size": config_dict["embedding_size"],
    "num_layers": config_dict["num_layers"],
    "lr": 0.001
}

"""# Dataset

A great publicly available dataset for training movie recommenders is the MovieLens 1M dataset. The MovieLens 1M dataset consists of 1 million movie ratings of score 1 to 5, from 6000 users and 4000 movies.
"""

DATA_PATH = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
# getcontext().prec = 10


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device="cpu"


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


class Goodreads(Dataset):
    def __init__(self, root, transform=None, pre_transform=None,
            transform_args=None, pre_transform_args=None):
        """
        root = where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (process data).
        """
        super(Goodreads, self).__init__(root, transform, pre_transform)
        self.transform = transform
        self.pre_transform = pre_transform
        self.transform_args = transform_args
        self.pre_transform_args = pre_transform_args

    @property
    def raw_file_names(self):
        return "ML_100K_Text.zip"

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        download_url(DATA_PATH, self.raw_dir)

    def _load(self):
        rnames = ['user_id','book_id','rating','u_id','b_id','ts']
        ratings = pd.read_table('ratings.dat', sep=',', 
                                header=rnames,  engine='python',
                                encoding='latin-1')
        dat = ratings

        return ratings,  dat

    def process(self):
        print('run process')
        # load information from file
        ratings, dat = self._load()

        users = list(set(ratings['u_id'].values))
        movies = list(set(ratings['b_id'].values))

        num_users = config_dict["num_users"]
        if num_users != -1:
            users = users[:num_users]

        user_ids = range(len(users))
        movie_ids = range(len(movies))

        user_to_id = dict(zip(users, user_ids))
        movie_to_id = dict(zip(movies, movie_ids))

        # get adjacency info
        self.num_user = len(users)
        self.num_item = len(movies)

        # initialize the adjacency matrix
        rat = torch.zeros(self.num_user, self.num_item)

        for index, row in ratings.iterrows():
            user=row[8]
            movie=row[9]
            rating=row[5]
            # user, movie, rating = row[:3]
            if num_users != -1:
                if user not in user_to_id: break
            # create ratings matrix where (i, j) entry represents the ratings
            # of movie j given by user i.
            rat[user_to_id[user], movie_to_id[movie]] = rating

        # create Data object
        data = Data(edge_index = rat,
                    raw_edge_index = rat.clone(),
                    users = users,
                    items = movies)

        # apply any pre-transformation
        if self.pre_transform is not None:
            data = self.pre_transform(data, self.pre_transform_args)

        # apply any post_transformation
        # if self.transform is not None:
        #     # data = self.transform(data, self.transform_args)
        data = self.transform(data, [rating_threshold])

        # save the processed data into .pt file
        torch.save(data, osp.join(self.processed_dir, f'data.pt'))
        print('process finished')
      
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

with open('sessions.txt', 'rb') as f:
      episodes=pickle.load(f)
with open('train_sessions.txt', 'rb') as f:
      train_sessions=pickle.load(f)
with open('test_sessions.txt', 'rb') as f:
      test_sessions=pickle.load(f)




with open('test_user_item.txt', "rb") as f:
    test_user_item = pickle.load(f)
# episodes.to(device)
root = os.getcwd()
movielens = Goodreads(root=root, transform=trans_ml)
data = movielens.get()
ratings, dat = movielens._load()
# Movies_df=movies
raw=pd.read_csv("ratings.dat", sep=',')
users = list(set(raw['u_id'].values))
movies = list(set(raw['b_id'].values))

sessions_per_user=[]
test_sessions_new=[]
for user in users:
    s=[]
    for episode in episodes:
        if episode['u_id'].min()==user:
            s.append(episode['session_id'].min())
    sessions_per_user.append(s)
    # print(s)
    test_sessions_new.append(np.random.randint(0, len(s), 1))

# print(test_sessions_new)



# with open('valid_train_movies.txt', 'rb') as f:
#     movies=pickle.load(f)
print(f"#users={len(users)}")
print(f"#books={len(movies)}")
print(f"#train sessions={len(train_sessions)}")
print(f"#test sessions={len(test_sessions)}")
user_ids = range(len(users))
movie_ids = range(len(movies))

user_to_id = dict(zip(users, user_ids))
movie_to_id = dict(zip(movies, movie_ids))
id_to_movie = dict(zip(movies, movie_ids))



num_epochs=50
reward_test_hist = []
genre_reward_test_hist = []
tail_reward_hist = []
recall_avg_hist=[]

user_movies_all={}
for i in range(len(users)):
    user_movies_all[i]=[]

movie_ctr=np.zeros(len(movies))

for idx, episode in enumerate(episodes):
    user_movies_all[user_to_id[episode['u_id'].min()]].extend(episode['b_id'].values)

train_episodes=[]
test_episodes=[]
for episode in episodes:
    if episode['session_id'].min()  in train_sessions:
        train_episodes.append(episode)

for episode in episodes:
    if episode['session_id'].min() in test_sessions:
        test_episodes.append(episode)

print(f"#users={len(users)}")
print(f"#movies={len(movies)}")
print(f"#train sessions={len(train_episodes)}")
print(f"#test sessions={len(test_sessions)}")
num_epochs_list=[50]
gamma=0.95



for num_epochs in num_epochs_list:
    brain = WarmAgent(model_config=model_config, device=device, gamma=0.5, epsilon=1.0, batch_size=512, alpha=0.0005)
    ########################### Training########################################################
    print(data['edge_index'])
    all_users_items = brain.lightGCN.embedding_user_item.weight.cpu().detach()
    for epoch in range(num_epochs):
        # print(epoch)
        reward_test=0
        genre_reward=0
        tail_reward =0
        recall =[]
        movielens = Goodreads(root=root, transform=trans_ml)
        data = movielens.get()
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
        for episode in train_episodes:
            if episode['session_id'].min() in train_sessions:
                num_ep += 1
                print(f"Max epochs={num_epochs}, Epoch={epoch}, Episode={num_ep-1}/{len(train_sessions)}")
                user_id=episode['u_id'].min()

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
                
    
                    actual_movie_seen=episode['b_id'].values[t]
                    movie_ctr[movie_to_id[actual_movie_seen]] += 1
                    m_id = movie_to_id[actual_movie_seen]
                    rating = episode['rating'].values[t]
                
                
                    action_list_K,  action_list_full = brain.chooseAction(state, testing, w_h_d[user_to_id[user_id]])
                
                    reward = -1
                    if movie_to_id[actual_movie_seen] in action_list_K:
                        reward = 1

                    # recommended_action = action_list_K[0]


                    w_h_d_true = w_h_d
                    r_h_d_true = r_h_d
                    # for recommended_action in action_list_K:
                    recommended_action = action_list_K[0]

                    w_h_d[user_to_id[user_id]].append(recommended_action)
                    rew = 1 if recommended_action==movie_to_id[actual_movie_seen] and rating>=3 else -1
                    r_h_d[user_to_id[user_id]].append(rew)
                    temp=state_selection(brain.lightGCN, all_users_items, data, user_to_id[user_id], [w_h_d[user_to_id[user_id]][-1]], [r_h_d[user_to_id[user_id]][-1]], config_dict, feature_dim=config_dict["embedding_size"])
                    state_sim=temp.cpu().detach()
                    state_sim = temp

                    transition_sim=[state, recommended_action, state_sim, rew]

                    brain.storeTransition(transition=transition_sim, terminal=terminal)

                    


                    # rat=data_orig["edge_index"]
                    reward_orig= 1 if rating >= 3 else -1
                    # reward_orig = 1
                    
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

        for popularity in [10]:
            recall_per_user=[]
            for episode in episodes:
                if episode['session_id'].min() in test_sessions:
                    testing=True
                    user_id=episode['u_id'].min()
                    temp=state_selection(brain.lightGCN, all_users_items, data, user_to_id[user_id], w_h_d[user_to_id[user_id]], r_h_d[user_to_id[user_id]], config_dict, feature_dim=config_dict["embedding_size"])
                    # print(temp)
                    state=temp.clone().detach()
                    actual_movies_seen=episode['b_id'].values
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
                    new_m_ids=[]
                    for m in m_ids:
                        if movie_ctr[m]>popularity-1:
                            new_m_ids.append(m)
                    if len(new_action_list)>0:
                        recall_per_user.append(len(set(new_action_list).intersection(set(new_m_ids)))/len(new_action_list))     
            # print(round(float(recall_per_user), 6))
            if len(recall_per_user)>0:
                print(f"Popularity status={popularity}, Mean Recall K10={round(float(np.mean(recall_per_user)), 6)}")
                # print(f"Popularity status={popularity}, Max RecallK20={round(float(np.max(recall_per_user)), 6)}")
                # print(f"Popularity status={popularity}, Median RecallK20={round(float(np.median(recall_per_user)), 6)}")
        for popularity in [10]:
            recall_per_user=[]
            for episode in episodes:
                if episode['session_id'].min() in test_sessions:
                    testing=True
                    user_id=episode['u_id'].min()
                    temp=state_selection(brain.lightGCN, all_users_items, data, user_to_id[user_id], w_h_d[user_to_id[user_id]], r_h_d[user_to_id[user_id]], config_dict, feature_dim=config_dict["embedding_size"])
                    # print(temp)
                    state=temp.clone().detach()
                    actual_movies_seen=episode['b_id'].values
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
                            
                            
                    action_list_K,  action_list_full = brain.chooseAction(state, testing, w_h_d[user_to_id[user_id]], K=5, movie_ctr=movie_ctr, popularity=popularity)
                    new_action_list=[]
                    for m in action_list_K:
                        if movie_ctr[m]>popularity-1:
                            new_action_list.append(m)
                    new_m_ids=[]
                    for m in m_ids:
                        if movie_ctr[m]>popularity-1:
                            new_m_ids.append(m)
                    if len(new_action_list)>0:
                        recall_per_user.append(len(set(new_action_list).intersection(set(new_m_ids)))/len(new_action_list))     
            # print(round(float(recall_per_user), 6))
            if len(recall_per_user)>0:
                print(f"Popularity status={popularity}, Mean Recall K5={round(float(np.mean(recall_per_user)), 6)}")
                # print(f"Popularity status={popularity}, Max RecallK20={round(float(np.max(recall_per_user)), 6)}")
                # print(f"Popularity status={popularity}, Median RecallK20={round(float(np.median(recall_per_user)), 6)}")
        

        for popularity in [10]:
            recall_per_user=[]
            for episode in episodes:
                if episode['session_id'].min() in test_sessions:
                    testing=True
                    user_id=episode['u_id'].min()
                    temp=state_selection(brain.lightGCN, all_users_items, data, user_to_id[user_id], w_h_d[user_to_id[user_id]], r_h_d[user_to_id[user_id]], config_dict, feature_dim=config_dict["embedding_size"])
                    # print(temp)
                    state=temp.clone().detach()
                    actual_movies_seen=episode['b_id'].values

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
                            
                            
                    action_list_K,  action_list_full = brain.chooseAction(state, testing, w_h_d[user_to_id[user_id]], K=len(actual_movies_seen), movie_ctr=movie_ctr, popularity=popularity)
                    new_action_list=[]
                    for m in action_list_K:
                        if movie_ctr[m]>popularity-1:
                            new_action_list.append(m)
                    new_m_ids=[]
                    for m in m_ids:
                        if movie_ctr[m]>popularity-1:
                            new_m_ids.append(m)
                    if len(new_action_list)>0:
                        recall_per_user.append(len(set(new_action_list).intersection(set(new_m_ids)))/len(new_action_list))     
            # print(round(float(recall_per_user), 6))
            if len(recall_per_user)>0:
                print(f"Popularity status={popularity}, Mean Recall New={round(float(np.mean(recall_per_user)), 6)}")
                # print(f"Popularity status={popularity}, Max RecallK20={round(float(np.max(recall_per_user)), 6)}")
        
        popularity = 10
        for popularity in [5]:
                recall_per_user=[]
                for episode in episodes:
                    if episode['session_id'].min() in test_sessions:
                        testing=True
                        user_id=episode['u_id'].min()
                        state = all_users_items[user_to_id[user_id]]
              
                        actual_movies_seen=episode['b_id'].values

                        m_ids=[]
                        for movie in list(actual_movies_seen):
                            m_ids.append(movie)
                        
                        m_ids = list(set(m_ids).union(set(user_movies_all[user_to_id[user_id]])))
                        m_ids_new=[]
                        for movie in m_ids:
                            m_ids_new.append(movie_to_id[movie])
                        m_ids = m_ids_new
                                
                                
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
                            recall_per_user.append(len(set(recommended_movies).intersection(set(m_ids)))/K)     
                # print(round(float(recall_per_user), 6))
                if len(recall_per_user)>0:
                    print(f"Rec@10={round(float(np.mean(recall_per_user)), 6)}")
                    # print(f"ALPHA={alpha} Max RecallK20={round(float(np.max(recall_per_user)), 6)}")
                    # print(f"ALPHA={alpha} Median RecallK20={round(float(np.median(recall_per_user)), 6)}")        


        for popularity in [5]:
                recall_per_user=[]
                for episode in episodes:
                    if episode['session_id'].min() in test_sessions:
                        testing=True
                        user_id=episode['u_id'].min()
                        state = all_users_items[user_to_id[user_id]]
                
                        actual_movies_seen=episode['b_id'].values

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
                            K=10
                            recall_per_user.append(len(set(recommended_movies).intersection(set(m_ids)))/K)     
                # print(round(float(recall_per_user), 6))
                if len(recall_per_user)>0:
                    print(f"Rec@15={round(float(np.mean(recall_per_user)), 6)}")
                    # print(f"ALPHA={alpha} Max RecallK20={round(float(np.max(recall_per_user)), 6)}")
                    # print(f"ALPHA={alpha} Median RecallK20={round(float(np.median(recall_per_user)), 6)}")        


        for popularity in [5]:
                recall_per_user=[]
                for episode in episodes:
                    if episode['session_id'].min() in test_sessions:
                        testing=True
                        user_id=episode['u_id'].min()
                        state = all_users_items[user_to_id[user_id]]
            
                        actual_movies_seen=episode['b_id'].values

                        m_ids=[]
                        for movie in list(actual_movies_seen):
                            m_ids.append(movie)
                        
                        m_ids = list(set(m_ids).union(set(user_movies_all[user_to_id[user_id]])))
                        m_ids_new=[]
                        for movie in m_ids:
                            m_ids_new.append(movie_to_id[movie])
                        m_ids = m_ids_new
                                
                                
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
                        # if L>0:
                        if L>0:
                            K=10
                            recall_per_user.append(len(set(recommended_movies).intersection(set(m_ids)))/K)     
                # print(round(float(recall_per_user), 6))
                if len(recall_per_user)>0:
                    print(f"Rec@20={round(float(np.mean(recall_per_user)), 6)}")
                    # print(f"ALPHA={alpha} Max RecallK20={round(float(np.max(recall_per_user)), 6)}")
                    # print(f"ALPHA={alpha} Median RecallK20={round(float(np.median(recall_per_user)), 6)}")        
