import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
This code takes "ratings.dat" file as input and generates four binary files: "sessions.txt", "train_sessions.txt",
"test_sessions.txt", "test_user_item.txt"
"""

column_names = ['user_id', 'item_id', 'rating', 'ts']
num_users=670
ratings = pd.read_csv(r"ratings.dat", sep = "::", names = column_names, engine='python')
ratings=ratings[ratings['user_id']<=num_users]
data = ratings
# print(data)

session_th=1800
is_ordered=False
user_key='user_id'
item_key='item_id'
time_key='ts'
if not is_ordered:
  # sort data by user and time
	data.sort_values(by=[time_key, user_key], ascending=True, inplace=True)
tdiff=np.diff(data[time_key].values)
split_session = tdiff > session_th
split_session = np.r_[True, split_session]
new_user = data[user_key].values[1:]!= data[user_key].values[:-1]
new_user_orig=new_user
new_user = np.r_[True, new_user]
new_session = np.logical_or(new_user, split_session)
session_ids = np.cumsum(new_session)
test_session = np.r_[False, new_user_orig]
data['session_id'] = session_ids
data['test_session'] = test_session
session_lengths=[]
session_ids = list(set(data['session_id'].values))
for sess_id in session_ids:
	session_lengths.append((data[data['session_id']==sess_id]).shape[0])



num_sessions_per_user=np.zeros(num_users)
for sess_id in session_ids:
  num_sessions_per_user[data[data['session_id']==sess_id]['user_id'].min()-1] += 1

valid_users=[]
for user in range(num_users):
  if num_sessions_per_user[user]>4:
    valid_users.append(user+1)


import pickle
with open('valid_users.txt', 'wb') as f:
  pickle.dump(valid_users,f)

all_valid_sessions=[]
train_sessions=[]
test_sessions=[]
for sess in session_ids:
  if data[data['session_id']==sess]['user_id'].min() in valid_users:
    all_valid_sessions.append(data[data['session_id']==sess])

with open('sessions.txt', 'wb') as f:
  pickle.dump(all_valid_sessions, f)

all_valid_sessions_long=[]


for user in valid_users:
  all_valid_sessions_user=list(set(np.unique(np.array(data[data["user_id"]==user]["session_id"].values))))
  train_sessions.extend(all_valid_sessions_user[:-1])
  test_sessions.append(all_valid_sessions_user[-1])

train_sessions_long=[]
long_valid_users=[]
for id in train_sessions:
  if data[data['session_id']==id].shape[0]>9:
    train_sessions_long.append(id)
    long_valid_users.append(data['user_id'].min())

test_sessions_valid=[]
for sess in test_sessions:
  if data[data["session_id"]==sess]['user_id'].min() in long_valid_users:
    test_sessions_valid.append(sess)


file = "train_sessions.txt"
with open(file, "wb") as f:
    pickle.dump(train_sessions_long, f)
file = "test_sessions.txt"
with open(file, "wb") as f:
    pickle.dump(test_sessions_valid, f)

test_user_item=[]

for episode in all_valid_sessions:
   if episode['session_id'].min() not in  train_sessions_long:
      for t in range(episode.shape[0]):
         test_user_item.append((episode['user_id'].values[t], episode['item_id'].values[t]))

file = "test_user_item.txt"
with open(file,"wb") as f:
   pickle.dump(test_user_item,f)
