import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
This code takes "ratings.dat" file as input which has user_id, item_id, 'rating' and 'ts' (timestamp) 
for every interaction.


The goal is to generate four binary files: "sessions.txt", "train_sessions.txt",
"test_sessions.txt", "test_user_item.txt"
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

column_names = ['user_id','item_id','rating','ts']
num_users=1000
ratings = pd.read_csv(r"ratings.dat", sep = ",", names = column_names, engine='python')
user_names_unique=list(set(ratings['user_id'].values))
user_ids = range(len(user_names_unique))
book_names_unique=list(set(ratings['item_id'].values))
item_ids = range(len(book_names_unique))
user_to_id = dict(zip(user_names_unique, user_ids))
book_to_id = dict(zip(book_names_unique, item_ids))
u_id=[]
for t in range(ratings.shape[0]):
  u_id.append(user_to_id[ratings['user_id'].values[t]])
ratings['u_id']=u_id
b_id=[]
for t in range(ratings.shape[0]):
  b_id.append(book_to_id[ratings['item_id'].values[t]])
ratings['b_id']=b_id



max_min=[]
for user in user_ids:
  data_user=ratings[ratings['u_id']==user]
  if data_user['ts'].min()>1484916795-86400*10*365 and data_user.shape[0]>300:
    max_min.append(user)



ratings=ratings.loc[ratings['u_id'].isin(max_min)]
all_books_size=max(ratings['b_id'].values)
book_ctr=np.zeros(all_books_size+1)

for t in range(ratings.shape[0]):
  book_ctr[ratings['b_id'].values[t]] += 1

most_popular_books=np.argsort(book_ctr)[-1000:]



ratings=ratings.loc[ratings['b_id'].isin(most_popular_books.tolist())]



data = ratings



session_th=86400*30*6
is_ordered=False
user_key='u_id'
item_key='b_id'
time_key='ts'
if not is_ordered:
  # sort data by user and time
	data.sort_values(by=[user_key, time_key], ascending=True, inplace=True)
start=data[time_key].values[0]
c_start=0
c_running=0
split_session=[True]

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



ratings=data

num_sessions_per_user=np.zeros(len(user_ids))
for sess_id in session_ids:
  num_sessions_per_user[data[data['session_id']==sess_id]['u_id'].min()] += 1

valid_users=[]
for user in range(len(user_ids)):
  if num_sessions_per_user[user]>4:
    valid_users.append(user)
print(f"Number of valid users={len(valid_users)}")
V=[]
for t in range(ratings.shape[0]):
  if ratings['u_id'].values[t] in valid_users:
    V.append(True)
  else:
    V.append(False)

ratings['Valid users'] = V

ratings=ratings[ratings['Valid users']==True]






# ratings.to_csv('GR_moreinteractions_669users.csv')



data=ratings
session_ids = list(set(data['session_id'].values))
print(len(session_ids))

all_sessions=[]
for sess in session_ids:
  all_sessions.append(data[data['session_id']==sess])

import pickle
train_sessions=[]
test_sessions=[]
for user in valid_users:
  all_valid_sessions_user=list(np.unique(np.array(data[data["u_id"]==user]["session_id"].values)))
  train_sessions.extend(all_valid_sessions_user[:-1])
  test_sessions.append(all_valid_sessions_user[-1])
train_sessions_long=[]
for sess in train_sessions:
  if data[data['session_id']==sess].shape[0]>6:
    train_sessions_long.append(sess)
test_user_item=[]
for sess in test_sessions:
  ep=data[data['session_id']==sess]
  for t in range(ep.shape[0]):
    test_user_item.append((ep['u_id'].values[t], ep['b_id'].values[t]))

print(len(all_sessions))
print(len(train_sessions))
print(len(train_sessions_long))
print(len(test_sessions))

with open('sessions.txt', 'wb') as f:
  pickle.dump(all_sessions,f)

with open('train_sessions.txt', 'wb') as f:
  pickle.dump(train_sessions,f)

with open('train_sessions_long.txt', 'wb') as f:
  pickle.dump(train_sessions_long,f)

with open('test_sessions.txt', 'wb') as f:
  pickle.dump(test_sessions, f)

with open('test_user_item.txt', 'wb') as f:
  pickle.dump(test_user_item, f)
