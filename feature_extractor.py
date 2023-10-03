import numpy as np
import torch

# def modified_feature_extractor_normalized(user_id, watched_hist, rating_hist, config_dict, feature_dim=31):
#     if len(watched_hist)>(feature_dim-1)/2:
#         user_feat = [user_id/config_dict["num_users"], *list(np.array(watched_hist[int(-(feature_dim-1)/2):])/config_dict["num_items"]), *rating_hist[int(-(feature_dim-1)/2):]]
#     else:
#         len_hist=len(watched_hist)
#         user_feat = [user_id/config_dict["num_users"], *list(np.array(watched_hist)/config_dict["num_items"]), *(list(np.zeros(int((feature_dim-1)/2-len_hist)))), *rating_hist,  *(list(np.zeros(int((feature_dim-1)/2-len_hist))))]
#     return user_feat

def modified_feature_extractor(model, data, user_id, watched_hist, rating_hist, config_dict, feature_dim=128):
    if len(watched_hist)>9:
        all_users_items = model(model.embedding_user_item.weight.clone(),
                            data["edge_index"])
        all_users = all_users_items[:model.num_users]
        all_items = all_users_items[model.num_users:]


        xs: List[Tensor] = []
        for i in range(10):
            x= all_items[watched_hist[-(i+1)]]
            if model.device is not None:
                x=x.to(model.device)
            xs.append(x)
        xs = torch.stack(xs)
        alpha = 1 / (10) * torch.ones(xs.shape)
        if model.device is not None:
            alpha = alpha.to(model.device)
            xs = xs.to(model.device)
        x = (xs * alpha).sum(dim=0)  # Sum along last 10 item features.
        user_emb=all_users[user_id]
        user_feat=torch.cat((user_emb, x), dim=0)


        # user_feat = [user_id, *watched_hist[int(-(feature_dim-1)/2):], *rating_hist[int(-(feature_dim-1)/2):]]
    elif len(watched_hist)>0:
        # print(len(watched_hist))
        len_hist=len(watched_hist)
        all_users_items = model(model.embedding_user_item.weight.clone(),
                            data["edge_index"])
        # print(all_users_items.shape)
        all_users = all_users_items[:model.num_users]
        all_items = all_users_items[model.num_users:]


        xs: List[Tensor] = []
        for i in range(len_hist):
            x= all_items[watched_hist[-(i+1)]]
            if model.device is not None:
                x=x.to(model.device)
            xs.append(x)
        xs = torch.stack(xs)
        alpha = 1 / (len_hist) * torch.ones(xs.shape)
        if model.device is not None:
            alpha = alpha.to(model.device)
            xs = xs.to(model.device)
        x = (xs * alpha).sum(dim=0)  # Sum along last 10 item features.
        user_emb=all_users[user_id]
        user_feat=torch.cat((user_emb, x), dim=0)
    else:
        len_hist=len(watched_hist)
        all_users_items = model(model.embedding_user_item.weight.clone(),
                            data["edge_index"])
        all_users = all_users_items[:model.num_users]
        all_items = all_users_items[model.num_users:]


        # xs: List[Tensor] = []
        # for i in range(len_hist):
        #     x= torch.zeros(64)
        #     if model.device is not None:
        #         x=x.to(self.device)
        #     xs.append(x)
        # xs = torch.stack(xs)
        # alpha = torch.ones(xs.shape)
        # if self.device is not None:
        #     alpha = alpha.to(self.device)
        #     xs = xs.to(self.device)
        x = torch.zeros(64)
        x=x.to(model.device)

        user_emb=all_users[user_id]
        user_feat=torch.cat((user_emb, x), dim=0)
        

    return user_feat

def modified_feature_extractor_64(model, all_users_items,data, user_id, watched_hist, rating_hist, config_dict, feature_dim=64):
    # if len(watched_hist)>9:
    if False:
        all_users_items = model(model.embedding_user_item.weight.clone(),
                            data["edge_index"])
        all_users = all_users_items[:model.num_users]
        all_items = all_users_items[model.num_users:]


        xs: List[Tensor] = []
        for i in range(10):
            x= all_items[watched_hist[-(i+1)]] # rating x movie feature vector
            # print(type(x))
            # x = torch.tensor(x, dtype=torch.float)
            # x = x/torch.norm(x)
            if model.device is not None:
                x=x.to(model.device)
            xs.append(x)
        # x = all_users[user_id]
        # x = torch.tensor(x, dtype=torch.float)
        # x = x/torch.norm(x)
        # x = x.to(model.device)
        # xs.append(x)
        xs = torch.stack(xs)
        alpha = (1/10)*torch.ones(xs.shape) #changed from 1/10 to 1/1
        if model.device is not None:
            alpha = alpha.to(model.device)
            xs = xs.to(model.device)
        x = (xs * alpha).sum(dim=0)  # Sum all the features across dimensions to get a  64 dimensional feature
        # user_emb=all_users[user_id]
        # user_feat=torch.cat((user_emb, x), dim=0)
        user_feat = x


        # user_feat = [user_id, *watched_hist[int(-(feature_dim-1)/2):], *rating_hist[int(-(feature_dim-1)/2):]]
    else:
        # print(len(watched_hist))
        len_hist=len(watched_hist)
        # all_users_items = model(model.embedding_user_item.weight.clone(),
        #                     data["edge_index"])
        # print(all_users_items.shape)
        all_users = all_users_items[:model.num_users]
        all_items = all_users_items[model.num_users:]


        xs: List[Tensor] = []
        for i in range(len_hist):
            x= all_items[watched_hist[-(i+1)]] # rating x movie feature vector
            # x = torch.tensor(x, dtype=torch.float)
            # x = x/torch.norm(x)
            if model.device is not None:
                x=x.to(model.device)
            xs.append(x)
        x = all_users[user_id]
        # print(type(x))
        # x = torch.tensor(x, dtype=torch.float)
        # x = x/torch.norm(x)
        x=x.to(model.device)
        xs.append(x)
        xs = torch.stack(xs)
        alpha = 1 / (len_hist+1) * torch.ones(xs.shape) #changed from 1/10 to 1/1
        if model.device is not None:
            alpha = alpha.to(model.device)
            xs = xs.to(model.device)
        x = (xs * alpha).sum(dim=0)  # Sum all the features across dimensions to get a  64 dimensional feature
        # user_emb=all_users[user_id]
        # user_feat=torch.cat((user_emb, x), dim=0)
        user_feat = x

        

    return user_feat

def state_selection(model, all_users_items, data, user_id, watched_hist, rating_hist, config_dict, feature_dim=64):
            
        len_hist=min(len(watched_hist), 1)
        
        # print(all_users_items.shape)
        all_users = all_users_items[:model.num_users]
        all_items = all_users_items[model.num_users:]


        xs: List[Tensor] = []
        for i in range(len_hist):
            x= all_items[watched_hist[-(i+1)]] # rating x movie feature vector
            # x = torch.tensor(x, dtype=torch.float)
            # x = x/torch.norm(x)
            if model.device is not None:
                x=x.to(model.device)
            xs.append(x)
        x = all_users[user_id]
        # print(type(x))
        # x = torch.tensor(x, dtype=torch.float)
        # x = x/torch.norm(x)
        x=x.to(model.device)
        xs.append(x)
        xs = torch.stack(xs)
        alpha = 1 / (len_hist+1) * torch.ones(xs.shape) #changed from 1/10 to 1/1
        if model.device is not None:
            alpha = alpha.to(model.device)
            xs = xs.to(model.device)
        x = (xs * alpha).sum(dim=0)  # Sum all the features across dimensions to get a  64 dimensional feature
        user_feat = x

        

        return user_feat

def modified_feature_extractor_64_episodewise_weighted(model, all_users_items, data, user_id, watched_hist, rating_hist, config_dict, feature_dim=64):
    # if len(watched_hist)>9:
    if False:
        all_users_items = model(model.embedding_user_item.weight.clone(),
                            data["edge_index"])
        all_users = all_users_items[:model.num_users]
        all_items = all_users_items[model.num_users:]


        xs: List[Tensor] = []
        for i in range(10):
            x= all_items[watched_hist[-(i+1)]] # rating x movie feature vector
            # print(type(x))
            # x = torch.tensor(x, dtype=torch.float)
            # x = x/torch.norm(x)
            if model.device is not None:
                x=x.to(model.device)
            xs.append(x)
        # x = all_users[user_id]
        # x = torch.tensor(x, dtype=torch.float)
        # x = x/torch.norm(x)
        # x = x.to(model.device)
        # xs.append(x)
        xs = torch.stack(xs)
        alpha = (1/10)*torch.ones(xs.shape) #changed from 1/10 to 1/1
        if model.device is not None:
            alpha = alpha.to(model.device)
            xs = xs.to(model.device)
        x = (xs * alpha).sum(dim=0)  # Sum all the features across dimensions to get a  64 dimensional feature
        # user_emb=all_users[user_id]
        # user_feat=torch.cat((user_emb, x), dim=0)
        user_feat = x


        # user_feat = [user_id, *watched_hist[int(-(feature_dim-1)/2):], *rating_hist[int(-(feature_dim-1)/2):]]
    else:
        # print(len(watched_hist))
        # if model is not None:
        #     all_users_items = model(model.embedding_user_item.weight.clone(), data["edge_index"])
            
        len_hist=len(watched_hist)
        
        # print(all_users_items.shape)
        all_users = all_users_items[:model.num_users]
        all_items = all_users_items[model.num_users:]


        xs: List[Tensor] = []
        for i in range(len_hist):
            x= rating_hist[-(i+1)]*all_items[watched_hist[-(i+1)]] # rating x movie feature vector
            # x = torch.tensor(x, dtype=torch.float)
            # x = x/torch.norm(x)
            if model.device is not None:
                x=x.to(model.device)
            xs.append(x)
        x = all_users[user_id]
        # print(type(x))
        # x = torch.tensor(x, dtype=torch.float)
        # x = x/torch.norm(x)
        x=x.to(model.device)
        xs.append(x)
        xs = torch.stack(xs)
        alpha = 1 / (len_hist+1) * torch.ones(xs.shape) #changed from 1/10 to 1/1
        if model.device is not None:
            alpha = alpha.to(model.device)
            xs = xs.to(model.device)
        x = (xs * alpha).sum(dim=0)  # Sum all the features across dimensions to get a  64 dimensional feature
        # user_emb=all_users[user_id]
        # user_feat=torch.cat((user_emb, x), dim=0)
        user_feat = x

        

    return user_feat

def modified_feature_extractor_64_diff_state(model, data, user_id, watched_hist, rating_hist, config_dict, feature_dim=64):
    liked_items=[]
    for idx, movie in enumerate(watched_hist):
        if rating_hist[idx]> 0:
            liked_items.append(movie)

    if len(liked_items)>15:
        all_users_items = model(model.embedding_user_item.weight.clone(),
                            data["edge_index"])
        all_users = all_users_items[:model.num_users]
        all_items = all_users_items[model.num_users:]


        xs: List[Tensor] = []
        for i in range(15):
            x= all_items[liked_items[-(i+1)]] # movie feature vector
            if model.device is not None:
                x=x.to(model.device)
            xs.append(x)
        # x = 2*all_users[user_id]
        # x=x.to(model.device)
        # xs.append(x)
        xs = torch.stack(xs)
        alpha = 1 / (15) * torch.ones(xs.shape) #changed from 1/10 to 1/1
        if model.device is not None:
            alpha = alpha.to(model.device)
            xs = xs.to(model.device)
        x = (xs * alpha).sum(dim=0)  # Sum all the features across dimensions to get a  64 dimensional feature
        # user_emb=all_users[user_id]
        # user_feat=torch.cat((user_emb, x), dim=0)
        user_feat = x


        # user_feat = [user_id, *watched_hist[int(-(feature_dim-1)/2):], *rating_hist[int(-(feature_dim-1)/2):]]
    else:
        # print(len(watched_hist))
        len_hist=len(liked_items)
        all_users_items = model(model.embedding_user_item.weight.clone(),
                            data["edge_index"])
        # print(all_users_items.shape)
        all_users = all_users_items[:model.num_users]
        all_items = all_users_items[model.num_users:]


        xs: List[Tensor] = []
        for i in range(len_hist):
            x= all_items[liked_items[-(i+1)]] 
            if model.device is not None:
                x=x.to(model.device)
            xs.append(x)
        x = all_users[user_id]
        x=x.to(model.device)
        xs.append(x)
        xs = torch.stack(xs)
        alpha = 1 / (len_hist+1) * torch.ones(xs.shape) #changed from 1/10 to 1/1
        if model.device is not None:
            alpha = alpha.to(model.device)
            xs = xs.to(model.device)
        x = (xs * alpha).sum(dim=0)  # Sum all the features across dimensions to get a  64 dimensional feature
        # user_emb=all_users[user_id]
        # user_feat=torch.cat((user_emb, x), dim=0)
        user_feat = x

        

    return user_feat
