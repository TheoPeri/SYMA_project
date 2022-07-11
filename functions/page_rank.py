import pandas as pd
import numpy as np
from zipfile import ZipFile
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.naive_bayes import MultinomialNB
from joblib import Parallel, delayed
import pickle
plt.rcParams.update({'font.size': 15})

class PageRank(object):

    def __init__(self, candidate_list = []):
        # set attributs to store important informations
        
        # the matrix used by Page Rank
        self.adjacency_matrix = None
        
        # the cosine similarity matrix
        self.similar_matrix = None
        
        # the bayes model
        self.model = None
        
        # attributs used to swith between item id and location in the matrixes
        self.i_to_item = None
        self.item_to_i = None
        
        # the list of items that can be recommended
        self.candidate_list = candidate_list
        
        # used to plot informations on the results
        self.validate_rank = [] # contain a list of tuple (session_id, rank_of_the_item_that_should_be_recommended)
        
        # the matrix containing the caracteristics of each items (condensed)
        self.features_matrix = None
        
        # used by bayes
        self.candidate_list_i = None
        
    def create_bayes_model(self, item_features, X_train, y_train):
        # get list of items
        list_item = item_features["item_id"].unique()

        # create index to item id and item id to index
        self.i_to_item = list_item.tolist()
        self.item_to_i = {}
        for i, item in enumerate(list_item):
            self.item_to_i[item] = i
            
        # create a condensed matrix containing the item ids in index and item attributs in collumns
        # it is filled with 0 and 1 when the item have a specific attribut
        
        item_feat = item_features.copy()
        item_feat.drop(["feature_value_id"], axis=1, inplace=True)
        item_feat["count"] = 1
        item_feat.set_index(["item_id", "feature_category_id"], inplace = True)
        item_feat = item_feat[~item_feat.index.duplicated(keep='first')].unstack()
        item_feat = item_feat.droplevel(level=0, axis=1)
        item_feat = item_feat.fillna(0)
        item_feat = item_feat.astype("int32")

        # save it
        self.features_matrix = item_feat
        
        train_b = X_train.copy()[["session_id", "item_id"]]
        train_b.set_index(["session_id"], inplace = True)
        train_b = pd.merge(train_b, item_feat, left_on="item_id", right_index=True)
        train_b.drop(["item_id"], axis = 1, inplace = True)
        train_b = train_b.groupby(train_b.index).sum()
        
        train_buy = y_train.copy()[["session_id", "item_id"]]
        train_buy.set_index(["session_id"], inplace = True)
        cc = pd.merge(train_b, train_buy, left_index=True, right_index=True)
        cc.set_index(["item_id"], inplace = True)
        
        # load or train the bayes model on the matrix with the 
        try:
            # load model if already trained
            with open(os.path.join('matrix_saves', 'bayes.pkl'), 'rb') as handle:
                self.model = pickle.load(handle)
        except:
            self.model = MultinomialNB()
            y_unique = cc.index.unique()
            batch = 10000
            for i in range(0, cc.shape[0], batch):
                # partial_fit because the datas can not be fit in our RAM
                self.model.partial_fit(cc.iloc[i:i + batch], cc.iloc[i:i + batch].index, classes=y_unique)
                print(".", end = "")
            
            pickle.dump(self.model, open(os.path.join('matrix_saves', 'bayes.pkl'), 'wb'))
        
        
        self.candidate_list_i = [self.item_to_i[it] for it in self.model.classes_]
        
    def create_similar_matrix(self, item_features):
        # get list of items
        list_item = item_features["item_id"].unique()

        # create index to item id and item id to index
        self.i_to_item = list_item.tolist()
        self.item_to_i = {}
        for i, item in enumerate(list_item):
            self.item_to_i[item] = i
           
        # load or create similar matrix
        try:
            # load matrix if already calculated
            self.similar_matrix = np.load(os.path.join('matrix_saves', 'similar_matrix.npy'))
        except:
            # see https://towardsdatascience.com/collaborative-filtering-based-recommendation-systems-exemplified-ecbffe1c20b1
            
            # get nb of items
            nb_item = list_item.size

            # create similar matrix
            item_feat = item_features.copy()
            item_feat.set_index(["feature_category_id", "feature_value_id"], inplace = True)

            tmp_list = item_feat.index.to_list()
            from collections import Counter
            counts = Counter(tmp_list)
            tmp_list= [id for id in tmp_list if counts[id] < 2]
            item_feat.drop(tmp_list, inplace=True)
            item_feat = item_feat.sort_index()
            item_feat["item_id"] = [self.item_to_i[item] for item in item_feat["item_id"]]

            values_in_common = np.zeros((nb_item, nb_item), dtype=int)

            i= 0
            for index, list_item in item_feat.groupby(item_feat.index)["item_id"].apply(list).to_frame().iterrows():
                if i % 10 == 0:
                    print(".", end = "")
                my_list = list_item["item_id"]
                for k in range(len(my_list)):
                    values_in_common[my_list, np.roll(my_list, k)] += 1
                i+=1
        
            nb_attr = item_features.groupby(["item_id"]).count()
            nb_attr.index = [self.item_to_i[item] for item in nb_attr.index]
            nb_attr = nb_attr.sort_index()
            nb_attr["feature_category_id"] = nb_attr["feature_category_id"].apply(np.sqrt)
            nb_attr["feature_category_id"].values

            self.similar_matrix = (values_in_common / nb_attr["feature_category_id"].values).transpose() / nb_attr["feature_category_id"].values
            
            print("similar_matrix done")
            
            np.save(os.path.join('matrix_saves', 'similar_matrix'), self.similar_matrix)
        
    def create_adjacency_matrix(self, item_features, X_train, y_train, mode="session_to_purchase", g=0.9, leader=False):
        # 2 mode are implemented: "session_to_purchase" and "session_to_session_purchase"
        
        # get list of items
        list_item = item_features["item_id"].unique()
        
        # create index to item id and item id to index
        self.i_to_item = list_item.tolist()
        self.item_to_i = {}
        for i, item in enumerate(list_item):
            self.item_to_i[item] = i   

        # load or create adjacency matrix
        nb_item = list_item.size
        self.adjacency_matrix = np.zeros((nb_item, nb_item), dtype=float)
        
        try:
            # load adjacency matrix
            if mode == "session_to_session_purchase":
                if leader:
                    self.adjacency_matrix = np.load(os.path.join('matrix_saves', 'session_to_session_purchase_leader.npy'))
                else:
                    self.adjacency_matrix = np.load(os.path.join('matrix_saves', 'session_to_session_purchase.npy'))
            elif mode == "session_to_purchase":
                if leader:
                    self.adjacency_matrix = np.load(os.path.join('matrix_saves', 'session_to_purchase_leader.npy'))
                else:
                    self.adjacency_matrix = np.load(os.path.join('matrix_saves', 'session_to_purchase.npy'))
            else:
                raise Exception("mode not valid")
        except:
            if mode == "session_to_session_purchase":
                # Create adjacency matrix with items seen to items seen and bought
                df = pd.concat([X_train, y_train]).set_index("date", append=True).sort_index()
                df["item_next"] = -1
                df["item_next"][:-1] = df["item_id"][1:]
                df.reset_index(level=1, drop=True, inplace=True)
                df["item_next"][:-1][df["item_next"][:-1].index != df["item_next"][1:].index] = -1
                df = df[df.item_next != -1]
                df = df.groupby(["item_id", "item_next"]).size().to_frame()
                df.columns = ["value"]

            elif mode == "session_to_purchase":
                # Create adjacency matrix with items seen to items bought
                Xt = X_train.set_index("session_id")["item_id"].to_frame()
                yt = y_train.set_index("session_id")["item_id"].to_frame()

                tmp_1 = pd.merge(Xt, yt, left_index=True, right_index=True)

                tmp_2 = Xt
                tmp_2["value"] = 1
                tmp_2 = tmp_2["value"]
                tmp_2 = tmp_2.reset_index().groupby(["session_id"]).sum()
                tmp_2["value"] = 1/tmp_2["value"]

                df = pd.merge(tmp_1, tmp_2, left_index=True, right_index=True)
                df = df.reset_index(drop=True).groupby(["item_id_x", "item_id_y"]).sum()

            else:
                raise Exception("mode not valid")
                
            # Create adjacency matrix using previous informations
            i = 0
            for index, row in df.iterrows():
                item_1, item_2 = index
                item_1, item_2 = self.item_to_i[item_1], self.item_to_i[item_2]

                if i % 100000 == 0:
                    print(".", end = "")
                i+=1

                self.adjacency_matrix[item_1, item_2] = row["value"]
                
            # Normalization
            my_sum = self.adjacency_matrix.sum(axis=1).reshape((nb_item,1))
            self.adjacency_matrix[np.where(my_sum == 0)[0], :] = 1/nb_item
            my_sum[my_sum == 0] = 1
            self.adjacency_matrix /= my_sum
            
            # save it
            if mode == "session_to_session_purchase":
                if leader:
                    np.save(os.path.join('matrix_saves', 'session_to_session_purchase_leader'), self.adjacency_matrix)
                else:
                    np.save(os.path.join('matrix_saves', 'session_to_session_purchase'), self.adjacency_matrix)
            elif mode == "session_to_purchase":
                if leader:
                    np.save(os.path.join('matrix_saves', 'session_to_purchase_leader'), self.adjacency_matrix)
                else:
                    np.save(os.path.join('matrix_saves', 'session_to_purchase'), self.adjacency_matrix)
            else:
                raise Exception("mode not valid")
        
        # add the possibility to predict the others items
        self.adjacency_matrix *= g
        self.adjacency_matrix += (1 - g)/nb_item
        print("adj_matrix done")

        # transpose the matrix
        self.adjacency_matrix = self.adjacency_matrix.transpose()
    
    def process_page_rank(self, to_use, n_iter=10, bench=4096, save=False, save_file="res.csv", validate_to_use=None, restart=False, last_seen_better = True, similar_compute=True, multi_last_better = 31/30, leader = False):
        
        '''
        params:
            - to_use : session informations used to predict the recommended items
            - n_iter : number of iterations in the page rank algorithm
            - bench : size of batchs (used to have enougth RAM to cumpute the result)
            - save : indicate if the result need to be saved
            - save_file : name / path of the file that will be created
            - validate_to_use : session recommendation solution (used to cumpute informations on our predictions)
            - restart : indicate if a restart factor of 0.1 should be added
            - last_seen_better : indicate if the items seen in last in the session should be more important to cumpute the result
            - multi_last_better : multiplier used to know the importance of an element compared to the previous one
            - similar_compute : indicate if the cosinus similarity should be used
            - leader : indicate if we are cumputing the result for the leaderboard (small changes to improve results)
        '''
        # score of the resulting recommendations
        score = 0

        # store some shape informations
        n2 = None
        if self.features_matrix is not None:
            n = self.features_matrix.shape[0]
            n2 = self.features_matrix.shape[1]
        elif self.adjacency_matrix is not None:
            n = self.adjacency_matrix.shape[0]
        else:
            n = self.similar_matrix.shape[0]
            
        self.validate_rank = []

        # if the result will be saved, create the file
        if save:
            file = open(save_file,"w") 
            file.write("session_id,item_id,rank\n")
            
        # sort the inputs
        to_use = to_use.set_index(["session_id"]).sort_index()["item_id"].to_frame()
        if validate_to_use is not None:
            validate_to_use = validate_to_use.set_index(["session_id"]).sort_index()["item_id"].to_frame()

        # get list of session ids
        to_use_indexes = to_use.index.unique().tolist()
        
        # create weights array used to take more into account the latest items seen
        weights = [1]
        i = 0
        while weights[-1] < 100:
            weights.append(weights[-1]*multi_last_better)
            i+=1
            if i > 110:
                break
        
        nb_max_item = to_use.groupby("session_id")["item_id"].count().max()
        while len(weights) < nb_max_item:
            weights.append(100)
                
        weights = [[wei] for wei in weights]
        
        
        # iterate on each batch of the input datas
        for part in range(to_use.index.unique().size//bench + 1):
            # get the current batch of sessions
            shift = part * bench
            list_to_use = to_use_indexes[shift: shift + bench]
            sub_to_use = to_use.loc[list_to_use]

            # create matrix containing, for each session (collumn), the items seen weighted using the weights array
            # and divided by the sum of items, so the sum of each collumn = 1
            # the resulting array represent the importance of each item for each session
            array = np.full((n, len(list_to_use)), 1/n)

            i=0
            for index in list_to_use:
                row = sub_to_use.loc[index]
                session = index
                my_arr = np.zeros((n, 1), dtype=float)

                items_seen = [self.item_to_i[x] for x in row.values.flatten()]
                if leader:
                    my_arr[items_seen] = [[i] for i in range(1, len(items_seen) + 1)]
                elif last_seen_better:
                    my_arr[items_seen[-len(weights):]] = weights[:len(items_seen)]
                else:
                    my_arr[items_seen] = 1

                my_arr /= my_arr.sum()
                array[:,i] = my_arr.transpose()
                i+=1
                
            
            # create matrix containing used by the bayes model, it contain, for each session (collumn), the items seen
            array2 = None
            
            if n2 is not None:
                i = 0
                array2 = np.zeros((len(list_to_use), n2))
                for index in list_to_use:
                    row = sub_to_use.loc[index]
                    session = index
                    my_arr = np.zeros(n2, dtype=int)
                    
                    for item_seen in row.values.flatten():
                        my_arr += self.features_matrix.loc[item_seen].tolist()
                        
                    array2[i] = my_arr
                    i+=1
            
            print("array created")


            # save the original array to be used with the restart
            start = array.copy()
            
            # Page Rank case
            if self.adjacency_matrix is not None:
                # Execute the page rank alghoritm
                if restart:
                    for i in range(n_iter):
                        print(".", end = "")
                        array = self.adjacency_matrix @ (array * 0.9 + start * 0.1)
                else:
                    for i in range(n_iter):
                        print(".", end = "")
                        array = self.adjacency_matrix @ array

                # hybrid filtering => add the cosine similarity influence to the page rank result
                if similar_compute and self.similar_matrix is not None:
                    # hybrid filtering is done by multipling the result of each algorithm
                    array = (self.similar_matrix @ start) * array

            # Cosine similarity case
            elif similar_compute and self.similar_matrix is not None:
                array = self.similar_matrix @ start
                
            # Bayes case
            elif self.model is not None:
                
                def split(a, n):
                    k, m = divmod(len(a), n)
                    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
                
                n_cores = 4
                
                to_be_predicted2 = list(split(array2, n_cores))

                # pararallelized
                parallel = Parallel(n_jobs=n_cores)
                results = parallel(delayed(self.model.predict_proba)(to_be_predicted2[i]) for i in range(n_cores))

                array_tmp = np.vstack(results).transpose()
                array = np.zeros((n, len(list_to_use)))
                array[self.candidate_list_i] = array_tmp

            print(str(n_iter) + " iterations done")
            
            # remove item seen
            array[start != 0] = 0
            
            # set score to 0 for items not in the candidate list
            if self.candidate_list != []:
                candidates = [self.item_to_i[candidate] for candidate in self.candidate_list]
                not_candidates = [i for i in range(array.shape[0]) if i not in candidates]
                array[not_candidates] = 0
            
            # get the top 100 of items with best score
            rank = np.argsort(array, axis=0)[-100:][::-1].transpose()

            # iterate on each recommended item
            for i, cur_rank in enumerate(rank):
                if i % 100 == 0:
                    print(".", end = "")
                session = list_to_use[i]
                # if the optimal result is given, cumpute the score
                if validate_to_use is not None:
                    real_res = self.item_to_i[validate_to_use.loc[session]["item_id"]]
                    # item rank is 105 if optimal item not in the top 100
                    item_rank = 105
                    if real_res in cur_rank:
                        item_rank = (np.where(cur_rank == real_res)[0])[0] + 1
                        score += 1/(item_rank)
                    self.validate_rank.append([session, item_rank])
                    
                # add the recommended items to the resulting file
                if save:
                    file.write("".join([str(session) + "," + str(self.i_to_item[item]) + "," + str(r + 1) + "\n" for r, item in enumerate(cur_rank)]))

            print("add session [" + str(shift) + ":" + str(shift + len(list_to_use)) + "]")
        # print the score of this test
        if validate_to_use is not None:
            print("Score: " + str(score/validate_to_use.shape[0]))
        if save:
            file.close()
            
            
    def show_distrib_rank(self):
        # plot the distribution of the rank of the optimal item in our predictions
        ranks = [t[1] for t in self.validate_rank]
        from collections import Counter
        result = sorted(list(Counter(ranks).items()))
        res1 = [t[0] for t in result]
        res2 = [t[1] for t in result]
        plt.figure(figsize=(20,10))
        plt.bar(res1, res2)
        plt.title("Distribution of the rank of the optimal recommendation")
        plt.ylabel("number")
        plt.xlabel("optimal recommendation rank")
        
    def get_bad_sessions(self):
        # return the session ids of sessions where we dont recommended the optimal item
        bad_sessions = [t[0] for t in self.validate_rank if t[1] > 100]
        return bad_sessions
    
    def show_stat_bad_sessions(self, to_use):
        # plot the rank of the optimal item depending on the number of items seen during the session
        # permit to understand the efficiency of our algorithm on session with a lot or few item seen
        
        # number of items seen during each session
        to_use_count = to_use.groupby(["session_id"])["item_id"].count().to_frame()
        to_use_count.columns = ["nb_session_item"]
        
        # rank of the optimal prediction for each session
        df = pd.DataFrame(self.validate_rank)
        df.columns = ["session_id", "predict_rank"]
        df.set_index("session_id", inplace=True)
        
        concat = pd.concat([to_use_count, df], axis=1)
        res = concat.groupby(["nb_session_item", "predict_rank"]).size().to_frame()
        res.columns = ["nb"]
        res.reset_index(inplace=True)
        
        # to have resultat that can be seen (colors)
        res["nb"] = np.log(res["nb"] + 1)
        res["nb"] = np.log(res["nb"] + 1)
        res["nb"] = np.log(res["nb"] + 1)
        res["nb"] -= res["nb"].min()
        res["nb"] /= res["nb"].max()

        plt.figure(figsize=(20,10))
        plt.scatter(y=res["nb_session_item"],x=res["predict_rank"], c=np.array(res.nb.tolist()), cmap="winter")
        plt.title("Distribution of the rank of the optimal recommendation depending on the number of items seen")
        plt.ylabel("number of item seen")
        plt.xlabel("optimal recommendation rank")
        plt.show()

        concat["good_predict"] = concat["predict_rank"] < 100
        concat["nb"] = 1
        concat.drop(["predict_rank"], axis = 1, inplace = True)
        
        res2 = concat.groupby(["nb_session_item"]).sum()
        res2["percent_good"] = res2["good_predict"] / res2["nb"]
        res2.loc[res2["percent_good"] < 0.01, "percent_good"] = 0.01
        plt.figure(figsize=(20,10))
        plt.bar(res2.index, res2["percent_good"])
        plt.title("Percent of prediction in the top 100 depending on the number of items seen")
        plt.ylabel("Percent of prediction in the top 100")
        plt.xlabel("Number of items seen")
        plt.show()
        
        
        plt.figure(figsize=(20,10))
        plt.bar(res2.index, res2["nb"])
        plt.title("Number of session depending on the number of items seen")
        plt.ylabel("Number")
        plt.xlabel("Number of items seen")
        plt.yscale('log')
        plt.show()
        