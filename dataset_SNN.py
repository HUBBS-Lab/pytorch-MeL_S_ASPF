import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import pandas as pd
import sys


class Train_Siamese(Dataset):

    def __init__(self, dataPath, trainID, num_sample, epoch):
        super(Train_Siamese, self).__init__()
        np.random.seed(10)
        self.num_sample = num_sample
        self.trainID = trainID
        self.epoch = epoch
        self.datas_train, self.datas_test, self.num_classes, self.prob_dic = self.loadToMem(dataPath, trainID, num_sample)

    def loadToMem(self, dataPath,trainID, num_sample):
        labels = {}
        with open('feature/iemocap_labels.txt') as file:
            for l in file:
                if (l != '\n'):
                    labels[l.split(' ')[0]] = l.strip('\n').split(' ')[1]


        feature = pd.read_csv(dataPath)
        lab = np.array(feature['id'])
        feature = feature.drop(['id'], axis=1)
        feature = np.array(feature.values)
        datas_train = {}
        datas_test = {}
        all_labels = ['hap','ang','sad']
        prob_dic = {}

        for m in range (0, len(all_labels)):
            for n in range(0, len(trainID)):
                datas_train[all_labels[m]+'_'+trainID[n]] = []
                datas_test[all_labels[m]+'_'+trainID[n]] = []
                prob_dic[all_labels[m]+'_'+trainID[n]] = []

        
        idx = len(all_labels)
        for i in range(0, len(all_labels)):
            for j in range(0, len(feature)):
                if (labels[str(lab[j])] == all_labels[i]):

                    if (str(lab[j]).split('_')[1][0] == 'i'):
                        if ((len(datas_train.get(all_labels[i]+'_'+str(lab[j]).split('_')[0])) < num_sample)):
                            
                            # The initial purpose of the if blocks below is to left one or two samples for other Siamese method test set. The similar code is also used, but may not necessary. 
                            if (all_labels[i] == 'hap' and str(lab[j]).split('_')[0] == 'Ses04F' and (len(datas_train['hap_Ses04F']) > 3)):
                                datas_test[all_labels[i]+'_'+str(lab[j]).split('_')[0]].append(feature[j])
                            elif (all_labels[i] == 'ang' and str(lab[j]).split('_')[0] == 'Ses05M' and (len(datas_train['ang_Ses05M']) > 5)):
                                datas_test[all_labels[i]+'_'+str(lab[j]).split('_')[0]].append(feature[j])
                            elif (all_labels[i] == 'hap' and str(lab[j]).split('_')[0] == 'Ses01M' and (len(datas_train['hap_Ses01M']) > 7)):
                                datas_test[all_labels[i]+'_'+str(lab[j]).split('_')[0]].append(feature[j])
                            elif (all_labels[i] == 'ang' and str(lab[j]).split('_')[0] == 'Ses02F' and (len(datas_train['ang_Ses02F']) > 7)):
                                datas_test[all_labels[i]+'_'+str(lab[j]).split('_')[0]].append(feature[j])
                            else:
                                datas_train[all_labels[i]+'_'+str(lab[j]).split('_')[0]].append(feature[j])
                                prob_dic[all_labels[i]+'_'+str(lab[j]).split('_')[0]].append(1)
                        else:
                            datas_test[all_labels[i]+'_'+str(lab[j]).split('_')[0]].append(feature[j])

        # check if any speaker/emotion for the training set is empty
        for m in range (0, len(all_labels)):
            for n in range(0, len(trainID)):
                if (len(datas_train[all_labels[m]+'_'+trainID[n]]) == 0 or len(datas_test[all_labels[m]+'_'+trainID[n]]) == 0):
                    print(len(datas_train[all_labels[m]+'_'+trainID[n]]))
                    print(len(datas_test[all_labels[m]+'_'+trainID[n]]))
                    print(all_labels[m], trainID[n])
                    exit(0)


        return datas_train, datas_test, idx, prob_dic

    def __len__(self):
        # print(30*num_sample)
        return  self.epoch

    def __getitem__(self, index):
        label = None
        data1 = None
        data2 = None
        all_labels = ['hap','ang','sad']
        ran_1 = random.randint(0, 2)
        ran_2 = random.randint(0, 9)

        # get samples from same class
        if index % 2 == 1:
            label = 1.0
            index_list = list(range(len(self.datas_train[all_labels[ran_1]+'_'+self.trainID[ran_2]])))
            index1 = random.choices(index_list, weights = self.prob_dic[all_labels[ran_1]+'_'+self.trainID[ran_2]])[0]
            index2 = random.choices(index_list, weights = self.prob_dic[all_labels[ran_1]+'_'+self.trainID[ran_2]])[0]
            data1 = self.datas_train[all_labels[ran_1]+'_'+self.trainID[ran_2]][index1]
            data2 = self.datas_train[all_labels[ran_1]+'_'+self.trainID[ran_2]][index2]

            prob_return = [ran_1, ran_2, index1, ran_1, ran_2, index2]
            
        # get samples from different class
        else:
            label = 0.0
            idx1 = random.randint(0, 2)
            while idx1 == ran_1:
                idx1 = random.randint(0, 2)

            # print(all_labels[ran_1], self.trainID[ran_2])
            index_list_1 = list(range(len(self.datas_train[all_labels[ran_1]+'_'+self.trainID[ran_2]])))
            index_list_2 = list(range(len(self.datas_train[all_labels[idx1]+'_'+self.trainID[ran_2]])))
            index1 = random.choices(index_list_1, weights = self.prob_dic[all_labels[ran_1]+'_'+self.trainID[ran_2]])[0]
            index2 = random.choices(index_list_2, weights = self.prob_dic[all_labels[idx1]+'_'+self.trainID[ran_2]])[0]
            data1 = self.datas_train[all_labels[ran_1]+'_'+self.trainID[ran_2]][index1]
            data2 = self.datas_train[all_labels[idx1]+'_'+self.trainID[ran_2]][index2]

            prob_return = [ran_1, ran_2, index1, idx1, ran_2, index2]

        return data1, data2, torch.from_numpy(np.array([label], dtype=np.float32)), torch.from_numpy(np.array(prob_return))

    def update_prob(self, prob_list, prob_increase):
        all_labels = ['hap','ang','sad']
        # prob_list = [[1, 3, 0, 2, 3, 0]]

        for i in range(0, len(prob_list)):
            # print(all_labels[prob_list[i][0]]+'_'+self.trainID[prob_list[i][1]])
            self.prob_dic[all_labels[prob_list[i][0]]+'_'+self.trainID[prob_list[i][1]]][prob_list[i][2]] += prob_increase[i]
            self.prob_dic[all_labels[prob_list[i][3]]+'_'+self.trainID[prob_list[i][4]]][prob_list[i][5]] += prob_increase[i]












