import torch
from torch.utils.data import Dataset, DataLoader
import os
from numpy.random import choice as npc
import numpy as np
import time
import random
import pandas as pd


class Train(Dataset):

    def __init__(self, dataPath, trainID, num_sample, epoch):
        super(Train, self).__init__()
        np.random.seed(10)
        self.num_sample = num_sample
        self.trainID = trainID
        self.epoch = epoch
        self.datas_train, self.datas_test, self.num_classes = self.loadToMem(dataPath, trainID, num_sample)

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

        datas_train = {}
        datas_test = {}
        for m in range (0, len(all_labels)):
            for n in range(0, len(trainID)):
                datas_train[all_labels[m]+'_'+trainID[n]] = []
                datas_test[all_labels[m]+'_'+trainID[n]] = []

        
        idx = len(all_labels)
        for i in range(0, len(all_labels)):
            for j in range(0, len(feature)):
                if (labels[str(lab[j])] == all_labels[i]):
                    # since we repeat 10 times for every experiment, we also introduced some randomness here using ran_tmp
                    ran_tmp = random.randint(0, 3)
                    if ((len(datas_train.get(all_labels[i]+'_'+str(lab[j]).split('_')[0])) < num_sample) and str(lab[j]).split('_')[1][0] == 'i' and ran_tmp < 3):
                        datas_train[all_labels[i]+'_'+str(lab[j]).split('_')[0]].append(feature[j])
                        
                    elif (str(lab[j]).split('_')[1][0] == 'i'):
                        datas_test[all_labels[i]+'_'+str(lab[j]).split('_')[0]].append(feature[j])

        # sys.exit(0)
        return datas_train, datas_test, idx

    def __len__(self):
        return  30*self.num_sample*self.epoch

    def __getitem__(self, index):

        all_labels = ['hap','ang','sad']
        ran_1 = random.randint(0, 2)
        ran_2 = random.randint(0, 9)
        data = random.choice(self.datas_train[all_labels[ran_1]+'_'+self.trainID[ran_2]])
        return data, torch.from_numpy(np.array([ran_1], dtype=np.int))


class Test(Dataset):

    def __init__(self, num_classes, datas_train, datas_test, trainID, num_sample):
        np.random.seed(10)
        super(Test, self).__init__()

        self.datas_test = datas_test
        self.num_classes = num_classes
        self.datas_train = datas_train
        self.trainID = trainID
        self.num_sample = num_sample
        # = self.loadToMem(dataPath, trainID)
        # self.way = len(self.datas_train[0])+len(self.datas_train[1])+len(self.datas_train[2])+len(self.datas_train[3])
        # self.times = len(self.datas_test[0])+len(self.datas_test[1])+len(self.datas_test[2])+len(self.datas_test[3])


    def __len__(self):
        all_labels = ['hap','ang','sad']
        tmp = 0
        for i in range(0, 3):
            for j in range(0, 10):
                tmp = tmp+ len(self.datas_test[all_labels[i]+'_'+self.trainID[j]])
        return tmp

    def __getitem__(self, index):

        all_labels = ['hap','ang','sad']
        tmp = 0
        for i in range(0, 3):
            for j in range(0, 10):
                for p in range(0, len(self.datas_test[all_labels[i]+'_'+self.trainID[j]])):
                    if (tmp == index):
                        return self.datas_test[all_labels[i]+'_'+self.trainID[j]][p], i
                    tmp = tmp+1
        return 0

















