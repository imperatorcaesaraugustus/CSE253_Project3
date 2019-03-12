import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch as torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import os
        

class Smote(object):
    def __init__(self, samples, N, k = 5, r = 2):
        self.samples = [(samples[i].reshape(1, 224*224)).cpu().tolist() for i in range(len(samples))]
        self.samples = np.array([np.squeeze(self.samples[i]) for i in range(len(samples))])
        self.T, self.attrs_cnt = len(self.samples), len(self.samples[0])
        self.N = N
        self.k = k
        self.r = r  # dimension
        self.neuindex = 0
        
    def generate_points(self):
        if(self.N < 100):
            np.random.shuffle(self.samples)
            self.T = int(self.N*self.T/100)   # pick up N% samples from self.samples
            print(self.T, self.samples.shape)
            self.samples = self.samples[0:self.T]
            self.N = 100
            
        if(self.T < self.k):
            self.k = self.T - 1
        
        N = int(self.N/100)
        self.syn = np.zeros((self.T*N, 224, 224))    
        neighbors = NearestNeighbors(n_neighbors = self.k, algorithm = 'ball_tree', p = self.r).fit(self.samples)
        for i in range(len(self.samples)):
            vec = neighbors.kneighbors(self.samples[i].reshape((1,-1)), return_distance = False)[0] #K-neighbors
            self.faire(N, i, vec)
        return self.syn
    
    def faire(self, N, t, vec):
        for i in range(N):
            attrs = []
            rn = random.randint(0, self.k - 1)
            for attr in range(self.attrs_cnt):
                diff = self.samples[vec[rn]][attr] - self.samples[t][attr]
                delta = random.uniform(0, 1)
                attrs.append(self.samples[t][attr] + delta*diff)
            self.syn[self.neuindex]  = [attrs[i*224: i*224 + 224] for i in range(0, 224)]
            self.neuindex += 1

def pick_rare_classes_samples(train_loader, thres_judge, computing_device):
    print("Picking rare class samples")
    sample_list = [[] for i in range(14)]
    max_sample = [(55000 - 275000*thres_judge[i])/4 for i in range(14)]   # positive ratio of each class
    ans = []
    for batch_count, (images, labels) in enumerate(train_loader, 0):
        if batch_count % 500 == 1: print(batch_count, end = ',')
        if images.size()[0] != 16: break
        if batch_count > 510: break
        for i in range(16):
            for j in range(14):
                if labels[i][j] > 0.5 and len(sample_list[j]) < 20000*thres_judge[j]:
                    sample_list[j].append(images[i])
    for i in range(14):
        with torch.no_grad():
            smote = Smote(sample_list[i], N = 100*max_sample[i]/len(sample_list[i]))
            points = torch.Tensor(smote.generate_points())
            points = (points.unsqueeze(1))
            ans.append((points, i))
    print("Finishing generating rare class samples")
    return ans.to(computing_device)
