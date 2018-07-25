# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 19:38:06 2018

@author: wangz
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:01:52 2018

@author: wangz
"""
import numpy as np
import lda
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import time

torch.manual_seed(1000)
t = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()
docum = lda.datasets.load_reuters_titles()

V = len(vocab)
D = len(docum)
word_to_id = {}
K = 20
v_d = 50
k_d =10    

for i in range(V):
    word_to_id[vocab[i]] = i
    

class Toy(nn.Module):
    def __init__(self, vocab_size, v_dim, topic_size, k_dim):
        super(Toy, self).__init__()
        self.dtype = torch.float
        self.vocab_size = vocab_size
        self.topic_size = topic_size
        self.v_dim = v_dim
        self.k_dim = k_dim
        self.softmax =  nn.Softmax(dim=0)
        
        self.w2v = nn.Parameter(torch.FloatTensor(np.random.uniform(-np.sqrt(3/v_dim), np.sqrt(3/v_dim), (vocab_size, v_dim)), device=device))
        self.t2v = nn.Parameter(torch.FloatTensor(np.random.uniform(-np.sqrt(3/k_dim), np.sqrt(3/k_dim), (topic_size, k_dim)), device=device))
        self.B = nn.Parameter(torch.randn(v_dim, k_dim, dtype=self.dtype, device=device))
        self.A = nn.Parameter(torch.randn(v_dim, k_dim, dtype=self.dtype, device=device))
        
    def forward(self):
        sigma = torch.rand(1, dtype=self.dtype, device=device)
        alpha = torch.FloatTensor().new_full((self.topic_size, self.vocab_size), 0, dtype=self.dtype, device=device)
        mu = torch.FloatTensor().new_full((self.v_dim, self.vocab_size), 0, dtype=self.dtype, device=device)
        s = torch.FloatTensor().new_full((self.topic_size, self.vocab_size), 0, dtype=self.dtype, device=device)
        P = torch.FloatTensor().new_full((1,self.vocab_size), 0, dtype=self.dtype, device=device)
        RL = torch.Tensor([0], device=device)
        
        for i in range(self.vocab_size):
            s[:,i] = torch.mm(torch.mm(self.w2v[i].unsqueeze(0), self.B), self.t2v.t()) 

            alpha[:,i] = self.softmax(s[:,i].clone())

            mu[:,i] = torch.mm(self.A,(alpha[:,i].clone().unsqueeze(1).repeat(1, self.k_dim)* self.t2v).sum(0).unsqueeze(1)).squeeze(1)

            P[:,i] = 1/sigma * torch.mm((self.w2v[i]-mu[:,i]).unsqueeze(0), (self.w2v[i]-mu[:,i]).unsqueeze(1))     

        RL = ((alpha - torch.mean(alpha, 1).unsqueeze(1).repeat(1, self.vocab_size))**2).sum()
            
        return alpha, P, RL, s, sigma
    
    def MLE(self, doc, lamda, P, RL):

        return (doc * P).sum()/doc.shape[0] + lamda * RL

def main():
    lamda = 0
    itera = 500
    cost = []
    long = []
    
    model = Toy(V, v_d, K,  k_d).to(device)
    data.dtype = 'int32'
    n = torch.FloatTensor(data)
    import torch.utils.data as Data
    BATCH_SIZE = 64
    torch_dataset = Data.TensorDataset(n)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    learning_rate = .01
#    optimizer = optim.SGD(model.parameters(), lr=learning_rate ) 
#    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=.95)
    for epoch in range(itera):
        print('------current epoch:{}------'.format(epoch))
        losses = []
        scheduler.step()    
        for step, (batch_x,) in enumerate(loader, 0):
            optimizer.zero_grad()
            alpha, P, RL, s, sigma = model()
            loss = model.MLE(batch_x.to(device), lamda, P, RL)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
            if long == [] or loss.item() < min(long):
                result = s
                torch.save(model.state_dict(), 'save_model_paramsVD{}KD{}LR{}IT{}.pth'.format(v_d, k_d, learning_rate, itera))
            print(loss.item())
            long.append(loss.item())
        cost.append(sum(losses)/len(losses))     
        print('******the cost of epoch {} is {}******'.format(epoch, cost[epoch]))
        
    plt.figure()
    plt.plot(cost)
    plt.show()       
    print('the total time is :', time.time() - t)
    return result, sigma, cost, long

if __name__ == '__main__':
    __spec__ = None 
    score, sigma, cost, long = main()
