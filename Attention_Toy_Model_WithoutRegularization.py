
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

torch.manual_seed(1234)
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
        
        self.w2v = nn.Parameter(torch.FloatTensor(np.random.uniform(-np.sqrt(3/v_dim), np.sqrt(3/v_dim), (vocab_size, v_dim)), device = device))
        self.t2v = nn.Parameter(torch.FloatTensor(np.random.uniform(-np.sqrt(3/k_dim), np.sqrt(3/k_dim), (topic_size, k_dim)), device = device))
        self.B = nn.Parameter(torch.randn(v_dim, k_dim, dtype=self.dtype, device = device))
        self.A = nn.Parameter(torch.randn(v_dim, k_dim, dtype=self.dtype, device = device))
        
    def forward(self, doc):
        sigma = torch.rand(1, dtype=self.dtype, device = device)
        alpha = torch.FloatTensor().new_full((self.topic_size, self.vocab_size), 0, dtype=self.dtype, device = device)
        mu = torch.FloatTensor().new_full((self.v_dim, self.vocab_size), 0, dtype=self.dtype, device = device)
        s = torch.FloatTensor().new_full((self.topic_size, self.vocab_size), 0, dtype=self.dtype, device = device)
        P = torch.FloatTensor().new_full((1, self.vocab_size), 0, dtype=self.dtype, device = device)
#        RL = torch.FloatTensor().new_full((1, doc.shape[0]), 0, dtype=self.dtype, device = device)
#        a_mean = torch.FloatTensor().new_full((self.topic_size, doc.shape[0]), 0, dtype=self.dtype, device = device)
        
        s = torch.mm(torch.mm(self.w2v, self.B), self.t2v.t()).t()

        alpha = self.softmax(s.clone())
        
        mu = torch.mm(self.A,(alpha.t().view(self.vocab_size, self.topic_size,-1)* self.t2v).sum(1).t())

        P = 1/sigma * torch.mm((self.w2v-mu.t()), (self.w2v.t()-mu)).diag()     
        
#        a_mean = (1/doc.sum(1)).unsqueeze(1)*(doc.view(-1,1,V)*alpha).sum(2)
        
#        RL = (doc*(((alpha.t()-a_mean.view(-1, 1, self.topic_size)))**2).sum(2)).sum(1)
            
        return alpha, P,  s, sigma, self.w2v.clone(), self.t2v.clone()
    
    def MLE(self, doc, lamda, P):

        return ((doc * P).sum())/doc.shape[0]

def main():
    lamda = 100
    itera = 1000
    cost = []
    long = []
    
    model = Toy(V, v_d, K,  k_d).to(device)
#    alpha1, P1, RL1, s1, sigma1, w2v1, t2v1 = model()
#    for param in model.parameters():
#        print(param)
    data.dtype = 'int32'
    n = torch.FloatTensor(data)
    import torch.utils.data as Data
    BATCH_SIZE = 100
    torch_dataset = Data.TensorDataset(n)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    learning_rate = .01
#    optimizer = optim.SGD(model.parameters(), lr=learning_rate ) 
#    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=.8)
    for epoch in range(itera):
        print('------current epoch:{}------'.format(epoch))
        losses = []
#        scheduler.step()
        for step, (batch_x,) in enumerate(loader, 0):
            scheduler.step()
            optimizer.zero_grad()
            alpha, P, s, sigma, w2v, t2v = model(batch_x.to(device))
            loss = model.MLE(batch_x.to(device), lamda, P)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            losses.append(loss.item())
            if long == [] or loss.item() < min(long):
                result = s.clone()
                torch.save(model.state_dict(), 'save_model_paramsVD{}KD{}LR{}IT{}BS{}.pth'.format(v_d, k_d, learning_rate, itera, BATCH_SIZE))
            print(loss.item())
            long.append(loss.item())
        cost.append(sum(losses)/len(losses))     
        print('******the cost of epoch {} is {}******'.format(epoch, cost[epoch]))
        
    plt.figure()
    plt.plot(cost)
    plt.show()       
    print('the total time is :', time.time() - t)
    return result, sigma, cost, long, w2v, t2v, alpha, s#, s1, RL, RL1

if __name__ == '__main__':
    __spec__ = None 
    score, sigma, cost, long, word, topic, a, s = main()
    n_top_words = 5
    for i, topic_dist in enumerate(score.cpu().detach().numpy()):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
