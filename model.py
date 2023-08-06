from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F



class ParallelModule(nn.Module):
    def __init__(self, modules: Tuple[nn.Module]):
        super().__init__()
        self.parallel_modules = nn.ModuleList(modules)

    def forward(self, x: dict):
        y = dict()
        for k, v in x.items():
            y[k] = self.parallel_modules[k](v)  
        return y


class ProposedModel(nn.Module):
    def __init__(self, sample_shape: list, high_index: list, train_num, class_num,
                 com_dim, en_hidden, high_hidden, lambda2):
        super().__init__()
        self.sample_shape = sample_shape  
        self.view_num = len(sample_shape)
        self.com_dim = com_dim
        self.high_index = high_index  
        self.high_sample_shape = [sample_shape[i]
                                  for i in high_index]  
        self.new_samp_dim = sample_shape
        for i in range(self.view_num):
            if high_index.count(i) > 0:
                self.new_samp_dim[i] = high_hidden
            else:
                self.new_samp_dim[i] = sample_shape[i][-1]

        self.lambda2 = lambda2
        self.margin = torch.tensor(0.5)

        # S:Consensus matrix and W: each views' similarity matrix
        self.W = dict()
        self.S = nn.Parameter(torch.rand(train_num, train_num))
        self.A = nn.Parameter(torch.rand(self.view_num))
        # reduce dimension and add dimension networks

        self.rd_nets = ParallelModule([nn.Sequential(
            nn.Linear(shape[-1], high_hidden),
            nn.ReLU()
        ) for shape in self.high_sample_shape])

        self.encoder = ParallelModule([nn.Sequential(
            nn.Linear(dim, en_hidden),
            nn.BatchNorm1d(en_hidden),
            nn.ReLU(),
            nn.Linear(en_hidden, com_dim),
            nn.BatchNorm1d(com_dim),
            nn.ReLU()
        ) for dim in self.new_samp_dim])  

    def similar_matrix(self, X, k):

        # X(n,d)
        num = X.shape[0]
        sqr_x = torch.sum(X**2, dim=1, keepdim=True)
        dist = sqr_x + sqr_x.t() - 2*(X@X.t())
        knn = dict()
        sortx, index = torch.sort(dist, dim=1, descending=False)
        for i in range(num):
            knn[i] = index[i][1:k+1]
        W = torch.eye(num, num)
        sigma = 0.0
        for i in range(num):
            sigma += sortx[i][k+1]
        sigma = sigma / num
        sigma = 2 * sigma.pow(2)
        for i in range(num):
            for j in range(num):  
                if (knn[i].tolist().count(j) > 0 or knn[j].tolist().count(i) > 0):
                    if (W[i][j] <= 0):
                        W[i][j] = torch.exp(-dist[i][j]/sigma)
        return W

    def forward(self, x: dict, y, S, batch, unbalance):
        # reduce dimension  network
        view = dict()
        if unbalance == True:         
            #msr
            re_x = {}
            re_x[0] = x[0] 
            re_view = self.rd_nets(re_x)  # reduce dimension
            view[0] = re_view[0]
            view[1] = x[1]
            view[2] = x[2]
            view[3] = x[3]
            view[4] = x[4]
            view[5] = x[5]

        else:
            view = x
        # compute zn_v
        zn_v = self.encoder(view)
        fusion = sum(zn_v.values())/self.view_num
        ret = {
            'fusion': fusion
        }

        gama_l1norm = 0
        for k in range(self.view_num):  # for each view
            last_bn = None
            # nets of current view
            for net in self.encoder.parallel_modules[k]:
                if (isinstance(net, nn.BatchNorm1d)):
                    last_bn = net  # Find the last bn layer
            gama_l1norm += last_bn.weight.abs().mean()

        sqr_fusion = torch.sum(fusion**2, dim=1, keepdim=True)
        dist_sqr = sqr_fusion + sqr_fusion.t() - 2*(fusion @ fusion.t())
        dist_sqr = dist_sqr / (batch**2)
 
        A = torch.zeros((batch, batch),dtype=torch.bool)
        for i in range(batch):
            for j in range(batch): 
                if (S[i][j] > 0.5 and A[i][j] == False):
                    A[i][j], A[j][i] = True, True

        cons_loss =torch.tensor(0, dtype=torch.float32)
        zero =torch.tensor(0, dtype=torch.float)
                
        for i in range(batch):                                     
            for j in range(batch):
                if(i!=j):
                    if(A[i][j]==True):
                        cons_loss += dist_sqr[i][j]
                    else:
                        cons_loss += torch.max(self.margin-torch.sqrt(dist_sqr[i][j]), zero)**2
           
        cons_loss = cons_loss / (2*batch)
        ret['loss'] = cons_loss + self.lambda2 * gama_l1norm
        return ret

    def fusion_z(self, x, unbalance):
        view = dict()
        if unbalance == True:
            re_x = {}
            re_x[0] = x[0] 
            re_view = self.rd_nets(re_x)  # reduce dimension
            view[0] = re_view[0]
            view[1] = x[1]
            view[2] = x[2]
            view[3] = x[3]
            view[4] = x[4]
            view[5] = x[5]

        else:
            view = x
        # compute zn_v
        zn_v = self.encoder(view)
        fusion = sum(zn_v.values())/self.view_num
        ret = {
            'fusion': fusion
        }

        return ret
