import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import MultiViewDataset
from model import ProposedModel
from cluster import cluster
from ksvd import ApproximateKSVD
from sklearn.decomposition import MiniBatchDictionaryLearning as mbdl
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train(model, train_data, train_loader, valid_loader, epochs: list, k, save_weights_to, n_clusters, unbalance, device):
    model = model.to(device)
    optimizer1 = torch.optim.Adam([
        {'params': model.S},
        {'params': model.A},
    ], lr=0.01)
    # construct the consensus similarity matrix 
    train_num = len(train_data)
    x_tensor = {}
    print("flag---")
    for i in train_data.x.keys():
        x_tensor[i] = torch.tensor(train_data.x[i])
    for k in x_tensor.keys():
        x_tensor[k] = x_tensor[k].to(device)
    for i in range(model.view_num):
        # view-specific similarity matrices
        print("build similarity matrix "+str(i)+":---")
        model.W[i] = model.similar_matrix(x_tensor[i], k).to(device)
    print("similarity matrix training:------------------------")
    for epoch in range(epochs[0]):
        rec = torch.tensor(0)
        rec = rec.to(device)
        for i in x_tensor.keys():
            rec = rec + model.A[i] * model.W[i]

        loss = torch.dist(model.S, rec, 2)**2 / train_num**2
        optimizer1.zero_grad()
        loss.backward(retain_graph=True)
        optimizer1.step()
        model.A.data = nn.Softmax(dim=0)(model.A.data)
        print('epoch{}---:, loss{}'.format(epoch, loss))
    print("Dictionary training:-------------------------------")

    # forward
    model.S.requires_grad_(False)
    model.A.requires_grad_(False)
    optimizer2 = torch.optim.Adam([
        {'params': (p for n, p in model.named_parameters()
                    if p.requires_grad and 'weight' in n), 'weight_decay': 0.01},
        {'params': (p for n, p in model.named_parameters()
                    if p.requires_grad and 'weight' not in n)},
    ], lr=0.01)
    step_lr = torch.optim.lr_scheduler.StepLR(
        optimizer2, step_size=25, gamma=0.1)
    best_model_wts = model.state_dict()

    for epoch in range(epochs[1]):
        model.train()
        train_loss,  num_samples = 0, 0
        for batch in train_loader:
            x, y, index = batch['x'], batch['y'], batch['index']
            for k in x.keys():
                x[k] = x[k].to(device)
            index = index.to(device)
            batch_s = model.S[index, :]
            batch_s = batch_s[:, index]
            ret = model(x, y, S=batch_s, batch=len(y), unbalance=unbalance)
            optimizer2.zero_grad()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            ret['loss'].backward()
            optimizer2.step()
            train_loss += ret['loss'].mean().item() * len(y)  
            num_samples += len(y)

        train_loss = train_loss / num_samples
        step_lr.step()
        ret = model.fusion_z(x_tensor, unbalance)
        fusion = ret['fusion'].data.cpu().numpy()

        print("*******************************")
        acc, _, nmi, _, ri, _, f1, _ = cluster(
            n_clusters, fusion, train_data.y, count=5)
        print(
            f'Epoch {epoch:3d}: train loss {train_loss:.4f},acc {acc/100:.6f}, nmi {nmi/100:.4f},ri {ri/100:.4f}')


    if save_weights_to is not None:
        os.makedirs(os.path.dirname(save_weights_to), exist_ok=True)
        torch.save(best_model_wts, save_weights_to)
    model.load_state_dict(best_model_wts)
    return model


def validate(model, loader, unbalance, n_clusters, device='cuda'):
    model.eval()
    with torch.no_grad():
        acc, nmi, ri, f1, num_samples = 0, 0, 0, 0, 0
        for batch in loader:
            x = batch['x']
            y = batch['y']
            for k in x.keys():
                x[k] = x[k].to(device)
            y = y.to(device)
            ret = model.fusion_z(x, unbalance)  
            acc_avg, acc_std, nmi_avg, nmi_std, ri_avg, ri_std, f1_avg, f1_std = cluster(
                n_clusters, ret['fusion'], y)
            acc += acc_avg
            nmi += nmi_avg
            ri += ri_avg
            f1 += f1_avg
            num_samples += len(batch['y'])
        print(
            f'acc {acc/num_samples:.4f}, nmi {nmi/num_samples:.4f},ri {ri/num_samples:.4f},f1 {f1/num_samples:.4f}')
    return


def experiment(data_path, com_dim, en_hidden, low_hidden, high_hidden, epochs, k, unbalance,device):
    train_data = MultiViewDataset(data_path=data_path, train=True)
    valid_data = MultiViewDataset(data_path=data_path, train=False)
    if unbalance == True:
        low_index = 1
        high_index = [0]
        train_num = len(train_data)
        print("dictionary learning------------")
        #aksvd = mbdl(n_components=low_hidden, alpha=1, n_iter=20)
        aksvd = ApproximateKSVD(n_components=low_hidden, max_iter=30, tol=1e-6,
        transform_n_nonzero_coefs=4)
        dictionary = aksvd.fit(train_data.x[low_index]).components_
        gamma = aksvd.transform(train_data.x[low_index])
        train_data.x[low_index] = gamma.astype(np.float32)
        print(gamma.shape)

    else:
        high_index=[]
        train_num = len(train_data)


    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=128)
    umdl = ProposedModel(sample_shape=[s.shape for s in train_data[0]['x'].values()],
                         high_index=high_index,
                         train_num=train_num,
                         class_num=len(set(train_data.y)),
                         com_dim=com_dim,
                         en_hidden=en_hidden,  
                         high_hidden=high_hidden,
                         lambda2=0.1 # Loss bn by L1-norm
                         )
    print('---------------------------- Experiment ------------------------------')
    print('Dataset:', data_path)
    print('Number of views:', len(train_data.x), ' views with dims:',
          [v.shape[-1] for v in train_data.x.values()])
    print('Number of classes:', len(set(train_data.y)))
    print('Number of training samples:', len(train_data))
    print('Number of validating samples:', len(valid_data))
    print('Trainable Parameters:')

    for n, p in umdl.named_parameters():
        print('%-40s' % n, '\t', p.data.shape)
    print('----------------------------------------------------------------------')

    # ****load model****
    '''
    umdl.load_state_dict(torch.load('CODE/gragh_basline/toy_net_params.pth',map_location='cpu'))
    for n, p in umdl.named_parameters():
        print('%-40s' % n, '\t', p,'\n')
    '''
    # **********
    umdl = train(umdl, train_data, train_loader, valid_data, epochs,
                 k=k, save_weights_to=None, n_clusters=len(set(train_data.y)), unbalance=unbalance, device=device)
    print(umdl.A)



if __name__ == '__main__':

    experiment(data_path="/dataset/MSRCV1_6views.mat", com_dim=300, en_hidden=500,
               low_hidden=100, high_hidden=600, epochs=[50, 30], k=30, unbalance=False,device='cpu')
   