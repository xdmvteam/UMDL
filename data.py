import pickle
import scipy.io
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from sklearn.decomposition import PCA


def load_toy_example(train=True):
    x, y = pickle.load(open('dataset/toy_example/toy_dim100x12.pkl', 'rb'))
    num_train = int(len(y) * 0.8)
    if train:
        for v, k in enumerate(x.keys()):
            x[v] = x[k][:num_train]
        y = y[:num_train]
    else:
        for v, k in enumerate(x.keys()):
            x[v] = x[k][num_train:]
        y = y[num_train:]
    for k in x.keys():
        x[k] = MinMaxScaler([0, 1]).fit_transform(x[k]).astype(np.float32)
    return x, y.astype(np.int64)


class MultiViewDataset(Dataset):
    def __init__(self, data_path, train=True, custom_views=None):
        super().__init__()
        if data_path == 'toy-example':
            self.x, self.y = load_toy_example(train)
            return

        dataset = scipy.io.loadmat(data_path)
        mode = 'train' if train else 'test'
        num_views = 1
        
        self.x = dict()
        view = dataset[f'x{6}']
        self.x[0] = MinMaxScaler([0, 1]).fit_transform(view).astype(np.float32)
        '''for k in range(num_views):
            view = dataset[f'x{k+1}']
            self.x[k] = MinMaxScaler([0, 1]).fit_transform(view).astype(np.float32)'''
        self.y = dataset[f'gt'].flatten().astype(np.int64)

        '''pca = PCA(n_components=20)
        self.x[0] = pca.fit_transform(self.x[0])'''


        if min(self.y) > 0:
            self.y -= 1
        
        if custom_views is not None:
            self.x={k:self.x[k] for k in custom_views}

    def __getitem__(self, index):
        x = dict()
        for v in self.x.keys():
            x[v] = self.x[v][index]
        return {
            'x': x,
            'y': self.y[index],
            'index': index
        }

    def __len__(self):
        return len(self.y)
