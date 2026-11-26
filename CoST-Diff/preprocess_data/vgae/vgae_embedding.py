# original code
import torch
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import VGAE
import numpy as np
import os
import random
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化
    
    
set_seed(42)  # 你可以改成其他数值

def decode(model,epochs,out_channels,data,device):
    num_features = data.num_node_features
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    loss = 0
    for epoch in tqdm(range(1, epochs + 1)): # train
        all_loss = []
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        recon_loss = model.recon_loss(z, edge_index) # how to calculate the loss
        loss = recon_loss +20* (1 / data.num_nodes) * model.kl_loss()  # new line
        #print("all loss is ",float(loss))
        all_loss.append(loss)
        loss.backward()
        if(epoch%10==0):
            print("epoch is {} and loss is {}".format(epoch,loss))
        optimizer.step()
    model.eval()
    with torch.no_grad(): # pred
        z= model.encode(x, edge_index)
        print("z",z)
        print("z.shape",z.shape) # torch.Size([42372, 128])
    return z

class VGAE_gat(torch.nn.Module):
    def __init__(self, in_channels, out_channels,heads):
        super(VGAE_gat, self).__init__()      
        self.conv1 = GATConv(in_channels, out_channels,heads[0]) # cached only for transductive learning
        self.conv2 = GATConv(heads[0]*out_channels, out_channels,heads[1]) # cached only for transductive learning
        self.conv_mu = GATConv(heads[1]* out_channels, out_channels,heads[2],concat = False)
        self.conv_logstd = GATConv(heads[1]* out_channels, out_channels, heads[2],concat =False)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


"""
from torch_geometric.data import Data
import numpy as np

lac_features = np.load('./lac_feature.npy')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

out_dims = 64
data = Data(x= torch.tensor(lac_features,dtype=torch.float),edge_index=edges) # 修改其中的features

num_features = data.num_node_features

model = VGAE(VGAE_gat(num_features, out_dims,[2,2,1]))  # new line 修改其中的num_features
model = model.to(device)
z_gat= decode(model,100,out_dims,data,device) # 修改其中的data
"""
