import os
import argparse
from autoencoder import VGAE, GAE, EdgeDecoder, DirectedEdgeDecoder, InnerProductDecoder, DirectedInnerProductDecoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import copy
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import train_test_split_edges
from scipy.stats import ranksums
from torch_geometric.nn import GraphConv, SAGEConv, GATConv, GCNConv
import torch.nn.functional as F
from torch_geometric.transforms import LineGraph, RandomNodeSplit, RandomLinkSplit
from torch_geometric import seed_everything
import geopandas as gpd

global nodefile, nodes, nodes_df, node, node_rename, flowfile, colname, flow, scaler
global df, df1, df2, edge_name_to_y, edge_name_to_x, G, airport, G_line_graph, airport_line_graph, data
global  edge_array, edge_index_train,  edge_index_val, edge_index_calib_test, edge_weight_train, edge_weight_val, edge_weight_calib_test,  edge_index_train, edge_index_val, edge_index_calib_test
global edge_tensor, edge_weight_gae_training   
seed_everything(42)
np.random.seed(0)
A = np.random.seed(1)

def Dataset_An():
    global nodefile, nodes, nodes_df, node, node_rename, flowfile, colname, flow, scaler
    global df, df1, df2, edge_name_to_y, edge_name_to_x, G, airport, G_line_graph, airport_line_graph, data
    global  edge_array, edge_index_train,  edge_index_val, edge_index_calib_test, edge_weight_train, edge_weight_val, edge_weight_calib_test,  edge_index_train, edge_index_val, edge_index_calib_test
    global edge_tensor, edge_weight_gae_training  
    nodefile = os.path.join(os.path.dirname(__file__), '..', 'data', 'anaheim_nodes.geojson')
    nodes = gpd.read_file(nodefile)
    nodes_df = pd.DataFrame(nodes)
    nodes_df[['X', 'Y']] = nodes_df['geometry'].astype(str).str.split('(').str[-1].str.split(')').str[0].str.split(' ', expand=True).astype(np.float32)
    node = nodes_df.rename(columns={'id': 'node'})

    node_rename = {node: id for id, node in enumerate(range(1, 417))}
    node['node'] = node['node'].map(node_rename)

    flowfile = os.path.join(os.path.dirname(__file__), '..', 'data', 'Anaheim_flow.tntp')
    colname = 'Volume '
    flow = pd.read_csv(flowfile, sep='\t', usecols=['From ', 'To ', colname])

    flow['From '] = flow['From '].map(node_rename)
    flow['To '] = flow['To '].map(node_rename)
    flow = flow[(flow['From '].notna()) & (flow['To '].notna())]
    flow.drop(flow[flow[colname] <= 0].index, inplace=True)
    flow[colname] = np.log(flow[colname])

    scaler = StandardScaler()
    node[['X', 'Y']] = scaler.fit_transform(node[['X', 'Y']].values)
    #minmax = MinMaxScaler()
    #flow[[colname]] = minmax.fit_transform(flow[[colname]].values)

    df = flow.rename(columns={'From ': 's', 'To ': 'r', colname: 'w'})
    df1 = pd.merge(df, node, how='left', left_on='s', right_on='node')[['s', 'r', 'w', 'X', 'Y']].rename(columns={'X': 'X1', 'Y': 'Y1'})
    df2 = pd.merge(df1, node, how='left', left_on='r', right_on='node')[['s', 'r', 'w', 'X1', 'Y1', 'X', 'Y']].rename(columns={'X': 'X2', 'Y': 'Y2'})
    df2['feat'] = df2[['X1', 'Y1', 'X2', 'Y2']].values.tolist()

    edge_name_to_y = {(s, r): w for s, r, w in df2[['s', 'r', 'w']].values}
    edge_name_to_x = {(s, r): feat for s, r, feat in df2[['s', 'r', 'feat']].values}
        
    device = torch.device('cpu')

    G = nx.from_pandas_edgelist(df2, source='s', target='r', edge_attr='w', create_using=nx.DiGraph())
    airport = from_networkx(G)
    airport.x = torch.from_numpy(node[['X', 'Y']].values).to(torch.float32)
    print(airport)

    G_line_graph = nx.line_graph(G, create_using=nx.DiGraph())
    airport_line_graph = from_networkx(G_line_graph)
    airport_line_graph.x = torch.from_numpy(np.vstack([edge_name_to_x[e] for e in G_line_graph.nodes])).to(torch.float32)
    airport_line_graph.y = torch.from_numpy(np.vstack([edge_name_to_y[e] for e in G_line_graph.nodes])).to(torch.float32)
    print(airport_line_graph)

    split = RandomNodeSplit(num_val=0.1, num_test=0.4)
    data = split(airport_line_graph)
    data = data.to(device)

    edge_array = np.array(list(dict(G_line_graph.nodes).keys()))
    edge_index_train = edge_array[data.train_mask.detach().numpy()]
    edge_index_val = edge_array[data.val_mask.detach().numpy()]
    edge_index_calib_test = edge_array[data.test_mask.detach().numpy()]

    edge_weight_train = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_train])).to(device)
    edge_weight_val = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_val])).to(device)
    edge_weight_calib_test = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_calib_test])).to(device)

    edge_index_train = torch.LongTensor(edge_index_train).T.to(device)
    edge_index_val = torch.LongTensor(edge_index_val).T.to(device)
    edge_index_calib_test = torch.LongTensor(edge_index_calib_test).T.to(device)
    edge_tensor = torch.LongTensor(edge_array).T.to(device)
    edge_weight_gae_training = [edge_name_to_y[tuple(edge)] if train else 1.0 for edge, train in zip(edge_array, data.train_mask.detach().numpy())]
    edge_weight_gae_training = torch.Tensor(edge_weight_gae_training).to(device) # torch.ones(edge_array.shape[0]).to(device)

def Dataset_Ch():
    global nodefile, nodes, nodes_df, node, node_rename, flowfile, colname, flow, scaler
    global df, df1, df2, edge_name_to_y, edge_name_to_x, G, airport, G_line_graph, airport_line_graph, data
    global  edge_array, edge_index_train,  edge_index_val, edge_index_calib_test, edge_weight_train, edge_weight_val, edge_weight_calib_test,  edge_index_train, edge_index_val, edge_index_calib_test
    global edge_tensor, edge_weight_gae_training  
    nodefile = os.path.join(os.path.dirname(__file__), '..', 'data', 'anaheim_nodes.geojson')
    nodes = gpd.read_file(nodefile)
    nodes_df = pd.DataFrame(nodes)
    nodes_df[['X', 'Y']] = nodes_df['geometry'].astype(str).str.split('(').str[-1].str.split(')').str[0].str.split(' ', expand=True).astype(np.float32)
    node = nodes_df.rename(columns={'id': 'node'})

    node_rename = {node: id for id, node in enumerate(range(1, 417))}
    node['node'] = node['node'].map(node_rename)
    flowfile = os.path.join(os.path.dirname(__file__), '..', 'data', 'Anaheim_flow.tntp')
    colname = 'Volume '
    flow = pd.read_csv(flowfile, sep='\t', usecols=['From ', 'To ', colname])

    flow['From '] = flow['From '].map(node_rename)
    flow['To '] = flow['To '].map(node_rename)
    flow = flow[(flow['From '].notna()) & (flow['To '].notna())]
    flow.drop(flow[flow[colname] <= 0].index, inplace=True)
    flow[colname] = np.log(flow[colname])

    scaler = StandardScaler()
    node[['X', 'Y']] = scaler.fit_transform(node[['X', 'Y']].values)
    #minmax = MinMaxScaler()
    #flow[[colname]] = minmax.fit_transform(flow[[colname]].values)

    df = flow.rename(columns={'From ': 's', 'To ': 'r', colname: 'w'})
    df1 = pd.merge(df, node, how='left', left_on='s', right_on='node')[['s', 'r', 'w', 'X', 'Y']].rename(columns={'X': 'X1', 'Y': 'Y1'})
    df2 = pd.merge(df1, node, how='left', left_on='r', right_on='node')[['s', 'r', 'w', 'X1', 'Y1', 'X', 'Y']].rename(columns={'X': 'X2', 'Y': 'Y2'})
    df2['feat'] = df2[['X1', 'Y1', 'X2', 'Y2']].values.tolist()

    edge_name_to_y = {(s, r): w for s, r, w in df2[['s', 'r', 'w']].values}
    edge_name_to_x = {(s, r): feat for s, r, feat in df2[['s', 'r', 'feat']].values}
        

    node_rename = {node: id for id, node in enumerate(range(388, 934))}
    nodefile = os.path.join(os.path.dirname(__file__), '..', 'data', 'ChicagoSketch_node.tntp')
    node = pd.read_csv(nodefile, sep='\t', usecols=['node', 'X', 'Y'])
    flowfile = '..\data\ChicagoSketch_flow.tntp'
    flowfile = os.path.join(os.path.dirname(__file__), '..', 'data', 'ChicagoSketch_flow.tntp')
    #flowfile = '..\data\Anaheim_flow.tntp'
    colname = 'Volume '
    flow = pd.read_csv(flowfile, sep='\t', usecols=['From ', 'To ', colname])

    node['node'] = node['node'].map(node_rename)
    node = node[node['node'].notna()]

    flow['From '] = flow['From '].map(node_rename)
    flow['To '] = flow['To '].map(node_rename)
    flow = flow[(flow['From '].notna()) & (flow['To '].notna())]
    flow.drop(flow[flow[colname] <= 0].index, inplace=True)
    flow[colname] = np.log(flow[colname])

    scaler = StandardScaler()
    node[['X', 'Y']] = scaler.fit_transform(node[['X', 'Y']].values)
    # minmax = MinMaxScaler()
    # flow[[colname]] = minmax.fit_transform(flow[[colname]].values)

    df = flow.rename(columns={'From ': 's', 'To ': 'r', colname: 'w'})
    df1 = pd.merge(df, node, how='left', left_on='s', right_on='node')[['s', 'r', 'w', 'X', 'Y']].rename(columns={'X': 'X1', 'Y': 'Y1'})
    df2 = pd.merge(df1, node, how='left', left_on='r', right_on='node')[['s', 'r', 'w', 'X1', 'Y1', 'X', 'Y']].rename(columns={'X': 'X2', 'Y': 'Y2'})
    df2['feat'] = df2[['X1', 'Y1', 'X2', 'Y2']].values.tolist()

    edge_name_to_y = {(s, r): w for s, r, w in df2[['s', 'r', 'w']].values}
    edge_name_to_x = {(s, r): feat for s, r, feat in df2[['s', 'r', 'feat']].values}



    device = torch.device('cpu')

    G = nx.from_pandas_edgelist(df2, source='s', target='r', edge_attr='w', create_using=nx.DiGraph())
    airport = from_networkx(G)
    airport.x = torch.from_numpy(node[['X', 'Y']].values).to(torch.float32)
    print(airport)

    G_line_graph = nx.line_graph(G, create_using=nx.DiGraph())
    airport_line_graph = from_networkx(G_line_graph)
    airport_line_graph.x = torch.from_numpy(np.vstack([edge_name_to_x[e] for e in G_line_graph.nodes])).to(torch.float32)
    airport_line_graph.y = torch.from_numpy(np.vstack([edge_name_to_y[e] for e in G_line_graph.nodes])).to(torch.float32)
    print(airport_line_graph)

    split = RandomNodeSplit(num_val=0.1, num_test=0.4)
    data = split(airport_line_graph)
    data = data.to(device)

    edge_array = np.array(list(dict(G_line_graph.nodes).keys()))
    edge_index_train = edge_array[data.train_mask.detach().numpy()]
    edge_index_val = edge_array[data.val_mask.detach().numpy()]
    edge_index_calib_test = edge_array[data.test_mask.detach().numpy()]

    edge_weight_train = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_train])).to(device)
    edge_weight_val = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_val])).to(device)
    edge_weight_calib_test = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_calib_test])).to(device)

    edge_index_train = torch.LongTensor(edge_index_train).T.to(device)
    edge_index_val = torch.LongTensor(edge_index_val).T.to(device)
    edge_index_calib_test = torch.LongTensor(edge_index_calib_test).T.to(device)
    edge_tensor = torch.LongTensor(edge_array).T.to(device)
    edge_weight_gae_training = [edge_name_to_y[tuple(edge)] if train else 1.0 for edge, train in zip(edge_array, data.train_mask.detach().numpy())]
    edge_weight_gae_training = torch.Tensor(edge_weight_gae_training).to(device) # torch.ones(edge_array.shape[0]).to(device)
# w_min, w_max = edge_weight_gae_training.min(), edge_weight_gae_training.max()
# edge

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gconv=SAGEConv, edge_weight=False):
        super().__init__()
        if gconv == 'SAGEConv':
            gconv = SAGEConv
        elif gconv == 'GraphConv':
            gconv = GraphConv
        elif gconv == 'GCNConv':
            gconv = GCNConv
        elif gconv == 'GATConv':
            gconv = GATConv
        self.conv1 = gconv(in_channels, hidden_channels)
        self.conv2 = gconv(hidden_channels, out_channels)
        self.gconv = gconv
        self.edge_weighted = edge_weight

    def forward(self, x, edge_index, edge_weight=None):
        if self.gconv not in [GCNConv, GraphConv]:
            edge_weight = None
        if edge_weight is not None:
            if self.edge_weighted =='True':
                edge_weight = (edge_weight).sigmoid()
            else:
                edge_weight = None
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class DirectedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, gconv=SAGEConv, edge_weight=False):
        super().__init__()
        if gconv == 'SAGEConv':
            gconv = SAGEConv
        elif gconv == 'GraphConv':
            gconv = GraphConv
        elif gconv == 'GCNConv':
            gconv = GCNConv
        elif gconv == 'GATConv':
            gconv = GATConv
        self.layers = [in_channels, hidden_channels, out_channels]
        self.num_layers = len(self.layers) - 1
        self.source = torch.nn.ModuleList()
        self.target = torch.nn.ModuleList()
        self.gconv = gconv
        self.edge_weighted = edge_weight
        for n_in, n_out in zip(self.layers[:-1], self.layers[1:]):
            self.source.append(gconv(n_in, n_out))
            self.target.append(gconv(n_in, n_out))

    def forward(self, s, t, edge_index, edge_weight=None):
        if self.gconv not in [GCNConv, GraphConv]:
            edge_weight = None
        if edge_weight is not None:
            if self.edge_weighted == 'True':
                edge_weight = (edge_weight).sigmoid()
            else:
                edge_weight = None
        for layer_id, (layer_s, layer_t) in enumerate(zip(self.source, self.target)):
            s_new = layer_s(t, edge_index, edge_weight)
            t_new = layer_t(s, torch.flip(edge_index, [0]), edge_weight)
            if layer_id < self.num_layers - 1:
                s_new = s_new.relu()
                t_new = t_new.relu()
                s_new = F.dropout(s_new, p=0.5, training=self.training)
                t_new = F.dropout(t_new, p=0.5, training=self.training)
            s = s_new
            t = t_new

        return s, t

def cqr_new(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    cal_scores = np.maximum(cal_labels-cal_upper, cal_lower-cal_labels)
    cal_scores = cal_scores / np.abs(cal_upper - cal_lower)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    prediction_sets = [val_lower - qhat * np.abs(val_upper - val_lower), val_upper + qhat * np.abs(val_upper - val_lower)]
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    eff = np.mean(prediction_sets[1] - prediction_sets[0])
    return prediction_sets, cov, eff

def cqr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    cal_scores = np.maximum(cal_labels-cal_upper, cal_lower-cal_labels)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    prediction_sets = [val_lower - qhat, val_upper + qhat]
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    eff = np.mean(val_upper + qhat - (val_lower - qhat))
    return prediction_sets, cov, eff

def qr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n, alpha):
    prediction_sets = [val_lower, val_upper]
    cov = ((val_labels >= prediction_sets[0]) & (val_labels <= prediction_sets[1])).mean()
    eff = np.mean(val_upper - val_lower)
    return prediction_sets, cov, eff

def worst_slice_coverage(x, edge_index_calib_test, idx, val_labels, prediction_sets):
    if torch.is_tensor(x):
        x = x.detach().numpy()
    if torch.is_tensor(edge_index_calib_test):
        edge_index_calib_test = edge_index_calib_test.detach().numpy()
    xtest = np.hstack([x[edge_index_calib_test[0, ~idx]], x[edge_index_calib_test[1, ~idx]]])
    ntest = xtest.shape[0]
    nfeat = xtest.shape[1]
    xtest_test = xtest[:ntest//4]
    unitvec = np.random.randn(nfeat, 1000)
    unitvec = unitvec / np.sqrt((unitvec**2).sum(axis=0))
    # ab_range = np.quantile((xtest_test @ unitvec).flatten(), np.linspace(0, 1, 11))
    values = (xtest_test @ unitvec).flatten()
    ab_range = np.linspace(values.min(), values.max(), 10)

    ws_cov_min = None
    for delta in np.linspace(0.1, 0.5, 5):
        ws_cov = 1
        ws_a = None
        ws_b = None
        ws_vec = None
        for vec in unitvec.T:
            value_vec = xtest_test @ vec.reshape(-1, 1)
            for a, b in zip(ab_range[:-1], ab_range[1:]):
                contained = np.bitwise_and(value_vec > a, value_vec < b).flatten()
                if contained.mean() > delta:
                    conditional_cov = ((val_labels[:ntest//4][contained] >= prediction_sets[0][:ntest//4][contained]) & (val_labels[:ntest//4][contained] <= prediction_sets[1][:ntest//4][contained])).mean()
                    if conditional_cov < ws_cov:
                        #print(f"Worst-Slice coverage = {conditional_cov:.4f}")
                        ws_cov = conditional_cov
                        ws_a = a
                        ws_b = b
                        ws_vec = vec
        if ws_vec is None:
            return None
        xtest_true = xtest[ntest//4:]
        value_vec = xtest_true @ ws_vec.reshape(-1, 1)
        contained = np.bitwise_and(value_vec > ws_a, value_vec < ws_b).flatten()
        ws_cov_true = ((val_labels[ntest//4:][contained] >= prediction_sets[0][ntest//4:][contained]) & (val_labels[ntest//4:][contained] <= prediction_sets[1][ntest//4:][contained])).mean()
        if ws_cov_min is not None and ws_cov_true < ws_cov_min:
            ws_cov_min = ws_cov_true
        elif ws_cov_min is None and ~np.isnan(ws_cov_true):
            ws_cov_min = ws_cov_true
    return ws_cov_min

EPOCHS = 5001
ALPHA = 0.05
LR = 0.01
WD = 5e-4
HIDDEN = 8
OUT = 2

SCORE = 'cqr'
#GNNCONV = GraphConv

def build_gae(gconv, CQR_or_CP='CQR'):
    if CQR_or_CP == 'CQR':
        encoder = GNN(in_channels=airport.x.shape[-1], hidden_channels=8, out_channels=3*OUT, gconv=gconv)
    else:
        encoder = GNN(in_channels=airport.x.shape[-1], hidden_channels=8, out_channels=OUT, gconv=gconv)
    decoder = InnerProductDecoder()
    model = GAE(encoder, decoder).to(device)
    return model


def train_gae_directed(model, optimizer, x, edge_index_train, edge_weight, alpha=ALPHA, val=False, edge_index_val=None, sigmoid=False):
    if val:
        model.eval()
    else:
        model.train()
    Z_source, Z_target = model(x, x, edge_tensor, edge_weight_gae_training)
    out_dim = Z_source.shape[-1] // 3
    z_mid_source = Z_source[:, :out_dim]; z_lower_source = Z_source[:, out_dim:2*out_dim]; z_upper_source = Z_source[:, 2*out_dim:]
    z_mid_target = Z_target[:, :out_dim]; z_lower_target = Z_target[:, out_dim:2*out_dim]; z_upper_target = Z_target[:, 2*out_dim:]
    if val:
        out = model.decoder(z_mid_source, z_mid_target, edge_index_val, sigmoid=sigmoid)
        lower = model.decoder(z_lower_source, z_lower_target, edge_index_val, sigmoid=sigmoid)
        upper = model.decoder(z_upper_source, z_upper_target, edge_index_val, sigmoid=sigmoid)
    else:
        out = model.decoder(z_mid_source, z_mid_target, edge_index_train, sigmoid=sigmoid)
        lower = model.decoder(z_lower_source, z_lower_target, edge_index_train, sigmoid=sigmoid)
        upper = model.decoder(z_upper_source, z_upper_target, edge_index_train, sigmoid=sigmoid)

    label = edge_weight
    mse_loss = F.mse_loss(out, label)
    low_bound = alpha / 2; upp_bound = 1 - alpha / 2
    low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
    upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
    loss = low_loss + upp_loss # mse_loss +

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss)

def run_conformal_regression_gae(labels, lower, upper, alpha, return_prediction_sets=False, return_conditional_coverage=False, score='cqr', x=None, edge_index_calib_test=None):
    num_runs = 100
    if torch.is_tensor(labels):
        labels = labels.detach().numpy()
    if torch.is_tensor(upper):
        upper = upper.detach().numpy()
    if torch.is_tensor(lower):
        lower = lower.detach().numpy()

    n_test_calib = labels.shape[0]
    n_calib = n_test_calib // 2
    idx = np.array([1] * n_calib + [0] * (n_test_calib-n_calib)) > 0

    cov_all = []
    eff_all = []
    if return_conditional_coverage:
        ws_cov_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []
    for k in range(num_runs):
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_labels, val_labels = labels[idx], labels[~idx]
        cal_upper, val_upper = upper[idx], upper[~idx]
        cal_lower, val_lower = lower[idx], lower[~idx]
        if score == 'cqr':
            prediction_sets, cov, eff = cqr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n_test_calib, alpha)
        elif score == 'qr':
            prediction_sets, cov, eff = qr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n_test_calib, alpha)
        elif score == 'cqr_new':
            prediction_sets, cov, eff = cqr_new(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n_test_calib, alpha)
        if return_conditional_coverage:
            ws_cov = worst_slice_coverage(x, edge_index_calib_test, idx, val_labels, prediction_sets)
            if ws_cov is not None:
                ws_cov_all.append(ws_cov)
        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)

    if return_prediction_sets:
        if return_conditional_coverage:
            return cov_all, eff_all, ws_cov_all, pred_set_all, val_labels_all, idx_all
        return cov_all, eff_all, pred_set_all, val_labels_all, idx_all
    else:
        return np.mean(cov_all), np.mean(eff_all)

def test_gae_directed(best_model, x, train_edge_index, calib_test_edge_index, calib_test_edge_weight, alpha=ALPHA, return_prediction_sets=False, score='cqr', conditional=True, sigmoid=False):
    best_model = best_model.cpu()
    best_model.eval()
    Z_source, Z_target = best_model(x.cpu(), x.cpu(), edge_tensor.cpu(), edge_weight_gae_training.cpu())
    out_dim = Z_source.shape[-1] // 3
    z_mid_source = Z_source[:, :out_dim]; z_lower_source = Z_source[:, out_dim:2*out_dim]; z_upper_source = Z_source[:, 2*out_dim:]
    z_mid_target = Z_target[:, :out_dim]; z_lower_target = Z_target[:, out_dim:2*out_dim]; z_upper_target = Z_target[:, 2*out_dim:]

    out = best_model.decoder(z_mid_source, z_mid_target, calib_test_edge_index.cpu(), sigmoid=sigmoid)
    lower = best_model.decoder(z_lower_source, z_lower_target, calib_test_edge_index.cpu(), sigmoid=sigmoid)
    upper = best_model.decoder(z_upper_source, z_upper_target, calib_test_edge_index.cpu(), sigmoid=sigmoid)
    if conditional:
        return run_conformal_regression_gae(calib_test_edge_weight.cpu(), lower, upper, alpha, return_prediction_sets=return_prediction_sets, return_conditional_coverage=True, x=x, edge_index_calib_test=calib_test_edge_index, score=score)

    return run_conformal_regression_gae(calib_test_edge_weight.cpu(), lower, upper, alpha, return_prediction_sets=return_prediction_sets, score=score)

def test_gae_directed_basegae(best_model_train, x, train_edge_index, calib_test_edge_index, calib_test_edge_weight, alpha=ALPHA, return_prediction_sets=False, sigmoid=False, conditional=False):
    best_model_train = best_model_train.cpu()
    best_model_train.eval()
    Z_source, Z_target = best_model_train(x.cpu(), x.cpu(), edge_tensor.cpu(), edge_weight_gae_training.cpu())
    out = best_model_train.decoder(Z_source, Z_target, calib_test_edge_index.cpu(), sigmoid=sigmoid)

    if conditional:
        return run_conformal_regression_basegae(calib_test_edge_weight.cpu(), out, alpha, return_prediction_sets=return_prediction_sets, return_conditional_coverage=True, x=x, edge_index_calib_test=calib_test_edge_index)
    return run_conformal_regression_basegae(calib_test_edge_weight.cpu(), out, alpha, return_prediction_sets=return_prediction_sets)


def train_gae_basegae(model, optimizer, x, edge_index_train, edge_weight, alpha=ALPHA, val=False, edge_index_val=None, sigmoid=False):
    if val:
        model.eval()
    else:
        model.train()
    Z = model(x, edge_tensor, edge_weight_gae_training)
    if val:
        out = model.decoder(Z, edge_index_val, sigmoid=sigmoid)
    else:
        out = model.decoder(Z, edge_index_train, sigmoid=sigmoid)

    label = edge_weight
    mse_loss = F.mse_loss(out, label)
    loss = mse_loss

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss)

def test_gae_basegae(best_model_train, x, train_edge_index, calib_test_edge_index, calib_test_edge_weight, alpha=ALPHA, return_prediction_sets=False, sigmoid=False, conditional=False):
    best_model_train = best_model_train.cpu()
    best_model_train.eval()
    Z = best_model_train(x.cpu(), edge_tensor.cpu(), edge_weight_gae_training.cpu())
    out = best_model_train.decoder(Z, calib_test_edge_index.cpu(), sigmoid=sigmoid)

    if conditional:
        return run_conformal_regression_basegae(calib_test_edge_weight.cpu(), out, alpha, return_prediction_sets=return_prediction_sets, return_conditional_coverage=True, x=x, edge_index_calib_test=calib_test_edge_index)
    return run_conformal_regression_basegae(calib_test_edge_weight.cpu(), out, alpha, return_prediction_sets=return_prediction_sets)

def train_gae(model, optimizer, x, edge_index_train, edge_weight, alpha=ALPHA, val=False, edge_index_val=None, sigmoid=False):
    if val:
        model.eval()
    else:
        model.train()
    Z = model(x, edge_tensor, edge_weight_gae_training)
    out_dim = Z.shape[-1] // 3
    z_mid = Z[:, :out_dim]; z_lower = Z[:, out_dim:2*out_dim]; z_upper = Z[:, 2*out_dim:]
    if val:
        out = model.decoder(z_mid, edge_index_val, sigmoid=sigmoid)
        lower = model.decoder(z_lower, edge_index_val, sigmoid=sigmoid)
        upper = model.decoder(z_upper, edge_index_val, sigmoid=sigmoid)
    else:
        out = model.decoder(z_mid, edge_index_train, sigmoid=sigmoid)
        lower = model.decoder(z_lower, edge_index_train, sigmoid=sigmoid)
        upper = model.decoder(z_upper, edge_index_train, sigmoid=sigmoid)

    label = edge_weight
    mse_loss = F.mse_loss(out, label)
    low_bound = alpha / 2; upp_bound = 1 - alpha / 2
    low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
    upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
    loss = low_loss + upp_loss # mse_loss +

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss)

def test_gae(best_model, x, train_edge_index, calib_test_edge_index, calib_test_edge_weight, alpha=ALPHA, return_prediction_sets=False, score='cqr', conditional=True, sigmoid=False):
    best_model = best_model.cpu()
    best_model.eval()
    Z = best_model(x.cpu(), edge_tensor.cpu(), edge_weight_gae_training.cpu())
    out_dim = Z.shape[-1] // 3
    z_mid = Z[:, :out_dim]; z_lower = Z[:, out_dim:2*out_dim]; z_upper = Z[:, 2*out_dim:]
    model=best_model
    out = model.decoder(z_mid, calib_test_edge_index.cpu(), sigmoid=sigmoid)
    lower = model.decoder(z_lower, calib_test_edge_index.cpu(), sigmoid=sigmoid)
    upper = model.decoder(z_upper, calib_test_edge_index.cpu(), sigmoid=sigmoid)
    if conditional:
        return run_conformal_regression_gae(calib_test_edge_weight.cpu(), lower, upper, alpha, return_prediction_sets=return_prediction_sets, return_conditional_coverage=True, x=x, edge_index_calib_test=calib_test_edge_index,score=score)

    return run_conformal_regression_gae(calib_test_edge_weight.cpu(), lower, upper, alpha, return_prediction_sets=return_prediction_sets, score=score)


def train_gae_for_loop(model, optimizer, x, edge_index_train_or_val, edge_weight_train_or_val, edge_index_all, edge_weight_filled_all=None,alpha=ALPHA, val=False):
    """
    x: all the node features, n*2
    edge_index_train_or_val: only the edge indices of the training (validation) edges, 2*m_train
    edge_weight_train_or_val: only the edge weights of the training (validation) edges, m_train
    edge_index_all: all the edge indices, 2*m
    edge_weight_filled_all: all the edge weights (with the weights of remaining edges appended with 1), m
    alpha: self-defined error rate, ALPHA=0.05
    val: validation or training
    """

    if val:
        model.eval()
    else:
        model.train()
    Z = model(x, edge_index_all, edge_weight_filled_all)
    out_dim = Z.shape[-1] // 3
    z_mid = Z[:, :out_dim]; z_lower = Z[:, out_dim:2*out_dim]; z_upper = Z[:, 2*out_dim:]
    out = model.decoder(z_mid, edge_index_train_or_val)
    lower = model.decoder(z_lower, edge_index_train_or_val)
    upper = model.decoder(z_upper, edge_index_train_or_val)

    label = edge_weight_train_or_val
    low_bound = alpha / 2; upp_bound = 1 - alpha / 2
    low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
    upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
    loss = low_loss + upp_loss

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # cov_all, eff_all, ws_cov_all, pred_set_all, val_labels_all, idx_all = test_gae_directed(best_model, x, edge_index_train, edge_index_calib_test, edge_weight_calib_test,                 
    # return_prediction_sets=True, score='cqr_new', sigmoid=use_sigmoid, conditional=True)
    #cov_all, eff_all, pred_set_all, val_labels_all, idx_all = test_gae_directed(best_model, x, edge_index_train, edge_index_calib_test, edge_weight_calib_test,                           _prediction_sets=True, score=SCORE, sigmoid=use_sigmoid, conditional=False)

    return float(loss)


def train_baselinegraph(model, optimizer, alpha=ALPHA, val=False):
    if val:
        model.eval()
    else:
        model.train()
    mask = data.val_mask if val else data.train_mask
    out = model(data.x, data.edge_index)
    label = data.y[mask]
    mse_loss = F.mse_loss(out[mask], data.y[mask])
    loss = mse_loss

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss)


def cqr_baselinegraph(cal_y, val_y, cal_yhat, val_yhat, n, alpha):
    cal_scores = np.abs(cal_y - cal_yhat)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    prediction_sets = [val_yhat - qhat, val_yhat + qhat]
    cov = ((val_y >= prediction_sets[0]) & (val_y <= prediction_sets[1])).mean()
    eff = np.mean(2 * qhat)  # np.mean(neg_qhat + pos_qhat)
    return prediction_sets, cov, eff

def run_conformal_regression_baselinegraph(pred, true, n_test_calib, alpha, return_prediction_sets=False, return_conditional_coverage=False, x=None, edge_index_calib_test=None):
    num_runs = 100
    n_calib = n_test_calib // 2
    try:
        pred = pred.detach().cpu().numpy()
    except:
        pass

    smx = pred[true.test_mask]
    labels = true.y[true.test_mask].detach().cpu().numpy().reshape(-1)
    idx = np.array([1] * n_calib + [0] * (labels.shape[0]-n_calib)) > 0

    cov_all = []
    eff_all = []
    if return_conditional_coverage:
        ws_cov_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []
    for k in range(num_runs):
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_labels, val_labels = labels[idx], labels[~idx]
        cal_predict, val_predict = smx[idx], smx[~idx]
        prediction_sets, cov, eff = cqr_baselinegraph(cal_labels, val_labels, cal_predict, val_predict, n_test_calib, alpha)
        if return_conditional_coverage:
            ws_cov = worst_slice_coverage(x, edge_index_calib_test, idx, val_labels, prediction_sets)
            if ws_cov is not None:
                ws_cov_all.append(ws_cov)
        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)

    if return_prediction_sets:
        if return_conditional_coverage:
            return cov_all, eff_all, ws_cov_all, pred_set_all, val_labels_all, idx_all
        return cov_all, eff_all, pred_set_all, val_labels_all, idx_all
    else:
        return np.mean(cov_all), np.mean(eff_all)


def test_baselinegraph(best_model_train, alpha=ALPHA, return_prediction_sets=False, conditional=True, x=None, calib_test_edge_index=None):
    best_model_train = best_model_train.cpu()
    best_model_train.eval()
    out = best_model_train(data.cpu().x, data.cpu().edge_index)
    if conditional:
        return run_conformal_regression_baselinegraph(out, data.cpu(), int(data.test_mask.sum()), alpha, return_prediction_sets=return_prediction_sets, return_conditional_coverage=True, x=x, edge_index_calib_test=calib_test_edge_index)
    return run_conformal_regression_baselinegraph(out, data.cpu(), int(data.test_mask.sum()), alpha, return_prediction_sets=return_prediction_sets)

def train_gae_directed_basegae(model, optimizer, x, edge_index_train, edge_weight, alpha=ALPHA, val=False, edge_index_val=None, sigmoid=False):
    if val:
        model.eval()
    else:
        model.train()
    Z_source, Z_target = model(x, x, edge_tensor, edge_weight_gae_training)
    if val:
        out = model.decoder(Z_source, Z_target, edge_index_val, sigmoid=sigmoid)
    else:
        out = model.decoder(Z_source, Z_target, edge_index_train, sigmoid=sigmoid)

    label = edge_weight
    mse_loss = F.mse_loss(out, label)
    loss = mse_loss

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss)

def cqr_basegae(cal_y, val_y, cal_yhat, val_yhat, n, alpha):
    cal_scores = np.abs(cal_y - cal_yhat)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, method='higher')
    prediction_sets = [val_yhat - qhat, val_yhat + qhat]
    cov = ((val_y >= prediction_sets[0]) & (val_y <= prediction_sets[1])).mean()
    eff = np.mean(2 * qhat)
    return prediction_sets, cov, eff

def run_conformal_regression_basegae(labels, out, alpha, return_prediction_sets=False, num_runs=100, return_conditional_coverage=False, x=None, edge_index_calib_test=None):
    if torch.is_tensor(labels):
        labels = labels.detach().numpy()
    if torch.is_tensor(out):
        out = out.detach().numpy()

    n_test_calib = labels.shape[0]
    n_calib = n_test_calib // 2
    idx = np.array([1] * n_calib + [0] * (n_test_calib-n_calib)) > 0

    cov_all = []
    eff_all = []
    if return_conditional_coverage:
        ws_cov_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []
    for k in range(num_runs):
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_labels, val_labels = labels[idx], labels[~idx]
        cal_predict, val_predict = out[idx], out[~idx]
        prediction_sets, cov, eff = cqr_basegae(cal_labels, val_labels, cal_predict, val_predict, n_test_calib, alpha)
        if return_conditional_coverage:
            ws_cov = worst_slice_coverage(x, edge_index_calib_test, idx, val_labels, prediction_sets)
            if ws_cov is not None:
                ws_cov_all.append(ws_cov)
        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)

    if return_prediction_sets:
        if return_conditional_coverage:
            return cov_all, eff_all, ws_cov_all, pred_set_all, val_labels_all, idx_all
        return cov_all, eff_all, pred_set_all, val_labels_all, idx_all
    else:
        return np.mean(cov_all), np.mean(eff_all) # coverage and efficiency averaged over multiple runs

def test_gae_directed_basegae(best_model_train, x, train_edge_index, calib_test_edge_index, calib_test_edge_weight, alpha=ALPHA, return_prediction_sets=False, sigmoid=False, conditional=False):
    best_model_train = best_model_train.cpu()
    best_model_train.eval()
    Z_source, Z_target = best_model_train(x.cpu(), x.cpu(), edge_tensor.cpu(), edge_weight_gae_training.cpu())
    out = best_model_train.decoder(Z_source, Z_target, calib_test_edge_index.cpu(), sigmoid=sigmoid)

    if conditional:
        return run_conformal_regression_basegae(calib_test_edge_weight.cpu(), out, alpha, return_prediction_sets=return_prediction_sets, return_conditional_coverage=True, x=x, edge_index_calib_test=calib_test_edge_index)
    return run_conformal_regression_basegae(calib_test_edge_weight.cpu(), out, alpha, return_prediction_sets=return_prediction_sets)

def train_gae_basegae(model, optimizer, x, edge_index_train, edge_weight, alpha=ALPHA, val=False, edge_index_val=None, sigmoid=False):
    if val:
        model.eval()
    else:
        model.train()
    Z = model(x, edge_tensor, edge_weight_gae_training)
    if val:
        out = model.decoder(Z, edge_index_val, sigmoid=sigmoid)
    else:
        out = model.decoder(Z, edge_index_train, sigmoid=sigmoid)

    label = edge_weight
    mse_loss = F.mse_loss(out, label)
    loss = mse_loss

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss)


def train_linegraph(model, optimizer, alpha=ALPHA, val=False):
    if val:
        model.eval()
    else:
        model.train()
    mask = data.val_mask if val else data.train_mask
    out = model(data.x, data.edge_index)
    label = data.y[mask]
    mse_loss = F.mse_loss(out[:, 0][mask], data.y[mask])
    low_bound = alpha / 2
    upp_bound = 1 - alpha / 2
    lower = out[:, 1][mask].reshape(-1,1)
    upper = out[:, 2][mask].reshape(-1,1)
    low_loss = torch.mean(torch.max((low_bound - 1) * (label - lower), low_bound * (label - lower)))
    upp_loss = torch.mean(torch.max((upp_bound - 1) * (label - upper), upp_bound * (label - upper)))
    loss = mse_loss + low_loss + upp_loss

    if not val:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss)


def run_conformal_regression_linegraph(pred, data, n_test_calib, alpha, return_prediction_sets=False, return_conditional_coverage=False, score='cqr', x=None, edge_index_calib_test=None):
    num_runs = 100
    n_calib = n_test_calib // 2
    try:
        pred = pred.detach().cpu().numpy()
    except:
        pass

    smx = pred[data.test_mask]
    labels = data.y[data.test_mask].detach().cpu().numpy().reshape(-1)
    upper, lower = smx[:, 2], smx[:, 1]
    idx = np.array([1] * n_calib + [0] * (labels.shape[0]-n_calib)) > 0

    cov_all = []
    eff_all = []
    if return_conditional_coverage:
        ws_cov_all = []
    if return_prediction_sets:
        pred_set_all = []
        val_labels_all = []
        idx_all = []
    for k in range(num_runs):
        np.random.seed(k)
        np.random.shuffle(idx)
        if return_prediction_sets:
            idx_all.append(idx)
        cal_labels, val_labels = labels[idx], labels[~idx]
        cal_upper, val_upper = upper[idx], upper[~idx]
        cal_lower, val_lower = lower[idx], lower[~idx]
        if score == 'cqr':
            prediction_sets, cov, eff = cqr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n_test_calib, alpha)
        elif score == 'qr':
            prediction_sets, cov, eff = qr(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n_test_calib, alpha)
        elif score == 'cqr_new':
            prediction_sets, cov, eff = cqr_new(cal_labels, cal_lower, cal_upper, val_labels, val_lower, val_upper, n_test_calib, alpha)
        if return_conditional_coverage:
            ws_cov = worst_slice_coverage(x, edge_index_calib_test, idx, val_labels, prediction_sets)
            if ws_cov is not None:
                ws_cov_all.append(ws_cov)
        cov_all.append(cov)
        eff_all.append(eff)
        if return_prediction_sets:
            pred_set_all.append(prediction_sets)
            val_labels_all.append(val_labels)

    if return_prediction_sets:
        if return_conditional_coverage:
            return cov_all, eff_all, ws_cov_all, pred_set_all, val_labels_all, idx_all
        return cov_all, eff_all, pred_set_all, val_labels_all, idx_all
    else:
        return np.mean(cov_all), np.mean(eff_all)


def test_linegraph(best_model_train, alpha=ALPHA, return_prediction_sets=False, score='cqr', conditional=True, x=None, calib_test_edge_index=None):
    best_model_train = best_model_train.cpu()
    best_model_train.eval()
    out = best_model_train(data.cpu().x, data.cpu().edge_index)
    if conditional:
        return run_conformal_regression_linegraph(out, data.cpu(), int(data.test_mask.sum()), alpha, return_prediction_sets=return_prediction_sets, return_conditional_coverage=True, x=x, edge_index_calib_test=calib_test_edge_index, score=score)
    return run_conformal_regression_linegraph(out, data.cpu(), int(data.test_mask.sum()), alpha, return_prediction_sets=return_prediction_sets, score=score)

cov_all_f=[]
ineff_all_f=[]

#optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
def function_model(DiGAE_or_GAE_or_LGNN, CP_or_CQR, SCORE, SEEDNUM, GNNCONV, Weighted, Conditional, Dataset):
    if Dataset == "An":
        Dataset_An()
    elif Dataset == "Ch":
        Dataset_Ch()

    for kk in range(SEEDNUM):
        seed_everything(kk)
        
        split = RandomNodeSplit(num_val=0.1, num_test=0.4)
        data = split(airport_line_graph)
        device = torch.device('cpu')

        data = data.to(device)
        edge_index_train = edge_array[data.train_mask.detach().numpy()]
        edge_index_val = edge_array[data.val_mask.detach().numpy()]
        edge_index_calib_test = edge_array[data.test_mask.detach().numpy()]
        edge_weight_train = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_train])).to(device)
        edge_weight_val = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_val])).to(device)
        edge_weight_calib_test = torch.Tensor(np.stack([edge_name_to_y[tuple(edge)] for edge in edge_index_calib_test])).to(device)
        edge_index_train = torch.LongTensor(edge_index_train).T.to(device)
        edge_index_val = torch.LongTensor(edge_index_val).T.to(device)
        edge_index_calib_test = torch.LongTensor(edge_index_calib_test).T.to(device)
        edge_weight_gae_training = [edge_name_to_y[tuple(edge)] if train else np.mean(list(edge_name_to_y.values())) for edge, train in zip(edge_array, data.train_mask.detach().numpy())]
        edge_weight_gae_training = torch.Tensor(edge_weight_gae_training).to(device)


        if DiGAE_or_GAE_or_LGNN == 'DiGAE':
            if CP_or_CQR == 'CQR':
                out_channels_f=3*OUT
            elif CP_or_CQR == 'CP':
                out_channels_f=OUT
            encoder = DirectedGNN(in_channels=airport.x.shape[-1], hidden_channels=HIDDEN, out_channels=out_channels_f, gconv=GNNCONV, edge_weight=Weighted)
            decoder = DirectedInnerProductDecoder()
            model = GAE(encoder, decoder).to(device)
        elif DiGAE_or_GAE_or_LGNN == 'GAE':
            if CP_or_CQR == 'CQR':
                out_channels_f=3*OUT
            elif CP_or_CQR == 'CP':
                out_channels_f=OUT
            encoder = GNN(in_channels=airport.x.shape[-1], hidden_channels=HIDDEN, out_channels=out_channels_f, gconv=GNNCONV, edge_weight=Weighted)
            decoder = InnerProductDecoder()
            model = GAE(encoder, decoder).to(device)
        elif DiGAE_or_GAE_or_LGNN == 'LGNN':
            if CP_or_CQR == 'CQR':
                out_channels_f=3
            elif CP_or_CQR == 'CP':
                out_channels_f=1
            model = GNN(in_channels=data.num_features, hidden_channels=32, out_channels=out_channels_f, gconv=GNNCONV, edge_weight=Weighted).to(device)

        #print(model)
        x = airport.x.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        best_val_loss = float('inf')
        best_model = None
        use_sigmoid = False
        for epoch in range(1, EPOCHS):
            if DiGAE_or_GAE_or_LGNN == 'DiGAE':
                if CP_or_CQR == 'CQR':
                    loss = train_gae_directed(model, optimizer, x, edge_index_train, edge_weight_train, sigmoid=use_sigmoid)
                elif CP_or_CQR == 'CP':
                    loss = train_gae_directed_basegae(model, optimizer, x, edge_index_train, edge_weight_train, sigmoid=use_sigmoid)
            elif DiGAE_or_GAE_or_LGNN == 'GAE':
                if CP_or_CQR == 'CQR':
                    loss = train_gae(model, optimizer, x, edge_index_train, edge_weight_train, sigmoid=use_sigmoid)
                elif CP_or_CQR == 'CP':
                    loss = train_gae_basegae(model, optimizer, x, edge_index_train, edge_weight_train, sigmoid=use_sigmoid)
            elif DiGAE_or_GAE_or_LGNN == 'LGNN':
                if CP_or_CQR == 'CQR':
                    loss = train_linegraph(model, optimizer)
                elif CP_or_CQR == 'CP':
                    loss = train_baselinegraph(model, optimizer)
            #if epoch % 100 == 1:
            #    print(f'Step: {epoch:03d}, Loss: {loss:.4f}')
            if DiGAE_or_GAE_or_LGNN == 'DiGAE':
                if CP_or_CQR == 'CQR':
            	    val_loss = train_gae_directed(model, optimizer, x, edge_index_train, edge_weight_val, val=True, edge_index_val=edge_index_val, sigmoid=use_sigmoid)
                elif CP_or_CQR == 'CP':
                    val_loss = train_gae_directed_basegae(model, optimizer, x, edge_index_train, edge_weight_val, val=True, edge_index_val=edge_index_val, sigmoid=use_sigmoid)
            elif DiGAE_or_GAE_or_LGNN == 'GAE':
                if CP_or_CQR == 'CQR':
                    val_loss = train_gae(model, optimizer, x, edge_index_train, edge_weight_val, val=True, edge_index_val=edge_index_val, sigmoid=use_sigmoid)
                elif CP_or_CQR == 'CP':
                    val_loss = train_gae_basegae(model, optimizer, x, edge_index_train, edge_weight_val, val=True, edge_index_val=edge_index_val, sigmoid=use_sigmoid)
            elif DiGAE_or_GAE_or_LGNN == 'LGNN':
                if CP_or_CQR == 'CQR':
                    val_loss = train_linegraph(model, optimizer, val=True)
                elif CP_or_CQR == 'CP':
                    val_loss = train_baselinegraph(model, optimizer, val=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
                #print(f'Step: {epoch:03d}, Best validation loss: {val_loss:.4f}')
        if DiGAE_or_GAE_or_LGNN == 'DiGAE':
            if CP_or_CQR != 'CP':
                if Conditional == 'False':
                    cov_all, eff_all, pred_set_all, val_labels_all, idx_all = test_gae_directed(best_model, x, edge_index_train, edge_index_calib_test, edge_weight_calib_test, return_prediction_sets=True, score=SCORE, sigmoid=use_sigmoid, conditional=False)
                elif Conditional =='True':
                    cov_all_o, eff_all, cov_all, pred_set_all, val_labels_all, idx_all = test_gae_directed(best_model, x, edge_index_train, edge_index_calib_test, edge_weight_calib_test, return_prediction_sets=True, score=SCORE, sigmoid=use_sigmoid, conditional=True)

                cov_all_f.extend(cov_all)
                ineff_all_f.extend(eff_all)
                print(f"{np.mean(cov_all_f):.4f}+/-{np.std(cov_all_f):.4f}, {np.mean(ineff_all_f):.4f}+/-{np.std(ineff_all_f):.4f}")
            elif CP_or_CQR == 'CP': 
                if Conditional =='False':
                    cov_all_basegae, eff_all_basegae, pred_set_all_basegae, val_labels_all_basegae, idx_all_basegae = test_gae_directed_basegae(best_model, x, edge_index_train, edge_index_calib_test, edge_weight_calib_test, return_prediction_sets=True, sigmoid=use_sigmoid, conditional=False)
                else:
                    cov_all_basegae_o, eff_all_basegae, cov_all_basegae, pred_set_all_basegae, val_labels_all_basegae, idx_all_basegae = test_gae_directed_basegae(best_model, x, edge_index_train, edge_index_calib_test, edge_weight_calib_test, return_prediction_sets=True, sigmoid=use_sigmoid, conditional=True)
                cov_all_f.extend(cov_all_basegae)
                ineff_all_f.extend(eff_all_basegae)
                #print(f"{np.mean(cov_all_basegae):.4f}+/-{np.std(cov_all_basegae):.4f}, {np.mean(eff_all_basegae):.4f}+/-{np.std(eff_all_basegae):.4f}")
                print(f"{np.mean(cov_all_f):.4f}+/-{np.std(cov_all_f):.4f}, {np.mean(ineff_all_f):.4f}+/-{np.std(ineff_all_f):.4f}")
        elif DiGAE_or_GAE_or_LGNN == 'GAE':
            if CP_or_CQR != 'CP':
                cov_all, eff_all, pred_set_all, val_labels_all, idx_all = test_gae(best_model, x, edge_index_train, edge_index_calib_test, edge_weight_calib_test, return_prediction_sets=True, score=SCORE, sigmoid=use_sigmoid, conditional=False)
                #print(f"{np.mean(cov_all):.4f}+/-{np.std(cov_all):.4f}, {np.mean(eff_all):.4f}+/-{np.std(eff_all):.4f}")
                cov_all_f.extend(cov_all)
                ineff_all_f.extend(eff_all)
                print(f"{np.mean(cov_all_f):.4f}+/-{np.std(cov_all_f):.4f}, {np.mean(ineff_all_f):.4f}+/-{np.std(ineff_all_f):.4f}")
            elif CP_or_CQR == 'CP':
                if Conditional == 'True':
                    cov_all_basegae_o, eff_all_basegae, cov_all_basegae, pred_set_all_basegae, val_labels_all_basegae, idx_all_basegae = test_gae_basegae(best_model, x, edge_index_train, edge_index_calib_test, edge_weight_calib_test, return_prediction_sets=True, sigmoid=use_sigmoid, conditional=True)
                else:
                    cov_all_basegae, eff_all_basegae, pred_set_all_basegae, val_labels_all_basegae, idx_all_basegae = test_gae_basegae(best_model, x, edge_index_train, edge_index_calib_test, edge_weight_calib_test, return_prediction_sets=True, sigmoid=use_sigmoid, conditional=False)
                #print(f"{np.mean(cov_all_basegae):.4f}+/-{np.std(cov_all_basegae):.4f}, {np.mean(eff_all_basegae):.4f}+/-{np.std(eff_all_basegae):.4f}")
                cov_all_f.extend(cov_all_basegae)
                ineff_all_f.extend(eff_all_basegae)
                print(f"{np.mean(cov_all_f):.4f}+/-{np.std(cov_all_f):.4f}, {np.mean(ineff_all_f):.4f}+/-{np.std(ineff_all_f):.4f}")
        elif DiGAE_or_GAE_or_LGNN == 'LGNN':
            if CP_or_CQR != 'CP':
                if Conditional =='True':
                    cov_all_linegraph_o, eff_all_linegraph, cov_all_linegraph, pred_set_all_linegraph, val_labels_all_linegraph, idx_all_linegraph = test_linegraph(best_model, x=data.x, calib_test_edge_index=edge_index_calib_test, return_prediction_sets=True, score=SCORE, conditional=True)
                else:
                    cov_all_linegraph, eff_all_linegraph, pred_set_all_linegraph, val_labels_all_linegraph, idx_all_linegraph = test_linegraph(best_model, x=data.x, calib_test_edge_index=edge_index_calib_test, return_prediction_sets=True, score=SCORE, conditional=False)
                cov_all_f.extend(cov_all_linegraph)
                ineff_all_f.extend(eff_all_linegraph)
                #print(f"{np.mean(cov_all_linegraph):.4f}+/-{np.std(cov_all_linegraph):.4f}, {np.mean(eff_all_linegraph):.4f}+/-{np.std(eff_all_linegraph):.4f}")
                print(f"{np.mean(cov_all_f):.4f}+/-{np.std(cov_all_f):.4f}, {np.mean(ineff_all_f):.4f}+/-{np.std(ineff_all_f):.4f}")
            elif CP_or_CQR == 'CP':
                if Conditional == 'True':
                    cov_all_baselinegraph_o, eff_all_baselinegraph, cov_all_baselinegraph, pred_set_all_baselinegraph, val_labels_all_baselinegraph, idx_all_baselinegraph = test_baselinegraph(best_model, x=data.x, calib_test_edge_index=edge_index_calib_test, return_prediction_sets=True, conditional=True)
                else:
                    cov_all_baselinegraph, eff_all_baselinegraph, pred_set_all_baselinegraph, val_labels_all_baselinegraph, idx_all_baselinegraph = test_baselinegraph(best_model, x=data.x, calib_test_edge_index=edge_index_calib_test, return_prediction_sets=True, conditional=False)
                #print(f"{np.mean(cov_all_baselinegraph):.4f}+/-{np.std(cov_all_baselinegraph):.4f}, {np.mean(eff_all_baselinegraph):.4f}+/-{np.std(eff_all_baselinegraph):.4f}")
                cov_all_f.extend(cov_all_baselinegraph)
                ineff_all_f.extend(eff_all_baselinegraph)
                print(f"{np.mean(cov_all_f):.4f}+/-{np.std(cov_all_f):.4f}, {np.mean(ineff_all_f):.4f}+/-{np.std(ineff_all_f):.4f}")


parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--SCORE', type=str, default='cqr', help='SCORE option: qr, cqr, cqr_new')
parser.add_argument('--GNNCONV', type=str, default='GraphConv', help='GNNCONV option: GraphConv, SAGEConv, GCNConv, GATConv ')
parser.add_argument('--CP_or_CQR', type=str, help='CP_or_CQR option: CP CQR')
parser.add_argument('--Dataset', type=str, help='Description of Dataset option: An, Chicago')
parser.add_argument('--DiGAE_or_GAE_or_LGNN', type=str, help='DiGAE_or_GAE_or_LGNN option: Digae, GAE, LGNN')
parser.add_argument('--SEEDNUM', type=int, help='Loop number!: Recommand: 10/20')
parser.add_argument('--Weighted', type=str, help='Containing weighted edge or not: False, True')
parser.add_argument('--Conditional', type=str, help='Conditioal model or not-Using the worest cov result to do the calculation: True, False')

def main():
    # 解析命令行参数
    args = parser.parse_args()
    # 获取可选项的值
    SCORE = args.SCORE
    print(SCORE)
    GNNCONV = args.GNNCONV
    #print(GNNCONV)
    # 获取其他可选项的值
    CP_or_CQR = args.CP_or_CQR
    print(CP_or_CQR)
    Dataset = args.Dataset 
    DiGAE_or_GAE_or_LGNN = args.DiGAE_or_GAE_or_LGNN
    print(DiGAE_or_GAE_or_LGNN)
    SEEDNUM = args.SEEDNUM
    Weighted = args.Weighted
    Conditional = args.Conditional

    # 执行其他函数
    function_model(DiGAE_or_GAE_or_LGNN, CP_or_CQR, SCORE, SEEDNUM, GNNCONV, Weighted, Conditional, Dataset)
    #function2()

if __name__ == '__main__':
    main()



