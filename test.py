import argparse
import autoencoder
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



