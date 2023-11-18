from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling

EPS = 1e-15
MAX_LOGSTD = 10


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z: Tensor, edge_index: Tensor,
                sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class DirectedInnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z1: Tensor, z2: Tensor, edge_index: Tensor,
                sigmoid: bool = False) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z1[edge_index[0]] * z2[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z1: Tensor, z2: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z1, z2.t())
        return torch.sigmoid(adj) if sigmoid else adj


class EdgeDecoder(torch.nn.Module):
     """
     Edge Decoder module to infer the predictions. 

     Args:
     hidden_channels (int): The number of hidden channels.
     num_heads_GAT (int): The number of attention heads in GAT layer.
     """

     def __init__(self, hidden_channels=8, out_channels=1):

         super().__init__()

         self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
         self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

     def forward(self, z, edge_index, sigmoid=False):
         """
         Forward pass of the EdgeDecoder module.

         Args:
         z_dict (dict): node type as keys and temporal node embeddings 
         for each node as values. 
         edge_label_index (torch.Tensor): see previous section.

         Returns:
         torch.Tensor: Predicted edge labels.
         """
         row, col = edge_index

         z = torch.cat([z[row], z[col]], dim=-1)
         z = self.lin1(z)
         z = z.relu() # torch.nn.functional.leaky_relu(z, negative_slope=0.1)
         z = self.lin2(z)
         return torch.sigmoid(z) if sigmoid else z
 

class DirectedEdgeDecoder(torch.nn.Module):
     """
     Edge Decoder module to infer the predictions. 

     Args:
     hidden_channels (int): The number of hidden channels.
     num_heads_GAT (int): The number of attention heads in GAT layer.
     """

     def __init__(self, in_channels, hidden_channels=8, out_channels=1):

         super().__init__()

         self.lin1 = torch.nn.Linear(2 * in_channels, hidden_channels)
         self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

     def forward(self, z1, z2, edge_index, sigmoid=False):
         """
         Forward pass of the EdgeDecoder module.

         Args:
         z_dict (dict): node type as keys and temporal node embeddings 
         for each node as values. 
         edge_label_index (torch.Tensor): see previous section.

         Returns:
         torch.Tensor: Predicted edge labels.
         """
         row, col = edge_index

         z = torch.cat([z1[row], z2[col]], dim=-1)
         z = self.lin1(z)
         z = z.relu() # torch.nn.functional.leaky_relu(z, negative_slope=0.1)
         z = self.lin2(z)
         return torch.sigmoid(z) if sigmoid else z
 

class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = EdgeDecoder() if decoder is None else decoder
        # self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor, pos_edge_weight: Tensor,
                   neg_edge_index: Optional[Tensor] = None, return_all = False) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        edge_weight_pred = self.decoder(z, pos_edge_index, sigmoid=False)
        if return_all: 
          return (edge_weight_pred - pos_edge_weight) ** 2
        return torch.nn.functional.mse_loss(edge_weight_pred, pos_edge_weight)
        
        # pos_loss = -torch.log(
            # self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # if neg_edge_index is None:
            # neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        # neg_loss = -torch.log(1 -
                              # self.decoder(z, neg_edge_index, sigmoid=True) +
                              # EPS).mean()

        # return pos_loss + neg_loss


class VGAE(torch.nn.Module):
    def __init__(self, encoder: Module, decoder: Optional[Module] = None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        VGAE.reset_parameters(self)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def forward(self, *args, **kwargs) -> Tensor:
        """"""
        self.mu1, self.mu2, self.mu3, self.logstd = self.encoder(*args, **kwargs)
        self.logstd = self.logstd.clamp(max=10)
        z1 = self.reparametrize(self.mu1, self.logstd)
        z2 = self.reparametrize(self.mu2, self.logstd)
        z3 = self.reparametrize(self.mu3, self.logstd)
        return torch.cat([z1, z2, z3], dim=-1)

    def kl_loss(self, mu: Optional[Tensor] = None,
                logstd: Optional[Tensor] = None) -> Tensor:
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (torch.Tensor, optional): The latent space for :math:`\mu`. If
                set to :obj:`None`, uses the last computation of :math:`\mu`.
                (default: :obj:`None`)
            logstd (torch.Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`. (default: :obj:`None`)
        """
        mu1, mu2, mu3, logstd = self.mu1, self.mu2, self.mu3, self.logstd
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu1**2 - logstd.exp()**2, dim=1))\
            -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu2**2 - logstd.exp()**2, dim=1))\
            -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu3**2 - logstd.exp()**2, dim=1))
