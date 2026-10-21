import torch
import torch.nn as nn
from torch import Tensor
from .distribution import ZeroInflatedNegativeBinomial, ZeroInflatedPoison
from abc import abstractmethod
from typing import Optional, Tuple, Mapping
import torch.distributions as D
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv
from .gat_conv import GATConv
EPS = 1e-7


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn

    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()



class DataEncoder(nn.Module):
    r"""
    Abstract data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        self.gcn_layer1 = GATConv(in_features, h_dim, dropout=dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.gcn_layer2 = GATConv(h_dim, h_dim, dropout=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = nn.Linear(ptr_dim, out_features)
        # init_max_weights(self)

    @abstractmethod
    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        r"""
        Compute normalizer

        Parameters
        ----------
        x
            Input data

        Returns
        -------
        l
            Normalizer
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""
        Normalize data

        Parameters
        ----------
        x
            Input data
        l
            Normalizer

        Returns
        -------
        xnorm
            Normalized data
        """
        raise NotImplementedError  # pragma: no cover

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xrep: torch.Tensor, graph: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor], torch.Tensor]:
        r"""
        Encode data to sample latent distribution

        Parameters
        ----------
        x
            Input data
        xrep
            Alternative input data
        graph
            Adjacent matrix of KNN graph
        lazy_normalizer
            Whether to skip computing `x` normalizer (just return None)
            if `xrep` is non-empty

        Returns
        -------
        u
            Sample latent distribution
        normalizer
            Data normalizer

        Note
        ----
        Normalization is always computed on `x`.
        If xrep is empty, the normalized `x` will be used as input
        to the encoder neural network, otherwise xrep is used instead.
        :param x:
        :param lazy_normalizer:
        :param xrep:
        :param graph:
        """
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)

        # gcn_ptr = self.bn1(self.gcn_layer1(ptr, graph))
        gcn_ptr = self.gcn_layer1(ptr, graph)
        gcn_ptr = F.leaky_relu(gcn_ptr)
        # gcn_ptr = self.bn2(self.gcn_layer2(gcn_ptr, graph))
        gcn_ptr = self.gcn_layer1(ptr, graph)
        gcn_ptr = F.leaky_relu(gcn_ptr)
        loc = self.loc(gcn_ptr)
        std = F.softplus(self.std_lin(gcn_ptr)) + EPS

        return D.Normal(loc, std), l


class VanillaDataEncoder(DataEncoder):
    r"""
    Vanilla data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return x


class NBDataEncoder(DataEncoder):
    r"""
    Data encoder for negative binomial data

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    TOTAL_COUNT = 1e4

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()


class DataDecoder(nn.Module):
    r"""
    Abstract data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    """

    def __init__(self, out_features: int) -> None:  # pylint: disable=unused-argument
        super().__init__()
        # self.gcn_layer = GCN(input_dim=out_features, output_dim=out_features)

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, v: torch.Tensor,
            l: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> D.Normal:
        r"""
        Decode data from sample and feature latent

        Parameters
        ----------
        u
            Sample latent
        v
            Feature latent
        b
            Batch index
        l
            Optional normalizer
        graph
            Adjacent matrix

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        raise NotImplementedError  # pragma: no cover


class NBDataDecoder(DataDecoder):
    r"""
    Negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    """

    def __init__(self, out_features: int) -> None:
        super().__init__(out_features)
        self.scale_lin = torch.nn.Parameter(torch.ones(out_features))
        self.bias = torch.nn.Parameter(torch.ones(out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            l: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> D.NegativeBinomial:
        scale = nn.functional.softplus(self.scale_lin)
        logit_mu = scale * (u @ v.t()) + self.bias
        mu = nn.functional.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta
        if mask is not None:
            return D.NegativeBinomial(
                log_theta.exp()[mask, :],
                logits=((mu + EPS).log() - log_theta)[mask, :]
            )
        else:
            return D.NegativeBinomial(
                log_theta.exp(),
                logits=(mu + EPS).log() - log_theta
            )


class ZINBDataDecoder(NBDataDecoder):
    r"""
    Zero-inflated negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    """

    def __init__(self, out_features: int) -> None:
        super().__init__(out_features)
        self.zi_logits = torch.nn.Parameter(torch.ones(out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            l: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ):
        scale = torch.nn.functional.softplus(self.scale_lin)
        feat = u @ v.t()
        logit_mu = scale * feat + self.bias
        mu = torch.nn.functional.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta.expand_as(mu)
        if mask is not None:
            return ZeroInflatedNegativeBinomial(zi_logits=self.zi_logits.expand_as(mu)[mask, :],
                                                total_count=log_theta.exp()[mask, :],
                                                logits=((mu + EPS).log() - log_theta)[mask, :]
                                                )
        else:
            return ZeroInflatedNegativeBinomial(zi_logits=self.zi_logits.expand_as(mu),
                                                total_count=log_theta.exp(),
                                                logits=(mu + EPS).log() - log_theta
                                                )


class ZIPDataDecoder(DataDecoder):
    r"""
    Zero-inflated poison data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    """

    def __init__(self, out_features: int) -> None:
        super(ZIPDataDecoder, self).__init__(out_features)
        self.zi_logits = torch.nn.Parameter(torch.ones(out_features))
        self.scale_lin = torch.nn.Parameter(torch.ones(out_features))
        self.bias = torch.nn.Parameter(torch.ones(out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            l: Optional[torch.Tensor], mask: Optional[torch.Tensor] = None
    ):
        scale = torch.nn.functional.softplus(self.scale_lin)
        feat = u @ v.t()
        logit_mu = scale * feat + self.bias
        mu = torch.nn.functional.softmax(logit_mu, dim=1) * l
        if mask is not None:
            return ZeroInflatedPoison(zi_logits=self.zi_logits.expand_as(mu)[mask, :],
                                      mu=mu[mask, :],
                                      )
        else:
            return ZeroInflatedPoison(zi_logits=self.zi_logits.expand_as(mu),
                                      mu=mu,
                                      )


class VanillaDataDecoder(nn.Module):
    r"""
    Abstract data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    """

    def __init__(self, input_features: int, out_features: int, hidden_dim=256) -> None:  # pylint: disable=unused-argument
        super().__init__()
        self.layer = GATConv(input_features, hidden_dim)
        # self.layer = nn.Linear(input_features, input_features)
        self.bn = nn.BatchNorm1d(input_features)
        self.dropout = nn.Dropout(p=0.2)
        self.logit_mu = nn.Linear(hidden_dim, out_features)

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor,
            l: Optional[torch.Tensor],
            graph,
            mask: Optional[torch.Tensor]
    ) -> D.Normal:
        r"""
        Decode data from sample and feature latent

        Parameters
        ----------
        u
            Sample latent
        l
            Optional normalizer
        graph
            Adjacent matrix

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        raise NotImplementedError  # pragma: no cover


class VanillaZINBDecoder(VanillaDataDecoder):
    def __init__(self, input_features, out_features):
        super(VanillaZINBDecoder, self).__init__(input_features, out_features)
        self.log_theta = nn.Parameter(torch.zeros(out_features))
        self.zi_logits = nn.Parameter(torch.zeros(out_features))

    def forward(
            self, u: torch.Tensor,
            l: Optional[torch.Tensor],
            graph,
            mask: Optional[torch.Tensor] = None
    ):
        u = F.leaky_relu(self.layer(u, graph))
        mu = torch.nn.functional.softmax(self.logit_mu(u), dim=1) * l
        log_theta = self.log_theta.expand_as(mu)
        zi_logits = self.zi_logits.expand_as(mu)
        if mask is not None:
            return ZeroInflatedNegativeBinomial(zi_logits=zi_logits[mask,:],
                                                total_count=log_theta.exp()[mask,:],
                                                logits=((mu + EPS).log() - log_theta)[mask,:]
                                                )
        else:
            return ZeroInflatedNegativeBinomial(zi_logits=zi_logits,
                                                total_count=log_theta.exp(),
                                                logits=((mu + EPS).log() - log_theta)
                                                )


class VanillaZIPDecoder(VanillaDataDecoder):
    def __init__(self, input_features, out_features):
        super(VanillaZIPDecoder, self).__init__(input_features, out_features)
        self.zi_logits = nn.Parameter(torch.zeros(out_features))
        init_max_weights(self)

    def forward(
            self, u: torch.Tensor,
            l: Optional[torch.Tensor],
            graph,
            mask: Optional[torch.Tensor] = None
    ):
        u = F.leaky_relu(self.layer(u, graph))
       #  u = F.leaky_relu(self.layer(u))
        mu = torch.nn.functional.softmax(self.logit_mu(u), dim=1) * l
        zi_logits = self.zi_logits.expand_as(mu)
        if mask is not None:
            return ZeroInflatedPoison(zi_logits=zi_logits[mask, :],
                                      mu=mu[mask, :],
                                      )
        else:
            return ZeroInflatedPoison(zi_logits=zi_logits,
                                      mu=mu,
                                      )


class Prior(nn.Module):
    r"""
    Prior distribution

    Parameters
    ----------
    loc
        Mean of the normal distribution
    std
        Standard deviation of the normal distribution
    """

    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)


class Cross_Module(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super(Cross_Module, self).__init__()
        if not hidden_dim:
            hidden_dim = int(input_dim / 2)
        self.layer = nn.Sequential(*[nn.Linear(input_dim, hidden_dim),
                                     nn.LeakyReLU(),
                                     nn.BatchNorm1d(hidden_dim),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(hidden_dim, int(hidden_dim / 2)),
                                     nn.LeakyReLU(),
                                     nn.BatchNorm1d(int(hidden_dim / 2)),
                                     nn.Dropout(p=0.1),
                                     nn.Linear(int(hidden_dim / 2), output_dim),
                                     ])
        init_max_weights(self)

    def forward(self, input):
        return self.layer(input)


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))

        for param in self.parameters():
            if len(param.shape) == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="linear")

    def forward(
            self, input: torch.Tensor, eidx: torch.Tensor,
            enorm: torch.Tensor, esgn: torch.Tensor
    ) -> torch.Tensor:
        sidx, tidx = eidx  # source index and target index

        message = input[sidx] * (esgn * enorm).unsqueeze(1)  # n_edges * n_features
        res = torch.zeros_like(input)
        tidx = tidx.unsqueeze(1).expand_as(message)  # n_edges * n_features
        res.scatter_add_(0, tidx, message)
        res = torch.mm(res, self.weight) + self.bias

        return res


class GraphConv(torch.nn.Module):
    r"""
    Graph convolution (propagation only)
    """

    def forward(
            self, input: torch.Tensor, eidx: torch.Tensor,
            enorm: torch.Tensor, esgn: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Forward propagation

        Parameters
        ----------
        input
            Input data (:math:`n_{vertices} \times n_{features}`)
        eidx
            Vertex indices of edges (:math:`2 \times n_{edges}`)
        enorm
            Normalized weight of edges (:math:`n_{edges}`)
        esgn
            Sign of edges (:math:`n_{edges}`)

        Returns
        -------
        result
            Graph convolution result (:math:`n_{vertices} \times n_{features}`)
        """
        sidx, tidx = eidx  # source index and target index
        message = input[sidx] * (esgn * enorm).unsqueeze(1)  # n_edges * n_features
        res = torch.zeros_like(input)
        tidx = tidx.unsqueeze(1).expand_as(message)  # n_edges * n_features
        res.scatter_add_(0, tidx, message)
        return res


class GraphEncoder(nn.Module):
    r"""
    Graph encoder

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    """

    def __init__(
            self, vnum: int, out_features: int
    ) -> None:
        super().__init__()
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        self.conv = GraphConv()
        # self.act = nn.PReLU(out_features)
        self.act = nn.LeakyReLU()
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        ptr = self.act(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)


class GraphDecoder(nn.Module):
    r"""
    Graph decoder
    """

    def forward(
            self, v: torch.Tensor, eidx: torch.Tensor, esgn: torch.Tensor
    ) -> D.Bernoulli:
        sidx, tidx = eidx  # Source index and target index
        output = esgn * (v[sidx] * v[tidx]).sum(dim=1)

        return D.Bernoulli(logits=output)


class AttentionLayer(nn.Module):
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.

    Returns
    -------
    Aggregated representations and modality weights.

    """

    def __init__(self, in_feat, out_feat):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = nn.Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, input_list, return_attn=False):
        emb = []
        for i in input_list:
            emb.append(torch.unsqueeze(i, dim=1))
        emb = torch.cat(emb, dim=1)

        # self.v = torch.tanh(torch.matmul(emb, self.w_omega))
        self.v = torch.tanh(emb @ self.w_omega)
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6, dim=-1)

        emb_combined = torch.matmul(torch.transpose(emb, 1, 2), torch.unsqueeze(self.alpha, -1))
        if return_attn:
            return torch.squeeze(emb_combined), self.alpha
        else:
            return torch.squeeze(emb_combined)


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def forward(self, z: Tensor, graph: torch.sparse_coo,
                sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """

        value = (z[graph._indices()[0]] * z[graph._indices()[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class FusionProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super(FusionProjectionLayer, self).__init__()
        if not hidden_dim:
            hidden_dim = int(input_dim / 2)
        # self.layer = nn.Sequential(*[nn.Linear(2*input_dim, hidden_dim),
        #                              nn.LeakyReLU(),
        #                              nn.BatchNorm1d(hidden_dim),
        #                              nn.Dropout(p=0.1),
        #                              # nn.Linear(hidden_dim, int(hidden_dim / 2)),
        #                              # nn.LeakyReLU(),
        #                              # nn.BatchNorm1d(int(hidden_dim / 2)),
        #                              # nn.Dropout(p=0.1),
        #                              ])
        self.layer = AttentionLayer(in_feat=hidden_dim, out_feat=hidden_dim)
        self.loc = torch.nn.Linear(hidden_dim, output_dim)
        self.std_lin = nn.Linear(hidden_dim, output_dim)
        init_max_weights(self)

    def forward(self, z1, z2):
        # z = self.layer(torch.concat([z1, z2], dim=-1))
        z = self.layer([z1, z2])
        return z


class Classifier(nn.Module):

    r"""
    Modality classifier

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int = 2,
            h_depth: int = 2, h_dim: Optional[int] = 32,
            dropout: float = 0.2
    ) -> None:
        super(Classifier, self).__init__()
        cls = [nn.Linear(in_features, h_dim), nn.ReLU(), nn.Dropout(dropout)]
        for i in range(1, h_depth):
            if i != h_depth:
                cls = cls + [nn.Linear(h_dim, out_features)]
            else:
                cls = cls + [nn.Linear(h_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)]
        self.cls = nn.Sequential(*cls)

    def forward(self, input):
        pred = self.cls(input)

        return pred