from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import random
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from scipy.sparse import coo_matrix
import pandas as pd
import sklearn

EPS = 1e-10
MAX_LOGSTD = 10


def spatial_weight(arr, alpha, mu, sigma):
    results = alpha * np.exp(-(arr - mu) ** 2 / 2 / (sigma ** 2))
    return results


def edge_index_to_vector(
        edge_index: Tensor,
        size: Tuple[int, int],
        bipartite: bool,
        force_undirected: bool = False,
) -> Tuple[Tensor, int]:
    row, col = edge_index

    if bipartite:  # No need to account for self-loops.
        idx = (row * size[1]).add_(col)
        population = size[0] * size[1]
        return idx, population

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        # We only operate on the upper triangular matrix:
        mask = row < col
        row, col = row[mask], col[mask]
        offset = torch.arange(1, num_nodes, device=row.device).cumsum(0)[row]
        idx = row.mul_(num_nodes).add_(col).sub_(offset)
        population = (num_nodes * (num_nodes + 1)) // 2 - num_nodes
        return idx, population

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        # We remove self-loops as we do not want to take them into account
        # when sampling negative values.
        mask = row != col
        row, col = row[mask], col[mask]
        col[row < col] -= 1
        idx = row.mul_(num_nodes - 1).add_(col)
        population = num_nodes * num_nodes - num_nodes
        return idx, population


def vector_to_edge_index(idx: Tensor, size: Tuple[int, int], bipartite: bool,
                         force_undirected: bool = False) -> Tensor:
    if bipartite:  # No need to account for self-loops.
        row = idx.div(size[1], rounding_mode='floor')
        col = idx % size[1]
        return torch.stack([row, col], dim=0)

    elif force_undirected:
        assert size[0] == size[1]
        num_nodes = size[0]

        offset = torch.arange(1, num_nodes, device=idx.device).cumsum(0)
        end = torch.arange(num_nodes, num_nodes * num_nodes, num_nodes,
                           device=idx.device)
        row = torch.bucketize(idx, end.sub_(offset), right=True)
        col = offset[row].add_(idx) % num_nodes
        return torch.stack([torch.cat([row, col]), torch.cat([col, row])], 0)

    else:
        assert size[0] == size[1]
        num_nodes = size[0]

        row = idx.div(num_nodes - 1, rounding_mode='floor')
        col = idx % (num_nodes - 1)
        col[row <= col] += 1
        return torch.stack([row, col], dim=0)


def sample(population: int, k: int, device=None) -> Tensor:
    if population <= k:
        return torch.arange(population, device=device)
    else:
        return torch.tensor(random.sample(range(population), k), device=device)


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))


def negative_sampling(edge_index: Tensor,
                      num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
                      num_neg_samples: Optional[int] = None,
                      method: str = "sparse",
                      force_undirected: bool = False) -> Tensor:
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return.
            If set to :obj:`None`, will try to return a negative edge for every
            positive edge. (default: :obj:`None`)
        method (string, optional): The method to use for negative sampling,
            *i.e.*, :obj:`"sparse"` or :obj:`"dense"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, while
            :obj:`"dense"` can perform faster true-negative checks.
            (default: :obj:`"sparse"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor
    """
    assert method in ['sparse', 'dense']

    size = num_nodes
    bipartite = isinstance(size, (tuple, list))
    size = maybe_num_nodes(edge_index) if size is None else size
    size = (size, size) if not bipartite else size
    force_undirected = False if bipartite else force_undirected

    idx, population = edge_index_to_vector(edge_index, size, bipartite,
                                           force_undirected)

    if idx.numel() >= population:
        return edge_index.new_empty((2, 0))

    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)
    if force_undirected:
        num_neg_samples = num_neg_samples // 2

    prob = 1. - idx.numel() / population  # Probability to sample a negative.
    sample_size = int(1.1 * num_neg_samples / prob)  # (Over)-sample size.

    neg_idx = None
    if method == 'dense':
        # The dense version creates a mask of shape `population` to check for
        # invalid samples.
        mask = idx.new_ones(population, dtype=torch.bool)
        mask[idx] = False
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, idx.device)
            rnd = rnd[mask[rnd]]  # Filter true negatives.
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break
            mask[neg_idx] = False

    else:  # 'sparse'
        # The sparse version checks for invalid samples via `np.isin`.
        idx = idx.to('cpu')
        for _ in range(3):  # Number of tries to sample negative indices.
            rnd = sample(population, sample_size, device='cpu')
            mask = np.isin(rnd, idx)
            if neg_idx is not None:
                mask |= np.isin(rnd, neg_idx.to('cpu'))
            mask = torch.from_numpy(mask).to(torch.bool)
            rnd = rnd[~mask].to(edge_index.device)
            neg_idx = rnd if neg_idx is None else torch.cat([neg_idx, rnd])
            if neg_idx.numel() >= num_neg_samples:
                neg_idx = neg_idx[:num_neg_samples]
                break

    edge_index = vector_to_edge_index(neg_idx, size, bipartite, force_undirected).cpu()
    sparse_mx = coo_matrix((np.ones(num_neg_samples), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))

    return sparse_mx_to_torch_sparse_tensor(sparse_mx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def graph_construction_spatial(adata, n_neighbors=3):
    print("==> ST dataset Cell numbers: %d" % adata.shape[0])
    if 'spatial' not in adata.obsm.keys():
        raise ValueError("Position information can not found in input data"
                         "Please add position as: AnnData.obsm['spatial'] = position")
    cell_position = adata.obsm['spatial']
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = {}
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    # ensure symmtric
    n_spot = adj['x'].max() + 1
    adj = coo_matrix((adj['value'], (adj['x'], adj['y'])), shape=(n_spot, n_spot))
    adj = (adj + adj.T).toarray()
    adj = np.where(adj > 1, 1, adj)
    # graph preprocess
    adj = coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_ = sparse_mx_to_torch_sparse_tensor(adj_)
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)

    return adj_normalized, adj_


from torch_geometric.nn import GCNConv


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


def cal_spatial_net(adata,
                    use_rep: str = "spatial",
                    cutoff: float = 1.,
                    KNN_neighbors: int = 3,
                    model: str = 'both',
                    verbose: bool = True,
                    ):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    use_rep
        Use the indicated representation in .obsm to construct the neighbor networks.
    cutoff
        radius cutoff when model='Radius' and the number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose
        distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.
    verbose
        Whether to output log information.
    copy
        If an :class:`~anndata.AnnData` is passed, determines whether a copy
        is returned.
    dropEdege
        Randomly drop percent% edges.
    cluster_key : str or None
        Column in adata.obs indicating cell class/type. If provided, keep only edges within the same class.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN', 'both'])
    if verbose:
        print(f"- INFO - Calculating '{use_rep}' graph.")
    # coor.columns = ['imagerow', 'imagecol']

    if model != 'both':
        if model == 'Radius':
            coor = pd.DataFrame(adata.obsm['spatial'])
            coor.index = adata.obs.index
            nbrs = sklearn.neighbors.NearestNeighbors(radius=cutoff).fit(coor)
            distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
            KNN_list = []
            for it in range(indices.shape[0]):
                # KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))
                KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], np.ones(feat_indices.shape[1]))))

        if model == 'KNN':
            coor = pd.DataFrame(adata.obsm[use_rep])
            coor.index = adata.obs.index
            nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=KNN_neighbors + 1).fit(coor)
            distances, indices = nbrs.kneighbors(coor)
            KNN_list = []
            for it in range(indices.shape[0]):
                # KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
                KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], np.ones(feat_indices.shape[1]))))

        KNN_df = pd.concat(KNN_list)
        KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

        adj = coo_matrix((KNN_df['Distance'].to_numpy(), (KNN_df['Cell1'].to_numpy(), KNN_df['Cell2'].to_numpy())))
        # graph preprocess
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        print("Mean degree: {}".format(rowsum.mean()))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        adj_ = sparse_mx_to_torch_sparse_tensor(adj_)
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)

    else:
        assert ('spatial' in adata.obsm.keys())
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index
        feat = pd.DataFrame(adata.obsm[use_rep])
        feat.index = adata.obs.index
        spatial_nbrs = sklearn.neighbors.NearestNeighbors(radius=cutoff).fit(coor)
        spatial_distances, spatial_indices = spatial_nbrs.radius_neighbors(coor, return_distance=True)
        feat_nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=KNN_neighbors + 1).fit(feat)
        feat_distances, feat_indices = feat_nbrs.kneighbors(feat)
        KNN_list = []
        for it in range(spatial_indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * spatial_indices[it].shape[0], spatial_indices[it], spatial_distances[it])))
        for it in range(feat_indices.shape[0]):
            # KNN_list.append(pd.DataFrame(zip([it] * feat_indices.shape[1], feat_indices[it, :], feat_distances[it, :])))
            KNN_list.append(pd.DataFrame(zip([it] * feat_indices.shape[1], feat_indices[it, :], np.ones(feat_indices.shape[1]))))

        KNN_df = pd.concat(KNN_list)
        KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
        KNN_df.drop_duplicates(inplace=True)

        adj = coo_matrix((KNN_df['Distance'].to_numpy(), (KNN_df['Cell1'].to_numpy(), KNN_df['Cell2'].to_numpy())))
        # graph preprocess
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = np.array(adj_.sum(1))
        print("Mean degree: {}".format(rowsum.mean()))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        adj_ = sparse_mx_to_torch_sparse_tensor(adj_)
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)

    return adj_normalized, adj_


if __name__ == '__main__':
    import scanpy as sc

    data = sc.read_h5ad(r'D:\SpatialMO\data\five_modalities\slice4\rna-pp2000.h5ad')
    # graph = graph_construction_spatial(data)
    graph = cal_spatial_net(data, use_rep='spatial', cutoff=1)
