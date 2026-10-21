import scanpy as sc
import numpy as np
import networkx as nx
import pandas as pd
from typing import List, Tuple, Union, Optional, Dict, Iterable
from anndata import AnnData
import torch
from math import ceil
from .utils import vertex_degrees, get_default_numpy_dtype
from .graph import graph_construction_spatial, negative_sampling, cal_spatial_net
import h5py
import scipy

Array = Union[np.ndarray, scipy.sparse.spmatrix]
BackedArray = Union[h5py.Dataset]
AnyArray = Union[Array, BackedArray]


def _extract_x(adata: AnnData, use_layer, features) -> AnyArray:
    default_dtype = get_default_numpy_dtype()
    if not np.array_equal(adata.var_names, features):
        adata = adata[:, features]  # This will load all data to memory if backed
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError(
                f"Configured data layer '{use_layer}' "
                f"cannot be found in input data!"
            )
        x = adata.layers[use_layer]
    else:
        x = adata.X
    if x.dtype.type is not default_dtype:
        if isinstance(x, (h5py.Dataset)):
            raise RuntimeError(
                f"User is responsible for ensuring a {default_dtype} dtype "
                f"when using backed data!"
            )
        x = x.astype(default_dtype)
    if scipy.sparse.issparse(x):
        x = x.todense()
    return x


def _extract_xrep(adata: AnnData, use_rep, rep_dim) -> AnyArray:
    default_dtype = get_default_numpy_dtype()
    use_rep = use_rep
    rep_dim = rep_dim
    if use_rep:
        if use_rep not in adata.obsm:
            raise ValueError(
                f"Configured data representation '{use_rep}' "
                f"cannot be found in input data!"
            )
        xrep = np.asarray(adata.obsm[use_rep]).astype(default_dtype)
        if xrep.shape[1] != rep_dim:
            raise ValueError(
                f"Input representation dimensionality {xrep.shape[1]} "
                f"does not match the configured {rep_dim}!"
            )
        return xrep
    return np.empty((adata.shape[0], 0), dtype=default_dtype)


def form_graph_data(graph: nx.Graph, vertices: pd.Index,):
    default_dtype = get_default_numpy_dtype()
    i, j, w, s = [], [], [], []
    for k, v in dict(graph.edges).items():
        i.append(k[0])
        j.append(k[1])
        w.append(v["weight"])
        s.append(v["sign"])
    eidx = np.stack([
        vertices.get_indexer(i),
        vertices.get_indexer(j)
    ]).astype(np.int64)
    if eidx.min() < 0:
        raise ValueError("Missing vertices!")
    ewt = np.asarray(w).astype(default_dtype)
    esgn = np.asarray(s).astype(default_dtype)
    if set(esgn).difference({-1, 1}):
        raise ValueError("Invalid edge sign!")
    return eidx, ewt, esgn


def form_adatas(adatas: List[sc.AnnData], features: List[str], use_layers: List[str], graph_list, graph_norm_list, neg_graph,
                use_rep: List[str] = None, rep_dim: List[int] = None, cutoff: int = 1., KNN_neighbors: int = 1):

    # construct spatial graph
    graph_list = graph_list
    graph_norm_list = graph_norm_list
    neg_graph = neg_graph

    # extract x and xrep
    x, xrep = [], []
    for i in range(len(adatas)):
        xrep.append(_extract_xrep(adata=adatas[i], use_rep=use_rep[i], rep_dim=rep_dim[i]))
        x.append(_extract_x(adata=adatas[i], use_layer=use_layers[i], features=features[i]))

    return x, xrep, graph_list, graph_norm_list, neg_graph

