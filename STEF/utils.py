import scipy.sparse
import torch
from anndata import AnnData
from typing import Any, Mapping, Optional, TypeVar, Union, Iterable
import networkx as nx
import scanpy as sc
import pandas as pd
import scipy
import numpy as np
import anndata
import sklearn.preprocessing
from typing import Optional

RandomState = Optional[Union[np.random.RandomState, int]]


def get_default_numpy_dtype() -> type:
    r"""
    Get numpy dtype matching that of the pytorch default dtype

    Returns
    -------
    dtype
        Default numpy dtype
    """
    return getattr(np, str(torch.get_default_dtype()).replace("torch.", ""))


def vertex_degrees(
        eidx: np.ndarray, ewt: np.ndarray,
        vnum: Optional[int] = None, direction: str = "both"
) -> np.ndarray:
    r"""
    Compute vertex degrees

    Parameters
    ----------
    eidx
        Vertex indices of edges (:math:`2 \times n_{edges}`)
    ewt
        Weight of edges (:math:`n_{edges}`)
    vnum
        Total number of vertices (determined by max edge index if not specified)
    direction
        Direction of vertex degree, should be one of {"in", "out", "both"}

    Returns
    -------
    degrees
        Vertex degrees
    """
    vnum = vnum or eidx.max() + 1
    adj = scipy.sparse.coo_matrix((ewt, (eidx[0], eidx[1])), shape=(vnum, vnum))
    if direction == "in":
        return adj.sum(axis=0).A1
    elif direction == "out":
        return adj.sum(axis=1).A1
    elif direction == "both":
        return adj.sum(axis=0).A1 + adj.sum(axis=1).A1 - adj.diagonal()
    raise ValueError("Unrecognized direction!")


def normalize_edges(
        eidx: np.ndarray, ewt: np.ndarray, method: str = "keepvar"
) -> np.ndarray:
    r"""
    Normalize graph edge weights

    Parameters
    ----------
    eidx
        Vertex indices of edges (:math:`2 \times n_{edges}`)
    ewt
        Weight of edges (:math:`n_{edges}`)
    method
        Normalization method, should be one of {"in", "out", "sym", "keepvar"}

    Returns
    -------
    enorm
        Normalized weight of edges (:math:`n_{edges}`)
    """
    if method not in ("in", "out", "sym", "keepvar"):
        raise ValueError("Unrecognized method!")
    enorm = ewt
    if method in ("in", "keepvar", "sym"):
        in_degrees = vertex_degrees(eidx, ewt, direction="in")
        in_normalizer = np.power(
            in_degrees[eidx[1]],
            -1 if method == "in" else -0.5
        )
        in_normalizer[~np.isfinite(in_normalizer)] = 0  # In case there are unconnected vertices
        enorm = enorm * in_normalizer
    if method in ("out", "sym"):
        out_degrees = vertex_degrees(eidx, ewt, direction="out")
        out_normalizer = np.power(
            out_degrees[eidx[0]],
            -1 if method == "out" else -0.5
        )
        out_normalizer[~np.isfinite(out_normalizer)] = 0  # In case there are unconnected vertices
        enorm = enorm * out_normalizer
    return enorm


def get_rs(x: RandomState = None) -> np.random.RandomState:
    r"""
    Get random state object

    Parameters
    ----------
    x
        Object that can be converted to a random state object

    Returns
    -------
    rs
        Random state object
    """
    if isinstance(x, int):
        return np.random.RandomState(x)
    if isinstance(x, np.random.RandomState):
        return x
    return np.random


def check_graph(
        graph: nx.Graph, adatas: Iterable[AnnData],
        cov: str = "error", attr: str = "error",
        loop: str = "error", sym: str = "error"
) -> None:
    r"""
    Check if a graph is a valid guidance graph

    Parameters
    ----------
    graph
        Graph to be checked
    adatas
        AnnData objects where graph nodes are variables
    cov
        Action to take if graph nodes does not cover all variables,
        must be one of {"ignore", "warn", "error"}
    attr
        Action to take if graph edges does not contain required attributes,
        must be one of {"ignore", "warn", "error"}
    loop
        Action to take if graph does not contain self-loops,
        must be one of {"ignore", "warn", "error"}
    sym
        Action to take if graph is not symmetric,
        must be one of {"ignore", "warn", "error"}
    """
    passed = True

    print("Checking variable coverage...")
    if not all(
            all(graph.has_node(var_name) for var_name in adata.var_names)
            for adata in adatas
    ):
        passed = False
        msg = "Some variables are not covered by the graph!"
        if cov == "error":
            raise ValueError(msg)
        elif cov == "warn":
            check_graph.logger.warning(msg)
        elif cov != "ignore":
            raise ValueError(f"Invalid `cov`: {cov}")

    print("Checking edge attributes...")
    if not all(
            "weight" in edge_attr and "sign" in edge_attr
            for edge_attr in dict(graph.edges).values()
    ):
        passed = False
        msg = "Missing weight or sign as edge attribute!"
        if attr == "error":
            raise ValueError(msg)
        elif attr == "warn":
            check_graph.logger.warning(msg)
        elif cov != "ignore":
            raise ValueError(f"Invalid `attr`: {attr}")

    print("Checking self-loops...")
    if not all(
            graph.has_edge(node, node) for node in graph.nodes
    ):
        passed = False
        msg = "Missing self-loop!"
        if loop == "error":
            raise ValueError(msg)
        elif loop == "warn":
            print(msg)
        elif loop != "ignore":
            raise ValueError(f"Invalid `loop`: {loop}")

    print("Checking graph symmetry...")
    if not all(
            graph.has_edge(e[1], e[0]) for e in graph.edges
    ):
        passed = False
        msg = "Graph is not symmetric!"
        if sym == "error":
            raise ValueError(msg)
        elif sym == "warn":
            print(msg)
        elif sym != "ignore":
            raise ValueError(f"Invalid `sym`: {sym}")

    if passed:
        print("All checks passed!")


def search_res(adata: anndata.AnnData, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res


def prod(x: Iterable) -> Any:
    r"""
    Product of elements

    Parameters
    ----------
    x
        Input elements

    Returns
    -------
    prod
        Product

    Note
    ----
    For compatibility with Python<=3.7
    """
    try:
        from math import prod  # pylint: disable=redefined-outer-name
        return prod(x)
    except ImportError:
        ans = 1
        for item in x:
            ans = ans * item
        return ans


def search_res(adata: sc.AnnData, n_clusters: int, method='leiden', use_rep='fusion', start=0.1, end=2.0,
               increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float
        The end value for searching.
    increment : float
        The step size to increase.

    Returns
    -------
    res : float
        Resolution.

    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=30, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    if not label:
        print('Resolution is not found. Please try bigger range or smaller step!')
        return 1

    return res


def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(
        adata: anndata.AnnData, n_components: int = 51,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    # X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    # adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:, 1:]