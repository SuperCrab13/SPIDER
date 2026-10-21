import os
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Set, Union, List
import tqdm
from networkx.algorithms.bipartite import biadjacency_matrix
import networkx as nx
import networkx as nx
import numpy as np
import pandas as pd
from itertools import chain
import pybedtools
from pybedtools import BedTool
from pybedtools.cbedtools import Interval
from scipy.stats import pearsonr
import scanpy as sc


class ConstrainedDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs) -> None:
        df = pd.DataFrame(*args, **kwargs)
        df = self.rectify(df)
        self.verify(df)
        super().__init__(df)

    @property
    def _constructor(self) -> type:
        return type(self)

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        pass

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self)

    def __repr__(self) -> str:
        return repr(self.df)


class Bed(ConstrainedDataFrame):
    COLUMNS = pd.Index(["chrom", "chromStart", "chromEnd", "name", "score", "strand"])

    @classmethod
    def rectify(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = super(Bed, cls).rectify(df)
        COLUMNS = cls.COLUMNS.copy(deep=True)
        for item in COLUMNS:
            if item in df:
                if item in ("chromStart", "chromEnd"):
                    df[item] = df[item].astype(int)
                else:
                    df[item] = df[item].astype(str)
            elif item not in ("chrom", "chromStart", "chromEnd"):
                df[item] = "."
            else:
                raise ValueError(f"Required column {item} is missing!")
        return df.loc[:, COLUMNS]

    @classmethod
    def verify(cls, df: pd.DataFrame) -> None:
        super(Bed, cls).verify(df)
        if len(df.columns) != len(cls.COLUMNS) or np.any(df.columns != cls.COLUMNS):
            raise ValueError("Invalid BED format!")

    @classmethod
    def read_bed(cls, fname: os.PathLike) -> "Bed":
        COLUMNS = cls.COLUMNS.copy(deep=True)
        loaded = pd.read_csv(fname, sep="\t", header=None, comment="#")
        loaded.columns = COLUMNS[:loaded.shape[1]]
        return cls(loaded)

    def to_bedtool(self) -> pybedtools.BedTool:
        return BedTool(Interval(row["chrom"], row["chromStart"], row["chromEnd"],
                                name=row["name"], score=row["score"], strand=row["strand"])
                       for _, row in self.iterrows())

    def strand_specific_start_site(self) -> "Bed":
        if set(self["strand"]) != set(["+", "-"]):
            raise ValueError("Not all features are strand specific!")
        df = pd.DataFrame(self, copy=True)
        pos_strand = df.query("strand == '+'").index
        neg_strand = df.query("strand == '-'").index
        df.loc[pos_strand, "chromEnd"] = df.loc[pos_strand, "chromStart"] + 1
        df.loc[neg_strand, "chromStart"] = df.loc[neg_strand, "chromEnd"] - 1
        return type(self)(df)

    def expand(
            self, upstream: int, downstream: int,
            chr_len: Optional[Mapping[str, int]] = None
    ) -> "Bed":
        r"""
        Expand genomic features towards upstream and downstream

        Parameters
        ----------
        upstream
            Number of bps to expand in the upstream direction
        downstream
            Number of bps to expand in the downstream direction
        chr_len
            Length of each chromosome

        Returns
        -------
        expanded_bed
            A new :class:`Bed` object, containing expanded features
            of the current :class:`Bed` object

        Note
        ----
        Starting position < 0 after expansion is always trimmed.
        Ending position exceeding chromosome length is trimed only if
        ``chr_len`` is specified.
        """
        if upstream == downstream == 0:
            return self
        df = pd.DataFrame(self, copy=True)
        if upstream == downstream:  # symmetric
            df["chromStart"] -= upstream
            df["chromEnd"] += downstream
        else:  # asymmetric
            if set(df["strand"]) != set(["+", "-"]):
                raise ValueError("Not all features are strand specific!")
            pos_strand = df.query("strand == '+'").index
            neg_strand = df.query("strand == '-'").index
            if upstream:
                df.loc[pos_strand, "chromStart"] -= upstream
                df.loc[neg_strand, "chromEnd"] += upstream
            if downstream:
                df.loc[pos_strand, "chromEnd"] += downstream
                df.loc[neg_strand, "chromStart"] -= downstream
        df["chromStart"] = np.maximum(df["chromStart"], 0)
        if chr_len:
            chr_len = df["chrom"].map(chr_len)
            df["chromEnd"] = np.minimum(df["chromEnd"], chr_len)
        return type(self)(df)


def interval_dist(x: Interval, y: Interval) -> int:
    if x.chrom != y.chrom:
        return np.inf * (-1 if x.chrom < y.chrom else 1)
    if x.start < y.stop and y.start < x.stop:
        return 0
    if x.stop <= y.start:
        return x.stop - y.start - 1
    if y.stop <= x.start:
        return x.start - y.stop + 1


def window_graph(
        left: Union[Bed, str], right: Union[Bed, str], window_size: int,
        left_sorted: bool = False, right_sorted: bool = False,
        attr_fn: Optional[Callable[[Interval, Interval, float], Mapping[str, Any]]] = None
) -> nx.MultiDiGraph:
    if isinstance(left, Bed):
        left = left.to_bedtool()
    else:
        left = pybedtools.BedTool(left)
    if not left_sorted:
        left = left.sort(stream=True)
    left = iter(left)
    if isinstance(right, Bed):
        right = right.to_bedtool()
    else:
        right = pybedtools.BedTool(right)
    if not right_sorted:
        right = right.sort(stream=True)
    right = iter(right)

    attr_fn = attr_fn or (lambda l, r, d: {})
    graph = nx.MultiDiGraph()
    window = {}
    for l in left:
        for r in list(window.keys()):
            d = interval_dist(l, r)
            if -window_size <= d <= window_size:
                graph.add_edge(l.name, r.name, **attr_fn(l, r, d))
            elif d > window_size:
                del window[r]
            else:
                break
        else:
            for r in right:
                d = interval_dist(l, r)
                if -window_size <= d <= window_size:
                    graph.add_edge(l.name, r.name, **attr_fn(l, r, d))
                elif d > window_size:
                    continue
                window[r] = None
                if d < -window_size:
                    break
    pybedtools.cleanup()
    return graph


def ens_trim_version(x: str) -> str:
    return re.sub(r"\.[0-9_-]+$", "", x)


def compose_multigraph(*graphs: nx.Graph) -> nx.MultiGraph:
    r"""
    Compose multi-graph from multiple graphs with no edge collision

    Parameters
    ----------
    graphs
        An arbitrary number of graphs to be composed from

    Returns
    -------
    composed
        Composed multi-graph

    Note
    ----
    The resulting multi-graph would be directed if any of the input graphs
    is directed.
    """
    if any(nx.is_directed(graph) for graph in graphs):
        graphs = [graph.to_directed() for graph in graphs]
        composed = nx.MultiDiGraph()
    else:
        composed = nx.MultiGraph()
    composed.add_edges_from(
        (e[0], e[1], graph.edges[e]) for graph in graphs for e in graph.edges
    )
    return composed


def dist_power_decay(x: int, bandwidth: int = 1000) -> float:
    return ((x + bandwidth) / bandwidth) ** (-0.75)


def edge_weight(arr, corr, alpha=1, mu=0, sigma=50):
    d_weight = alpha * np.exp(-(arr - mu) ** 2 / 2 / (sigma ** 2))
    results = np.multiply(corr, d_weight)

    return results


def rna_anchored_guidance_graph(
    rna: sc.AnnData, *others: sc.AnnData, gtf: os.PathLike,
    gene_region: str = "combined", promoter_len: int = 2000,
    extend_range: int = 0, extend_fn: Callable[[int], float] = dist_power_decay,
    signs: Optional[List[int]] = None, gtf_by: str = 'gene_name'
) -> nx.MultiDiGraph:
    r"""
    Build guidance graph anchored on RNA genes

    Parameters
    ----------
    rna
        Anchor RNA dataset
    *others
        Other datasets
    gtf
        Path to the GTF file
    gene_region
        Defines the genomic region of genes, must be one of
        ``{"gene_body", "promoter", "combined"}``.
    promoter_len
        Defines the length of gene promoters (bp upstream of TSS)
    extend_range
        Maximal extend distance beyond gene regions
    extend_fn
        Distance-decreasing weight function for the extended regions
        (by default :func:`dist_power_decay`)
    signs
        Sign of edges between RNA genes and features in each ``*others``
        dataset, must have the same length as ``*others``. Signs must be
        one of ``{-1, 1}``. By default, all edges have positive signs of ``1``.
    propagate_highly_variable
        Whether to propagate highly variable genes to other datasets,
        datasets in ``*others`` would be modified in place.
    corrupt_rate
        **CAUTION: DO NOT USE**, only for evaluation purpose
    random_state
        **CAUTION: DO NOT USE**, only for evaluation purpose
    gtf_by

    Returns
    -------
    graph
        Prior regulatory graph

    Note
    ----
    In this function, features in the same dataset can only connect to
    anchor genes via the same edge sign. For more flexibility, please
    construct the guidance graph manually.
    """
    signs = signs or [1] * len(others)
    if len(others) != len(signs):
        raise RuntimeError("Length of ``others`` and ``signs`` must match!")
    if set(signs).difference({-1, 1}):
        raise RuntimeError("``signs`` can only contain {-1, 1}!")

    # load gtf
    COLUMNS = pd.Index(["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"])
    gtf_by = gtf_by
    loaded = pd.read_csv(gtf, sep="\t", header=None, comment="#")
    loaded.columns = COLUMNS[:loaded.shape[1]]
    loaded = loaded.query("feature == 'gene'")

    pattern = re.compile(r'([^\s]+) "([^"]+)";')
    splitted = pd.DataFrame.from_records(np.vectorize(lambda x: {
        key: val for key, val in pattern.findall(x)
    })(loaded["attribute"]), index=loaded.index)
    loaded = loaded.assign(**splitted)

    if gtf_by == 'gene_id':
        loaded[gtf_by] = [i.split('.')[0] for i in loaded[gtf_by]]
    loaded = loaded.sort_values("seqname").drop_duplicates(subset=[gtf_by], keep="last")

    bed_df = pd.DataFrame(loaded, copy=True).loc[:, ("seqname", "start", "end", "score", "strand")]
    bed_df.insert(3, "name", np.repeat(".", len(bed_df)) if gtf_by is None else loaded[gtf_by])
    bed_df["start"] -= 1
    bed_df.columns = ("chrom", "chromStart", "chromEnd", "name", "score", "strand")

    for item in bed_df.columns:
        if item in ("chromStart", "chromEnd"):
            bed_df[item] = bed_df[item].astype(int)
        else:
            bed_df[item] = bed_df[item].astype(str)

    gene_df = pd.concat([
        pd.DataFrame(bed_df),
        pd.DataFrame(loaded).drop(columns=COLUMNS)
    ], axis=1).set_index(gtf_by).reindex(rna.var_names)

    gene_df = gene_df[['chrom', 'chromStart', 'chromEnd', 'strand']]
    gene_df.dropna(inplace=True)

    rna = rna[:, gene_df.index]

    rna.var['chrom'] = gene_df.loc[rna.var_names, 'chrom']
    rna.var['chromStart'] = gene_df.loc[rna.var_names, 'chromStart']
    rna.var['chromEnd'] = gene_df.loc[rna.var_names, 'chromEnd']
    rna.var['strand'] = gene_df.loc[rna.var_names, 'strand']

    rna_bed = Bed(rna.var.assign(name=rna.var_names))
    other_beds = [Bed(other.var.assign(name=other.var_names)) for other in others]
    if gene_region == "promoter":
        rna_bed = rna_bed.strand_specific_start_site().expand(promoter_len, 0)
    elif gene_region == "combined":
        rna_bed = rna_bed.expand(promoter_len, 0)
    elif gene_region != "gene_body":
        raise ValueError("Unrecognized `gene_range`!")
    graphs = [window_graph(
        rna_bed, other_bed, window_size=extend_range,
        attr_fn=lambda l, r, d, s=sign: {
            "dist": abs(d), "weight": extend_fn(abs(d)), "sign": s
        }
    ) for other_bed, sign in zip(other_beds, signs)]
    graph = compose_multigraph(*graphs)

    rgraph = graph.reverse()
    nx.set_edge_attributes(graph, "fwd", name="type")
    nx.set_edge_attributes(rgraph, "rev", name="type")
    graph = compose_multigraph(graph, rgraph)

    all_features = set(chain.from_iterable(
        map(lambda x: set(x.var_names) & set(graph.nodes()), [rna, *others])
    ))
    for item in all_features:
        graph.add_edge(item, item, weight=1.0, sign=1, type="loop")
    return graph


read_bed = Bed.read_bed

from scipy.stats import pearsonr, spearmanr

def prune_graph_by_correlation(
    rna_adata,
    atac_adata,
    graph,
    method="pearson",
    threshold=0.2,
    use_pvalue=True,
    pvalue_threshold=0.05,
    inplace=False
):
    """
    Remove gene-peak edges with low correlation based on expression and accessibility.

    Parameters
    ----------
    rna_adata : AnnData
        RNA expression data, obs = cells, var = genes.
    atac_adata : AnnData
        ATAC peak accessibility data, obs = cells, var = peaks.
    graph : nx.Graph
        Graph containing gene and peak nodes. Edges should be between genes and peaks.
    method : str, default='pearson'
        Correlation method, either 'pearson' or 'spearman'.
    threshold : float, default=0.3
        Correlation threshold. When use_pvalue=False, keep edges with |corr| >= threshold.
    use_pvalue : bool, default=False
        If True, use p-value for filtering: keep edges with pvalue <= pvalue_threshold.
    pvalue_threshold : float, default=0.05
        P-value threshold, used only when use_pvalue=True.
    inplace : bool, default=False
        If False, return a copy of the graph; if True, modify the original graph.

    Returns
    -------
    nx.Graph
        Pruned graph.
    """
    # Copy if needed
    if not inplace:
        graph = graph.copy()

    # 1. Align cells: take intersection and order them consistently
    common_cells = rna_adata.obs_names.intersection(atac_adata.obs_names)
    if len(common_cells) == 0:
        raise ValueError("No common cells between RNA and ATAC data.")
    rna = rna_adata[common_cells, :].copy()
    atac = atac_adata[common_cells, :].copy()
    # Reorder ATAC to match RNA cell order
    atac = atac[rna.obs_names, :]

    # 2. Extract matrices and convert to dense if necessary
    X = rna.X.T  # genes x cells
    Y = atac.X.T  # peaks x cells
    if hasattr(X, "toarray"):
        X = X.toarray()
    if hasattr(Y, "toarray"):
        Y = Y.toarray()

    # Create index mappings: gene name -> row index, peak name -> row index
    gene_index = {gene: i for i, gene in enumerate(rna.var_names)}
    peak_index = {peak: i for i, peak in enumerate(atac.var_names)}

    # 3. Iterate over all gene-peak edges in the graph
    genes_set = set(gene_index.keys())
    peaks_set = set(peak_index.keys())

    edges_to_remove = []
    print("### Pruning Edges ####")
    for u, v in tqdm.tqdm(list(graph.edges()), desc="Processing edges"):
        # Determine which node is the gene and which is the peak
        if u in genes_set and v in peaks_set:
            gene, peak = u, v
        elif v in genes_set and u in peaks_set:
            gene, peak = v, u
        else:
            # Not a gene-peak edge, keep it
            continue

        # Extract expression and accessibility vectors
        gene_vec = X[gene_index[gene], :]
        peak_vec = Y[peak_index[peak], :]

        # Remove edges where either vector is constant (undefined correlation)
        if np.std(gene_vec) == 0 or np.std(peak_vec) == 0:
            edges_to_remove.append((u, v))
            continue

        # Compute correlation
        if method == "pearson":
            corr, pval = pearsonr(gene_vec, peak_vec)
        elif method == "spearman":
            corr, pval = spearmanr(gene_vec, peak_vec)
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")

        # Decide whether to remove based on threshold
        if use_pvalue:
            if pval > pvalue_threshold:
                edges_to_remove.append((u, v))
        else:
            if abs(corr) < threshold:
                edges_to_remove.append((u, v))

    # Remove low-correlation edges
    graph.remove_edges_from(edges_to_remove)
    print(f"Removed {len(edges_to_remove)} low-correlation edges. Remaining edges: {graph.number_of_edges()}")

    return graph