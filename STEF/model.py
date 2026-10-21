import os
import pandas as pd
import torch
import torch.nn as nn
from .model_utils import *
from typing import Mapping, List, Any, Optional
import numpy as np
import pathlib
import dill
from itertools import chain
import networkx as nx
from .utils import normalize_edges
from .data_utils import form_graph_data, form_adatas
import scanpy as sc
from .loss_util import *
import torch.nn.functional as F

_ENCODER_MAP: Mapping[str, type] = {}
_DECODER_MAP: Mapping[str, type] = {}
_VANILLA_DECODER_MAP: Mapping[str, type] = {}


def get_chained_attr(x: Any, attr: str) -> Any:
    r"""
    Get attribute from an object, with support for chained attribute names.

    Parameters
    ----------
    x
        Object to get attribute from
    attr
        Attribute name

    Returns
    -------
    attr_value
        Attribute value
    """
    for k in attr.split("."):
        if not hasattr(x, k):
            raise AttributeError(f"{attr} not found!")
        x = getattr(x, k)
    return x


def register_prob_model(prob_model: str, encoder: type, decoder: type, vanilla_decoder: type) -> None:
    r"""
    Register probabilistic model

    Parameters
    ----------
    prob_model
        Data probabilistic model
    encoder
        Encoder type of the probabilistic model
    decoder
        Decoder type of the shared model
    vanilla_decoder
        Decoder type of private model
    """
    _ENCODER_MAP[prob_model] = encoder
    _DECODER_MAP[prob_model] = decoder
    _VANILLA_DECODER_MAP[prob_model] = vanilla_decoder


register_prob_model("ZINB", NBDataEncoder, ZINBDataDecoder, VanillaZINBDecoder)
register_prob_model('ZIP', NBDataEncoder, ZIPDataDecoder, VanillaZIPDecoder)


class STEF(nn.Module):
    def __init__(self, modalities: Mapping[str, Mapping], vertices: pd.Index, var_names: Mapping[str, pd.Index],
                 latent_dim: int = 40,
                 dropout: float = 0.1, h_depth: int = 1):
        super(STEF, self).__init__()
        self.latent_dim = latent_dim
        self.var_names = var_names
        self.modalities = modalities
        self.vertices = vertices
        # prior knowledge module
        self.g2v = GraphEncoder(self.vertices.size, latent_dim)
        self.v2g = GraphDecoder()

        self.linked_feature_id = {}
        self.unlinked_feature_id = {}
        self.idx = {}
        for k, modality in modalities.items():
            self.linked_feature_id[k] = self.var_names[k].get_indexer(modality['linked_features']).astype(np.int64)
            self.unlinked_feature_id[k] = self.var_names[k].get_indexer(modality['unlinked_features']).astype(np.int64)
            self.idx[k] = self.vertices.get_indexer(modality['linked_features']).astype(np.int64)
        # align module
        self.linked_idx, self.sh_x2u, self.sh_u2x, self.cross_u, self.cross_u2x= {}, {}, {}, {}, {}
        for k, modality in modalities.items():
            self.linked_idx[k] = self.var_names[k].get_indexer(modality['linked_features']).astype(np.int64)
            if self.linked_idx[k].min() < 0:
                raise ValueError("Not all linked features exist in the AnnData!")
            self.linked_idx[k] = torch.as_tensor(self.linked_idx[k])

            self.sh_x2u[k] = _ENCODER_MAP[modality['prob_model']](
                modality["rep_dim"] or len(modality["features"]), latent_dim,
                h_depth=h_depth, dropout=dropout
            )
            self.sh_u2x[k] = _DECODER_MAP[modality["prob_model"]](out_features=len(modality['linked_features']))
        # specific module
        self.unlinked_idx, self.sp_x2u, self.sp_u2x = {}, {}, {}
        for k, modality in modalities.items():
            self.unlinked_idx[k] = self.var_names[k].get_indexer(modality['unlinked_features']).astype(np.int64)
            if self.unlinked_idx[k].min() < 0:
                raise ValueError("Not all unlinked features exist in the AnnData!")
            self.unlinked_idx[k] = torch.as_tensor(self.unlinked_idx[k])

            self.sp_x2u[k] = _ENCODER_MAP[modality['prob_model']](
                modality["rep_dim"] or len(modality["features"]), latent_dim,
                h_depth=h_depth, dropout=dropout)
            self.sp_u2x[k] = _VANILLA_DECODER_MAP[modality["prob_model"]](input_features=latent_dim,
                                                                          out_features=len(
                                                                          modality['unlinked_features']))

        self.graph_decoder = InnerProductDecoder()
        # Decoder with input of 
        decoder = {}
        for k, modality in modalities.items():
            # decoder[k] = _VANILLA_DECODER_MAP[modality["prob_model"]](input_features=4*latent_dim, out_features=len(modality['features']))
            decoder[k] = _VANILLA_DECODER_MAP[modality["prob_model"]](input_features=4*latent_dim, out_features=len(modality['features']))
        self.decoder = nn.ModuleDict(decoder)
        self.prior = Prior(0., 1.)

        self.sp_u2x, self.sp_x2u, self.sh_u2x, self.sh_x2u = nn.ModuleDict(self.sp_u2x), nn.ModuleDict(self.sp_x2u), \
                                                             nn.ModuleDict(self.sh_u2x), nn.ModuleDict(self.sh_x2u)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.HSIC_reg = HSICLoss()
        self.align_reg = Discriminator(in_features=latent_dim)

    def relocate(self):
        self.sp_u2x, self.sp_x2u, self.sh_u2x, self.sh_x2u = self.sp_u2x.to(self.device), self.sp_x2u.to(self.device), \
                                                             self.sh_u2x.to(self.device), self.sh_x2u.to(self.device)
        self.prior = self.prior.to(self.device)
        self.g2v, self.v2g = self.g2v.to(self.device), self.v2g.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.HSIC_reg = self.HSIC_reg.to(self.device)
        self.align_reg = self.align_reg.to(self.device)

    def save(self, fname: os.PathLike) -> None:
        r"""
        Save model to file

        Parameters
        ----------
        file
            Specifies path to the file

        Note
        ----
        Only the network is saved but not the trainer
        """
        fname = pathlib.Path(fname)
        with fname.open("wb") as f:
            dill.dump(self, f, protocol=4, byref=False, recurse=True)

    def adopt_pretrained_model(
            self, path, submodule: Optional[str] = None
    ) -> None:
        r"""
        Adopt buffers and parameters from a pretrained model

        Parameters
        ----------
        path
            Source model to be adopted
        submodule
            Only adopt a specific submodule (e.g., ``"x2u"``)
        """
        fname = pathlib.Path(path)
        with fname.open("rb") as f:
            source = dill.load(f)
        source, target = source, self
        if submodule:
            source = get_chained_attr(source, submodule)
            target = get_chained_attr(target, submodule)
        for k, t in chain(target.named_parameters(), target.named_buffers()):
            try:
                s = get_chained_attr(source, k)
            except AttributeError:
                print("Missing: %s" % k)
                continue
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if isinstance(s, torch.nn.Parameter):
                s = s.data
            if s.shape != t.shape:
                print("Shape mismatch: %s" % k)
                continue
            s = s.to(device=t.device, dtype=t.dtype)
            t.copy_(s)
            print("Copied: %s" % k)

    def process_linked_feature(self, data, mask) -> (D.Normal, D.Normal, D.Bernoulli, D.Distribution, torch.Tensor, torch.Tensor):
        x_link, xrep, graph, graph_norm, neg_graph, eidx, ewt, esgn, enorm = data
        u, l = {}, {}
        for k in self.idx.keys():
            u[k], l[k] = self.sh_x2u[k](x_link[k], xrep[k],
                                        graph=graph_norm[k],
                                        lazy_normalizer=False)

        usamp = {k:
                     u[k].rsample()
                 for k in u.keys()
                 }
        v = self.g2v(eidx, enorm, esgn)
        vsamp = v.rsample()
        rec = {k: self.sh_u2x[k](
            usamp[k], vsamp[self.idx[k]], l[k], mask[k]
        ) for k in self.idx.keys()}

        graph_rec = self.v2g(vsamp, eidx, esgn)

        return u, usamp, v, graph_rec, rec

    def process_unlinked_feature(self, data, mask) -> (D.Normal, D.Distribution, torch.Tensor, torch.Tensor):
        x_unlink, xrep, graph, graph_norm, neg_graph, eidx, ewt, esgn, enorm = data
        u, l = {}, {}
        for k in self.idx.keys():
            u[k], l[k] = self.sp_x2u[k](x_unlink[k], xrep[k],
                                            graph=graph_norm[k],
                                            lazy_normalizer=False)
        usamp = {k:
                     u[k].rsample()
                 for k in u.keys()
                 }
        rec = {k: self.sp_u2x[k](
            u=usamp[k], l=l[k], graph=graph_norm[k], mask=mask[k]
        ) for k in self.idx.keys()}

        return u, usamp, rec

    @torch.no_grad()
    def encode_graph(
            self, graph
    ) -> np.ndarray:
        r"""
        Compute graph (feature) embedding

        Parameters
        ----------
        graph
            Input graph

        Returns
        -------
        graph_embedding
            Graph (feature) embedding
            with shape :math:`n_{feature} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{feature} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.eval()
        eidx, ewt, esgn = form_graph_data(graph, self.vertices)
        enorm = torch.as_tensor(
            normalize_edges(eidx, ewt),
            device=self.device
        )
        esgn = torch.as_tensor(esgn, device=self.device)
        eidx = torch.as_tensor(eidx, device=self.device)

        v = self.g2v(eidx, enorm, esgn)

        return v.mean.detach().cpu().numpy()

    @torch.no_grad()
    def encode_multimodal(self, adatas: List[sc.AnnData], args: Mapping[str, Any], graph_list, graph_norm_list, neg_graph):
        self.eval()
        self.relocate()
        device = self.device
        keys = self.modalities.keys()
        x, xrep, graph, graph_norm, neg_graph = form_adatas(adatas=adatas,
                                                            use_layers=args['use_layers'],
                                                            use_rep=args['use_rep'],
                                                            rep_dim=args['rep_dim'],
                                                            cutoff=args['cutoff'],
                                                            KNN_neighbors=args['KNN_neighbors'],
                                                            features=args['features'],
                                                            graph_list=graph_list,
                                                            graph_norm_list=graph_norm_list,
                                                            neg_graph=neg_graph)
        x = {
            k: torch.tensor(x[i], device=device, dtype=torch.float32)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: torch.tensor(xrep[i], device=device, dtype=torch.float32)
            for i, k in enumerate(keys)
        }
        graph = {
            k: graph[i].to(device)
            for i, k in enumerate(keys)
        }
        graph_norm = {
            k: graph_norm[i].to(device)
            for i, k in enumerate(keys)
        }
        unlinked_u, linked_u= {}, {}
        for k in self.idx.keys():
            linked_u[k], _ = self.sh_x2u[k](None, xrep[k],
                                           graph=graph_norm[k],
                                           lazy_normalizer=True)
            unlinked_u[k], _ = self.sp_x2u[k](None, xrep[k],
                                           graph=graph_norm[k],
                                           lazy_normalizer=True)

        # fusion = torch.cat([linked_u['rna'].mean, unlinked_u['rna'].mean, linked_u['epg'].mean, unlinked_u['epg'].mean], dim=-1)
        fusion = torch.cat([linked_u['rna'].mean, unlinked_u['rna'].mean, linked_u['epg'].mean, unlinked_u['epg'].mean], dim=-1)
        # fusion = torch.cat([linked_u['rna'].mean, linked_u['epg'].mean], dim=-1) # ablation share feature only
        # fusion = torch.cat([linked_u['epg'].mean, unlinked_u['epg'].mean], dim=-1) # ablation private feature only
        fusion = F.normalize(fusion, p=2, eps=1e-12, dim=1)
        return fusion.detach().cpu().numpy()

    @torch.no_grad()
    def decode_modality(self, adatas: List[sc.AnnData], args: Mapping[str, Any], graph, graph_list, graph_norm_list, neg_graph):
        self.eval()
        self.relocate()
        device = self.device
        keys = self.modalities.keys()
        eidx, ewt, esgn = form_graph_data(graph, self.vertices)
        enorm = torch.as_tensor(
            normalize_edges(eidx, ewt),
            device=self.device
        )
        esgn = torch.as_tensor(esgn, device=self.device)
        eidx = torch.as_tensor(eidx, device=self.device)
        
        x, xrep, graph, graph_norm, neg_graph = form_adatas(adatas=adatas,
                                                            use_layers=args['use_layers'],
                                                            use_rep=args['use_rep'],
                                                            rep_dim=args['rep_dim'],
                                                            cutoff=args['cutoff'],
                                                            KNN_neighbors=args['KNN_neighbors'],
                                                            features=args['features'],
                                                            graph_list=graph_list,
                                                            graph_norm_list=graph_norm_list,
                                                            neg_graph=neg_graph)
        x = {
            k: torch.tensor(x[i], device=device, dtype=torch.float32)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: torch.tensor(xrep[i], device=device, dtype=torch.float32)
            for i, k in enumerate(keys)
        }
        graph = {
            k: graph[i].to(device)
            for i, k in enumerate(keys)
        }
        graph_norm = {
            k: graph_norm[i].to(device)
            for i, k in enumerate(keys)
        }

        l, unlinked_u, linked_u = {}, {}, {}
        for k in self.idx.keys():
            linked_u[k], l[k] = self.sh_x2u[k](x[k], xrep[k],
                                              graph=graph_norm[k],
                                              lazy_normalizer=False)
            unlinked_u[k], l[k] = self.sp_x2u[k](x[k], xrep[k],
                                                  graph=graph_norm[k],
                                                  lazy_normalizer=False)
        l = {'rna': 1e4*torch.ones_like(l['rna']).reshape([-1, 1]), 'epg': 1e4*torch.ones_like(l['rna']).reshape([-1, 1])}
        # fusion = {k: torch.cat([linked_u[k].mean, unlinked_u[k].mean], dim=-1) for k in self.idx.keys()}
        fusion = torch.cat([linked_u['rna'].mean, unlinked_u['rna'].mean, linked_u['epg'].mean, unlinked_u['epg'].mean], dim=-1)
        # fusion = torch.cat([linked_u['epg'].mean, unlinked_u['epg'].mean], dim=-1)
        rec = {k: self.decoder[k](fusion, l[k], graph[k]).mean.detach().cpu().numpy() for k in self.decoder.keys()}
        
        return (rec[k] for k in self.modalities.keys())
