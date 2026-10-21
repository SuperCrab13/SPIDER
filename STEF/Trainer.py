import torch.nn.functional as F
import time
import torch.distributions as D
import torch
from .model import STEF
from typing import Mapping, List
from .data_utils import form_graph_data, form_adatas
import scanpy as sc
from .utils import normalize_edges
import pandas as pd
import itertools


class Trainer:
    def __init__(self, net: STEF, train_args: Mapping, adatas: List[sc.AnnData], graph, vertices, modalities: Mapping, spatial_graph, spatial_graph_norm, spatial_neg_graph):
        self.train_args = train_args
        self.net = net
        self.adatas = adatas
        self.graph = graph
        self.vertices = vertices
        sample_size = adatas[0].shape[0]
        if any(adata.shape[0] != sample_size for adata in adatas):
            raise ValueError("All modalities must have the same number of samples.")
        self.modalities = train_args['modality_config']
        self.modality_keys = tuple(self.net.idx.keys())
        if self.train_args['optim'] == 'RMSprop':
            self.vae_opt = torch.optim.RMSprop(lr=self.train_args['lr'],
                                               params=itertools.chain(self.net.sh_u2x.parameters(),
                                                                      self.net.sh_x2u.parameters(),
                                                                      self.net.decoder.parameters(),
                                                                      self.net.g2v.parameters(),
                                                                      self.net.v2g.parameters(),
                                                                      self.net.HSIC_reg.parameters()
                                                                      ),
                                               weight_decay=1e-5)
            self.dsc_opt = torch.optim.RMSprop(lr=self.train_args['lr'], params=self.net.align_reg.parameters(),
                                               weight_decay=1e-5)
        elif self.train_args['optim'] == 'Adam':
            self.vae_opt = torch.optim.Adam(lr=self.train_args['lr'],
                                            params=itertools.chain(self.net.sh_u2x.parameters(),
                                                                   self.net.sh_x2u.parameters(),
                                                                   self.net.decoder.parameters(),
                                                                   self.net.g2v.parameters(),
                                                                   self.net.v2g.parameters(),
                                                                   self.net.HSIC_reg.parameters()
                                                                   ),
                                            weight_decay=1e-5)
            self.dsc_opt = torch.optim.Adam(lr=self.train_args['lr'], params=self.net.align_reg.parameters(),
                                            weight_decay=1e-5)
        else:
            raise NotImplementedError
        # self.sche = torch.optim.lr_scheduler.ReduceLROnPlateau(self.vae_opt, patience=self.train_args['patience'])
        self.sche = torch.optim.lr_scheduler.StepLR(self.vae_opt, step_size=100, gamma=0.1)
        self.spatial_graph, self.spatial_graph_norm, self.spatial_neg_graph = spatial_graph, spatial_graph_norm, spatial_neg_graph

    def format_data(self):
        device = self.net.device
        keys = self.modality_keys
        x, xrep, graph, graph_norm, neg_graph = form_adatas(adatas=self.adatas,
                                                            use_layers=self.modalities['use_layers'],
                                                            use_rep=self.modalities['use_rep'],
                                                            rep_dim=self.modalities['rep_dim'],
                                                            graph_list = self.spatial_graph,
                                                            graph_norm_list = self.spatial_graph_norm,
                                                            neg_graph = self.spatial_neg_graph,
                                                            cutoff=self.train_args['cutoff'],
                                                            KNN_neighbors=self.train_args['KNN_neighbors'],
                                                            features=self.modalities['features'])
        eidx, ewt, esgn = form_graph_data(self.graph, vertices=self.vertices)

        x = {
            k: torch.as_tensor(x[i], device=device, dtype=torch.float32)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: torch.as_tensor(xrep[i], device=device, dtype=torch.float32)
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
        neg_graph = {
            k: neg_graph[i].to(device)
            for i, k in enumerate(keys)
        }

        enorm = torch.as_tensor(
            normalize_edges(eidx, ewt),
            device=self.net.device
        )
        eidx = torch.as_tensor(eidx, device=device, dtype=torch.int64)
        ewt = torch.as_tensor(ewt, device=device, dtype=torch.float32)
        esgn = torch.as_tensor(esgn, device=device, dtype=torch.float32)

        return x, xrep, graph, graph_norm, neg_graph, eidx, ewt, esgn, enorm

    def generate_mask(self, seed, shape, device):
        mask_ratio = float(self.train_args['p_mask'])
        if mask_ratio <= 0:
            full_mask = torch.arange(shape, device=device)
            return {k: full_mask for k in self.modality_keys}

        mask_size = int(mask_ratio * shape)
        if mask_size <= 0:
            full_mask = torch.arange(shape, device=device)
            return {k: full_mask for k in self.modality_keys}

        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        shared_mask = torch.randperm(shape, generator=generator, device=device)[:mask_size]
        return {k: shared_mask for k in self.modality_keys}

    def get_losses(self, epoch: int, data, dsc_only=False):
        x, xrep, graph, graph_norm, neg_graph, eidx, ewt, esgn, enorm = data
        l = {k: x[k].sum(dim=1, keepdim=True) for k in self.modality_keys}

        xrep_mask = {}
        sample_count = next(iter(x.values())).shape[0]
        if self.train_args['p_mask'] > 0:
            mask_period = max(1, int(1 / self.train_args['p_mask']))
            mask = self.generate_mask(epoch % mask_period, sample_count, next(iter(x.values())).device)
            for k in self.modality_keys:
                xrep_mask[k] = xrep[k].clone()
                xrep_mask[k][mask[k]] = 0
        else:
            full_mask = torch.arange(sample_count, device=next(iter(x.values())).device)
            mask = {k: full_mask for k in self.modality_keys}
            xrep_mask = xrep
        xrep = xrep_mask
        unlinked_x, linked_x = {}, {}
        for k in self.modality_keys:
            linked_idx = self.net.linked_idx[k].to(device=x[k].device, dtype=torch.long)
            unlinked_idx = self.net.unlinked_idx[k].to(device=x[k].device, dtype=torch.long)
            linked_x[k] = torch.index_select(x[k], dim=1, index=linked_idx).clone()
            unlinked_x[k] = torch.index_select(x[k], dim=1, index=unlinked_idx).clone()
        unlinked_data = (unlinked_x, xrep, graph, graph_norm, neg_graph, eidx, ewt, esgn, enorm)
        unlinked_u, unlinked_usamp, unlinked_rec = self.net.process_unlinked_feature(unlinked_data, mask)
        linked_data = (linked_x, xrep, graph, graph_norm, neg_graph, eidx, ewt, esgn, enorm)
        linked_u, linked_usamp, v, graph_rec, linked_rec = self.net.process_linked_feature(linked_data, mask)

        ## calculate loss
        unlinked_x_nll = {
            k: -unlinked_rec[k].log_prob(unlinked_x[k][mask[k], :]).mean()
            for k in self.modality_keys
        }
        unlinked_x_kl = {
            k: D.kl_divergence(unlinked_u[k], self.net.prior())[mask[k]].sum(dim=1).mean() / mask[k].shape[0]
            for k in self.modality_keys
        }

        unlinked_x_elbo = {
            k: unlinked_x_nll[k] + self.train_args['lam_kl'] * unlinked_x_kl[k]
            for k in self.modality_keys
        }
        unlinked_elbo = sum(unlinked_x_elbo.values())

        linked_x_nll = {
            k: -linked_rec[k].log_prob(linked_x[k][mask[k], :]).mean()
            for k in self.modality_keys
        }
        linked_x_kl = {
            k: D.kl_divergence(linked_u[k], self.net.prior())[mask[k]].sum(dim=1).mean() / mask[k].shape[0]
            for k in self.modality_keys
        }

        linked_x_elbo = {
            k: linked_x_nll[k] + self.train_args['lam_kl'] * linked_x_kl[k]
            for k in self.modality_keys
        }
        linked_elbo = sum(linked_x_elbo.values())        
        
        g_nll = -graph_rec.log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum().item()
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0) + (n_neg > 0)
        g_nll = (g_nll_pn[0] / max(n_neg, 1) + g_nll_pn[1] / max(n_pos, 1)) / avgc
        g_kl = D.kl_divergence(v, self.net.prior()).sum(dim=1).mean() / v.mean.shape[0]
        g_elbo = g_nll + self.train_args['lam_kl'] * g_kl

        ## fusion loss
        fusion_samp = torch.cat([linked_usamp['rna'], unlinked_usamp['rna'], linked_usamp['epg'], unlinked_usamp['epg']], dim=-1)
        # fusion_samp = torch.cat([linked_usamp['epg'], unlinked_usamp['epg']], dim=-1)
        nll = {
            k: -self.net.decoder[k](fusion_samp, l[k], graph[k], mask[k]).log_prob(x[k][mask[k], :]).mean()
            for k in self.modality_keys
        }

        elbo = {k: nll[k] for k in self.modality_keys}
        elbo = sum(elbo.values())
        spatial_nll = {}
        for k in self.modality_keys:
            spatial_nll[k] = -torch.log(1-self.net.graph_decoder(fusion_samp, neg_graph[k], sigmoid=True) + 1e-6).mean() - torch.log(self.net.graph_decoder(fusion_samp, graph[k], sigmoid=True) + 1e-6).mean()
            spatial_nll[k] = spatial_nll[k] / 2
            
        spatial_nll = sum(spatial_nll.values())
        reg = sum(self.net.HSIC_reg(linked_usamp[k], unlinked_usamp[k]) for k in self.modality_keys)/2
        mmd = self.net.align_reg(linked_usamp['rna'], linked_usamp['epg'])
        
        sum_loss = self.train_args['lam_spatial']*spatial_nll + self.train_args['lam_share']*linked_elbo + self.train_args['lam_graph']*g_elbo + self.train_args['lam_private']*unlinked_elbo + self.train_args['lam_data']*elbo + self.train_args['lam_hsic']*reg + mmd
        
        losses = {'spatial_nll': spatial_nll, 'linked_elbo': linked_elbo,
                  'unlinked_elbo': unlinked_elbo, 'g_elbo': g_elbo, 'sum_loss': sum_loss,
                  'elbo': elbo}
        return losses

    def train_step(
            self, epoch: int, data
    ):  # pragma: no cover
        self.net.train()

        losses = self.get_losses(epoch, data, dsc_only=True)
        self.net.zero_grad(set_to_none=True)
        losses["sum_loss"].backward()
        self.dsc_opt.step()

        losses = self.get_losses(epoch, data)
        self.net.zero_grad(set_to_none=True)
        losses["sum_loss"].backward()
        # torch.nn.utils.clip_grad_value_(self.net.parameters(), clip_value=1.0)
        self.vae_opt.step()

        return losses

    def fit(self, result_dir):
        loss, epoch_list = [], []
        start = time.time()
        total_start = start
        data = self.format_data()
        for epoch in range(self.train_args['max_epochs']):
            losses = self.train_step(epoch, data)
            self.sche.step()
            if epoch > 0 and (epoch % self.train_args['PRINT_LOSS_INTERVAL'] == 0):
                end = time.time()
                print(f'Epoch {epoch}:')
                print(f"| linked_elbo: {losses['linked_elbo'].item(): .3f}     | \n"
                      f"| unlinked_elbo: {losses['unlinked_elbo'].item(): .3f}   |\n"
                      f"| g_elbo: {losses['g_elbo'].item(): .3f}          |\n"
                      f"| spatial_nll: {losses['spatial_nll'].item(): .3f}     |\n"
                      f"| sum_loss: {losses['sum_loss'].item(): .3f}        |\n"
                      f"| fusion_loss: {losses['elbo'].item(): .3f}     |")
                print(f'|Time: {end - start: .3f}')
                start = time.time()
                loss.append(losses['sum_loss'].item())
                epoch_list.append(epoch)
        print(f"Total time: {time.time() - total_start: .3f}")
        pd.DataFrame({'epoch': epoch_list, 'loss': loss}).to_csv(result_dir+'/training_loss.csv')
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
