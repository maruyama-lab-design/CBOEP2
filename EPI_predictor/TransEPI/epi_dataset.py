#!/usr/bin/env python3

import argparse, os, sys, time
import warnings, json, gzip
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from misc_utils import hg19_chromsize

import numpy as np

from typing import Dict, List, Union

from functools import partial

from matplotlib import pyplot as plt


def custom_open(fn):
    if fn.endswith("gz"):
        return gzip.open(fn, 'rt')
    else:
        return open(fn, 'rt')


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument()

    p.add_argument('--seed', type=int, default=2020)
    return p


class EPIDataset(Dataset):
    def __init__(self, 
            datasets: Union[str, List], 
            feats_config: Dict[str, str], 
            feats_order: List[str], 
            seq_len: int=2500000, 
            bin_size: int=500, 
            use_mark: bool=False,
            use_mask: bool=False, # mask window and neighbor
            sin_encoding=False,
            rand_shift=False,
            
            **kwargs):
        super(EPIDataset, self).__init__()

        if type(datasets) is str:
            self.datasets = [datasets]
        else:
            self.datasets = datasets

        self.bin_size = int(bin_size)
        if use_mask == False:
            self.seq_len = int(seq_len)
            assert self.seq_len % self.bin_size == 0, "{} / {}".format(self.seq_len, self.bin_size)
            self.num_bins = seq_len // bin_size

        self.feats_order = list(feats_order)
        self.num_feats = len(feats_order)
        self.feats_config = json.load(open(feats_config))
        if "_location" in self.feats_config:
            location =self.feats_config["_location"] 
            del self.feats_config["_location"]
            for cell, assays in self.feats_config.items():
                for a, fn in assays.items():
                    self.feats_config[cell][a] = os.path.join(location, fn)
        else:
            location = os.path.dirname(os.path.abspath(feats_config))
            for cell, assays in self.feats_config.items():
                for a, fn in assays.items():
                    self.feats_config[cell][a] = os.path.join(location, fn)


        self.feats = dict() # cell_name -> feature_name -> chrom > features (array)
        self.chrom_bins = {
                chrom: (length // bin_size) for chrom, length in hg19_chromsize.items()
                }

        self.samples = list()
        self.metainfo = {
                'label': list(), 
                'dist': list(), 
                'chrom': list(), 
                'cell': list(),
                'enh_name': list(),
                'prom_name': list(),
                'shift': list()
                }

        self.sin_encoding = sin_encoding
        self.use_mark = use_mark
        self.mask_window = use_mask
        self.mask_neighbor = use_mask
        self.rand_shift = rand_shift

        self.load_datasets()
        self.feat_dim = len(self.feats_order) + 1
        if self.use_mark:
            self.feat_dim += 1
        if self.sin_encoding:
            self.feat_dim += 1

    def load_datasets(self):
        for fn in self.datasets:
            with custom_open(fn) as infile:
                for l in infile:
                    fields = l.strip().split(',')[:10]
                    label, dist, chrom, enh_start, enh_end, enh_name, \
                            _, prom_start, prom_end, prom_name = fields[0:10]
                    
                    if label == "label": # skip header
                        continue

                    label = int(float(label))
                    dist = int(float(dist))
                    enh_start, enh_end = int(float(enh_start)), int(float(enh_end))
                    prom_start, prom_end = int(float(prom_start)), int(float(prom_end))
                    


                    knock_range = None
                    if len(fields) > 10:
                        assert len(fields) == 11
                        knock_range = list()
                        for knock in fields[10].split(';'):
                            knock_start, knock_end = knock.split('-')
                            knock_start, knock_end = int(knock_start), int(knock_end)
                            knock_range.append((knock_start, knock_end))

                    cell = enh_name.split('|')[0]

                    if cell == "HeLa-S3":
                        cell = "HeLa"

                    enh_coord = (int(enh_start) + int(enh_end)) // 2 # mid point of enhancer
                    p_start, p_end = prom_name.split('|')[1].split(':')[-1].split('-') # two consecutive positions representing the TSS of a gene. 
                    tss_coord = (int(p_start) + int(p_end)) // 2 # TSS of gene. 

                    enh_bin = enh_coord // self.bin_size
                    prom_bin = tss_coord // self.bin_size

                    if self.mask_window and self.mask_neighbor:
                        seq_begin, seq_end, start_bin, stop_bin = -1, -1, -1, -1
                    else:
                        seq_begin = (enh_coord + tss_coord) // 2 - self.seq_len // 2
                        seq_end = (enh_coord + tss_coord) // 2 + self.seq_len // 2

                        # assert seq_begin <= enh_coord and seq_begin <= tss_coord, f"seq_begin:{seq_begin}, enh_coord:{enh_coord}, tss_coord:{tss_coord}"
                        if seq_begin <= enh_coord and seq_begin <= tss_coord:
                            f"seq_begin:{seq_begin}, enh_coord:{enh_coord}, tss_coord:{tss_coord}"

                        # assert enh_coord < seq_end and tss_coord < seq_end, f"seq_end:{seq_end}, enh_coord:{enh_coord}, tss_coord:{tss_coord}"
                        if enh_coord < seq_end and tss_coord < seq_end:
                            f"seq_end:{seq_end}, enh_coord:{enh_coord}, tss_coord:{tss_coord}"
                        
                        start_bin, stop_bin = seq_begin // self.bin_size, seq_end // self.bin_size


                    left_pad_bin, right_pad_bin = 0, 0
                    if start_bin < 0:
                        left_pad_bin = abs(start_bin)
                        start_bin = 0
                    if stop_bin > self.chrom_bins[chrom]: 
                        right_pad_bin = stop_bin - self.chrom_bins[chrom] 
                        stop_bin = self.chrom_bins[chrom]

                    shift = 0
                    if self.rand_shift:
                        if left_pad_bin > 0:
                            shift = left_pad_bin
                            start_bin = -left_pad_bin
                            left_pad_bin = 0
                        elif right_pad_bin > 0:
                            shift = -right_pad_bin
                            stop_bin = self.chrom_bins[chrom] + right_pad_bin
                            right_pad_bin = 0
                        else:
                            min_range = min(min(enh_bin, prom_bin) - start_bin, stop_bin - max(enh_bin, prom_bin))
                            if min_range > (self.num_bins / 4):
                                shift = np.random.randint(-self.num_bins // 5, self.num_bins // 5)
                            if start_bin + shift <= 0 or stop_bin + shift >= self.chrom_bins[chrom]:
                                shift = 0

                    self.samples.append((
                        start_bin + shift, stop_bin + shift, 
                        left_pad_bin, right_pad_bin, 
                        enh_bin, prom_bin, 
                        cell, chrom, np.log2(1 + 500000 / float(dist)),
                        int(label), knock_range
                    ))

                    self.metainfo['label'].append(int(label))
                    self.metainfo['dist'].append(float(dist))
                    self.metainfo['chrom'].append(chrom)
                    self.metainfo['cell'].append(cell)
                    self.metainfo['enh_name'].append(enh_name)
                    self.metainfo['prom_name'].append(prom_name)
                    self.metainfo['shift'].append(shift)

                    if cell not in self.feats:
                        self.feats[cell] = dict()
                        for feat in self.feats_order:
                            # self.feats[cell][feat] = torch.load(self.feats_config[cell][feat]) # for FutureWarning. @2024-10-28.
                            self.feats[cell][feat] = torch.load(self.feats_config[cell][feat], weights_only=True)

        for k in self.metainfo:
            self.metainfo[k] = np.array(self.metainfo[k])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        start_bin, stop_bin, left_pad, right_pad, enh_bin, prom_bin, cell, chrom, dist, label, knock_range = self.samples[idx]
        enh_idx = enh_bin - start_bin + left_pad
        prom_idx = prom_bin - start_bin + left_pad

        
        if self.mask_neighbor and self.mask_window:
            if prom_idx > enh_idx:
                enh_idx, prom_idx = 12, 37
            elif prom_idx < enh_idx:
                enh_idx, prom_idx = 37, 12
            else:
                enh_idx, prom_idx = 12, 37
        else:
            if enh_idx == self.seq_len // self.bin_size:
                enh_idx -= 1
            if prom_idx == self.seq_len // self.bin_size:
                prom_idx -= 1


        if self.mask_neighbor and self.mask_window:
            ar = torch.zeros((0, 50)) # 10bin + enh(5bin) + 10bin + 10bin + prm(5bin) + 10bin = 50bin
        else:
            ar = torch.zeros((0, stop_bin - start_bin))

        for feat in self.feats_order:
            if self.mask_neighbor and self.mask_window:
                enh_feats = self.feats[cell][feat][chrom][enh_bin-2:enh_bin+3].view(1, -1)
                enh_feats = torch.cat((torch.zeros(1, 10), enh_feats), dim=1)
                enh_feats = torch.cat((enh_feats, torch.zeros(1, 10)), dim=1)
                prom_feats = self.feats[cell][feat][chrom][prom_bin-2:prom_bin+3].view(1, -1)
                prom_feats = torch.cat((torch.zeros(1, 10), prom_feats), dim=1)
                prom_feats = torch.cat((prom_feats, torch.zeros(1, 10)), dim=1)
                assert enh_feats.size() == prom_feats.size()
                cat_feats = torch.cat((enh_feats, prom_feats), dim=1)
                ar = torch.cat((ar, cat_feats), dim=0)
            else:
                ar = torch.cat((ar, self.feats[cell][feat][chrom][start_bin:stop_bin].view(1, -1)), dim=0)

        if not (self.mask_neighbor and self.mask_window):
            ar = torch.cat((
                torch.zeros((self.num_feats, left_pad)),
                ar, 
                torch.zeros((self.num_feats, right_pad))
                ), dim=1)

        if knock_range is not None:
            print(f"kock_range: {knock_range}")
            dim, length = ar.size()
            mask = [1 for _ in range(self.num_bins)]
            for knock_start, knock_end in knock_range:
                knock_start = knock_start // self.bin_size - start_bin + left_pad
                knock_end = knock_end // self.bin_size - start_bin + left_pad
                for pos in range(max(0, knock_start), min(knock_end + 1, self.num_bins)):
                    mask[pos] = 0
            mask = np.array(mask, dtype=np.float32).reshape(1, -1)
            mask = np.concatenate([mask for _ in range(dim)], axis=0)
            mask = torch.FloatTensor(mask)
            ar = ar * mask


        if self.mask_neighbor and self.mask_window:
            pos_enc = torch.arange(50).view(1, -1)
            pos_enc = torch.cat((pos_enc - min(enh_idx, prom_idx), max(enh_idx, prom_idx) - pos_enc), dim=0)
        else:
            pos_enc = torch.arange(self.num_bins).view(1, -1)
            pos_enc = torch.cat((pos_enc - min(enh_idx, prom_idx), max(enh_idx, prom_idx) - pos_enc), dim=0)

        if self.sin_encoding: # may be always False
            print(f"sin_encoding: {self.sin_encoding}")
            pos_enc = torch.sin(pos_enc / 2 / self.num_bins * np.pi).view(2, -1)
        else:
            pos_enc = self.sym_log(pos_enc.min(dim=0)[0]).view(1, -1)

        ar = torch.cat((torch.as_tensor(pos_enc, dtype=torch.float), ar), dim=0)
        

        if self.use_mark:
            mark = [0 for i in range(self.num_bins)]
            mark[enh_idx] = 1
            mark[enh_idx - 1] = 1
            mark[enh_idx + 1] = 1
            mark[prom_idx] = 1
            mark[prom_idx - 1] = 1
            mark[prom_idx + 1] = 1
            ar = torch.cat((
                torch.as_tensor(mark, dtype=torch.float).view(1, -1),
                ar
            ), dim=0)

        return ar, torch.as_tensor([dist], dtype=torch.float), torch.as_tensor([enh_idx], dtype=torch.float), torch.as_tensor([prom_idx], dtype=torch.float), torch.as_tensor([label], dtype=torch.float)

    def sym_log(self, ar):
        sign = torch.sign(ar)
        ar = sign * torch.log10(1 + torch.abs(ar))
        return ar


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    np.random.seed(args.seed)

    all_data = EPIDataset(
            datasets=["../data/BENGI/GM12878.HiC-Benchmark.v3.tsv"],
            feats_config="../data/genomic_features/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            seq_len=2500000,
            bin_size=500,
            mask_window=True,
            mask_neighbor=True,
            sin_encoding=True,
            rand_shift=True
        )

    for i in range(0, len(all_data), 411):
        np.savetxt(
                "data_{}".format(i),
                all_data.__getitem__(i)[0].T,
                fmt="%.4f",
                header="{}\t{}\t{}\n{}".format(all_data.metainfo["label"][i], all_data.metainfo["enh_name"][i], all_data.metainfo["prom_name"][i], all_data.samples[i])
            )

