import sys
import argparse
import time
import re
import pickle
import copy
from datetime import datetime
import math
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from utils import args2train_test_sizes, print_time
from models.fcn import FCN
from init import init_fun
from optim_loss import loss_func, regularize, measure_accuracy, opt_algo
from datasets import RandomHierarchyModel

def calculate_synonymy_invariance(args, model):
    def calculate_synonymy_invariance_single(args, model, reset_layer):
        trained_model = copy.deepcopy(model)
        trained_model.eval()
        def generate_collate_fn(args):
            def custom_collate(batch):
                # Unpack the batch
                inputs, labels = zip(*batch)
                
                # Add [CLS] token to the beginning of each input sequence
                inputs = [torch.hstack((torch.tensor([[0.] for _ in range(args.num_features)]), seq)) for seq in inputs]
                
                # Stack the modified input sequences and labels
                inputs = torch.stack(inputs)
                labels = torch.stack(labels)
                
                return inputs, labels
            return custom_collate
        custom_collate = generate_collate_fn(args)


        real_dataset = RandomHierarchyModel(num_features=args.num_features,
                num_classes=args.num_classes,
                num_synonyms=args.m,
                tuple_size=args.s,	# size of the low-level representations
                num_layers=args.num_layers,
                seed_rules=args.seed_rules,
                seed_sample=args.seed_sample,
                train_size=args.ptr,
                test_size=args.pte,
                input_format='onehot',
                whitening=args.whitening,
                transform=None,
                reset_layer=reset_layer,
                )

        if args.cls_token:
            real_loader = torch.utils.data.DataLoader(
                real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)
        else:
            real_loader = torch.utils.data.DataLoader(
                real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            
        syn_dataset = RandomHierarchyModel(num_features=args.num_features,
                num_classes=args.num_classes,
                num_synonyms=args.m,
                tuple_size=args.s,	# size of the low-level representations
                num_layers=args.num_layers,
                seed_rules=args.seed_rules,
                seed_sample=args.seed_sample,
                train_size=args.ptr,
                test_size=args.pte,
                input_format='onehot',
                whitening=args.whitening,
                transform=None,
                reset_layer=2,
        )
        if args.cls_token:
            syn_loader = torch.utils.data.DataLoader(
                syn_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)
        else:
            syn_loader = torch.utils.data.DataLoader(
                syn_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
            
        nb_layers = len(trained_model.layers)
        res_list = []

        intermediate_original_output = [[] for _ in range(nb_layers)]
        intermediate_syn_output = [[] for _ in range(nb_layers)]

        def generate_hook_fn(layer_nb):
            def hook_fn(module, input, output):
                intermediate_original_output[layer_nb].append(output)
            return hook_fn

        def generate_hook_fn_syn(layer_nb):
            def hook_fn(module, input, output):
                intermediate_syn_output[layer_nb].append(output)
            return hook_fn

        for nb_layer in range(nb_layers):
            trained_model.layers[nb_layer].self_attn.register_forward_hook(generate_hook_fn(nb_layer))

        for batch_idx, (inputs, targets) in enumerate(real_loader):
            if batch_idx >= 3:
                break
            inputs, _ = inputs.to(args.device), targets.to(args.device)
            trained_model(inputs)

        for nb_layer in range(nb_layers):
            trained_model.layers[nb_layer].self_attn.register_forward_hook(generate_hook_fn_syn(nb_layer))

        for batch_idx, (inputs, targets) in enumerate(syn_loader):
            if batch_idx >= 3:
                break
            inputs, _ = inputs.to(args.device), targets.to(args.device)
            trained_model(inputs)

        res_list = []
        for nb_layer in range(nb_layers):
            res_list.append(((intermediate_original_output[nb_layer][0][0] - intermediate_syn_output[nb_layer][0][0]).norm()/(intermediate_original_output[nb_layer][0][0] - intermediate_original_output[nb_layer][1][0]).norm()).item())
        return res_list
    res_dict = {}
    for reset_layer in range(args.num_layers):
        res_dict[reset_layer] = calculate_synonymy_invariance_single(args, model, reset_layer)
    return res_dict
