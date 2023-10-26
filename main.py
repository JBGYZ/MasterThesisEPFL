import sys
import argparse
import time
import re
import pickle
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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import args2train_test_sizes, print_time
from models.fcn import FCN
from init import init_fun
from optim_loss import loss_func, regularize, measure_accuracy, opt_algo


def train(args, trainloader, net, criterion, testloader=None, writer=None):

    optimizer, scheduler = opt_algo(args, net)
    print(f"Training for {args.epochs} epochs...")

    start_time = time.time()

    num_batches = math.ceil(args.ptr / args.batch_size)
    checkpoint_batches = torch.linspace(0, num_batches, 10, dtype=int)

    epochlist = []
    trainloss = []
    trainerr = []
    testerr = []
    best = dict()
    best_acc = 0
    for epoch in range(args.epochs):

        # layerwise training
        if epoch % (args.epochs // args.net_layers + 1) == 0:
            if 'layerwise' in args.net:
                l = epoch // (args.epochs // args.net_layers + 1)
                net.init_layerwise_(l)
                print(f'Layer-wise training up to layer {l}.', flush=True)

        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            train_loss += loss.detach().item()
            regularize(loss, net, args.weight_decay, reg_type=args.reg_type)
            loss.backward()
            optimizer.step()

            correct, total = measure_accuracy(args, outputs, targets, correct, total)
        avg_epoch_time = (time.time() - start_time) / (epoch + 1)

        if epoch % 50 == 0:
            print(
                f"[Train epoch {epoch+1} / {args.epochs}, {print_time(avg_epoch_time)}/epoch, ETA: {print_time(avg_epoch_time * (args.epochs - epoch - 1))}]"
                f"[tr.Loss: {train_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[tr.Acc: {100.*correct/total:.03f}, {correct} / {total}]",
                flush=True
            )

            test_acc = test(args, testloader, net, criterion, print_flag=False)
            net.train()

            epochlist.append(epoch)
            trainloss.append(train_loss * args.alpha / (batch_idx + 1))
            trainerr.append(100 - 100. * correct / total)
            testerr.append(100 - test_acc)
            if test_acc > best_acc:
                best["acc"] = test_acc
                best["epoch"] = epoch
                best_acc = test_acc

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss * args.alpha / (batch_idx + 1), epoch)
                writer.add_scalar("Accuracy/train", 100. * correct / total, epoch)
                writer.add_scalar("Accuracy/test", test_acc, epoch)
        scheduler.step()

    out = {
        "args": args,
        "epoch": epochlist,
        "trainloss": trainloss,
        "trainerr": trainerr,
        "testerr": testerr,
        "best": best
    }
    with open(args.pickle, "wb") as handle:
        pickle.dump(out, handle)

def test(args, testloader, net, criterion, print_flag=True):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()

            correct, total = measure_accuracy(args, outputs, targets, correct, total)

        if print_flag:
            print(
                f"[TEST][te.Loss: {test_loss * args.alpha / (batch_idx + 1):.03f}]"
                f"[te.Acc: {100. * correct / total:.03f}, {correct} / {total}]",
                flush=True
            )

    return 100.0 * correct / total



def main():
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser()

    ### Tensors type ###
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")

    ### Seeds ###
    parser.add_argument(
        "--seed_init", type=int, default=0
    )  # seed random-hierarchy-model
    parser.add_argument("--seed_net", type=int, default=-1)  # network initalisation
    parser.add_argument("--seed_trainset", type=int, default=-1)  # training sample

    ### DATASET ARGS ###
    parser.add_argument("--dataset", type=str, required=True)  # hier1 for hierarchical
    parser.add_argument(
        "--ptr",
        type=float,
        default=0.8,
        help="Number of training point. If in [0, 1], fraction of training points w.r.t. total. If negative argument, P = |arg|*P_star",
    )
    parser.add_argument("--pte", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--scale_batch_size", type=int, default=0)

    parser.add_argument("--background_noise", type=float, default=0)

    # Hierarchical dataset #
    parser.add_argument("--num_features", type=int, default=8)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=-1)
    parser.add_argument("--input_format", type=str, default="onehot")
    parser.add_argument("--whitening", type=int, default=0)
    parser.add_argument("--auto_regression", type=int, default=0)  # not for now

    ### ARCHITECTURES ARGS ###
    parser.add_argument("--net", type=str, required=True)  # fcn or cnn
    parser.add_argument("--random_features", type=int, default=0)

    ## Nets params ##
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--net_layers", type=int, default=3)
    parser.add_argument("--filter_size", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch_norm", type=int, default=0)
    parser.add_argument(
        "--bias", type=int, default=1, help="for some archs, controls bias presence"
    )

    ## Transformer params ##
    parser.add_argument("--pos_encoder_type", type=str, default="absolute")
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--reducer_type", type=str, default="fc")
    parser.add_argument("--reducer_layers", type=int, default=1)
    parser.add_argument("--reducer_size", type=int, default=64)
    parser.add_argument("--embedding_type", type=str, default="none")
    parser.add_argument("--scaleup_dim", type=int, default=64)

    ## Auto-regression with Transformers ##
    parser.add_argument("--pmask", type=float, default=0.2)  # not for now

    ### ALGORITHM ARGS ###
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--optim", type=str, default="sgd")
    parser.add_argument("--scheduler", type=str, default="cosineannealing")
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float)
    parser.add_argument("--reg_type", default="l2", type=str)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--zero_loss_epochs", type=int, default=0)
    parser.add_argument("--zero_loss_threshold", type=float, default=0.01)
    parser.add_argument("--rescale_epochs", type=int, default=0)

    parser.add_argument(
        "--alpha", default=1.0, type=float, help="alpha-trick parameter"
    )

    ### Observables ###
    # how to use: 1 to compute stability every checkpoint; 2 at end of training. Default 0.
    parser.add_argument("--stability", type=int, default=0)
    parser.add_argument("--clustering_error", type=int, default=0)
    parser.add_argument("--locality", type=int, default=0)

    ### SAVING ARGS ###
    parser.add_argument("--save_init_net", type=int, default=1)
    parser.add_argument("--save_best_net", type=int, default=1)
    parser.add_argument("--save_last_net", type=int, default=1)
    parser.add_argument("--save_dynamics", type=int, default=0)

    ## saving path ##
    parser.add_argument("--pickle", type=str, required=False, default="None")
    parser.add_argument("--output", type=str, required=False, default="None")
    args = parser.parse_args()

    if args.pickle == "None":
        assert (
            args.output != "None"
        ), "either `pickle` or `output` must be given to the parser!!"
        args.pickle = args.output

    # special value -1 to set some equal arguments
    if args.seed_trainset == -1:
        args.seed_trainset = args.seed_init
    if args.seed_net == -1:
        args.seed_net = args.seed_init
    if args.num_classes == -1:
        args.num_classes = args.num_features
    if args.net_layers == -1:
        args.net_layers = args.num_layers
    if args.m == -1:
        args.m = args.num_features

    # define train and test sets sizes

    args.ptr, args.pte = args2train_test_sizes(args)
    print(f"Train size: {args.ptr}, Test size: {args.pte}")

    torch.set_default_dtype(torch.float32)

    criterion = partial(loss_func, args)

    trainloader, testloader, model = init_fun(args)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    args_string = ' '.join(sys.argv[1:])
    pattern = r'--(ptr|net|net_layers|nhead|dim_feedforward|scaleup_dim|num_features|pos_encoder_type)\s+([\w.]+)'
    # Use re.findall to extract matches
    matches = re.findall(pattern, args_string)
    # Create a dictionary to store the extracted arguments and their values
    arguments = dict(matches)
    # Concatenate the extracted arguments and their values into a string
    folder_name = "_".join([f"{arg}_{value}" for arg, value in arguments.items()])

    writer = SummaryWriter(log_dir=f'runs/feature{args.num_features}/{folder_name}')
    train(args, trainloader, model, criterion, testloader=testloader, writer=writer)
    test(args, testloader, model, criterion, print_flag=True)

# Example usage:
if __name__ == "__main__":
    main()

