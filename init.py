import torch

from datasets import dataset_initialization
from models import model_initialization

import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

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

def init_fun(args) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.nn.Module):
    """
        Initialize dataset and architecture.
    """
    torch.manual_seed(args.seed_init)

    trainset, testset, input_dim, ch = dataset_initialization(args)
    custom_collate = generate_collate_fn(args)

    if args.cls_token:
        input_dim += 1
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate)
    else:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    if testset:
        if args.cls_token:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=0, collate_fn=custom_collate)
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=0)
    else:
        testloader = None

    net = model_initialization(args, input_dim=input_dim, ch=ch)

    return trainloader, testloader, net



