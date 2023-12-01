import torch
import torch.backends.cudnn as cudnn

from .fcn import FCN
from .transformerencoder import TransformerEncoder, AlbertEncoder


def model_initialization(args, input_dim, ch):
    """
    Neural netowrk initialization.
    :param args: parser arguments
    :return: neural network as torch.nn.Module
    """

    num_outputs = 1 if args.loss == "hinge" else args.num_classes

    ### Define network architecture ###
    torch.manual_seed(args.seed_net)

    net = None

    if args.net == "fcn":
        net = FCN(
            num_layers=args.net_layers,
            input_channels=ch * input_dim,
            h=args.width,
            out_dim=num_outputs,
            bias=args.bias,
        )
    elif args.net == "transformer":
        net = TransformerEncoder(
            args=args,
            num_layers=args.net_layers,
            d_model=args.num_features,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            ch=ch,
            input_dim=input_dim,
            num_outputs=num_outputs,
            reducer_type=args.reducer_type,
        )
    elif args.net == "albert":
        net = AlbertEncoder(
            args=args,
            num_layers=args.net_layers,
            d_model=args.num_features,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            ch=ch,
            input_dim=input_dim,
            num_outputs=num_outputs,
            reducer_type=args.reducer_type,
        )        

    assert net is not None, "Network architecture not in the list!"

    if args.random_features:
        for param in [p for p in net.parameters()][:-2]:
            param.requires_grad = False

    net = net.to(args.device)

    # if args.device == "cuda":
    #     net = torch.nn.DataParallel(net)
    #     cudnn.benchmark = True

    return net
