from collections import OrderedDict
import warnings
import argparse 
import wandb 
import flwr as fl
import os  
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np 
from torch.nn import GroupNorm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision.models import resnet18
from model import Net, mnist_Net


warnings.filterwarnings("ignore", category=UserWarning)

wandb.init(project="project name", entity="entity name", group='group name', job_type='job type name')


def train(net, trainloader, epochs, args=None):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss().cuda()
    if (args.strategy == 'fedavg') | (args.strategy == 'fedadagrad') :
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.0)
    elif (args.strategy == 'fedadam') | (args.strategy == 'fedyogi') : 
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)    
    net.train()

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            wandb.log({"train_loss": loss})
            loss.backward()
            if args.clip > 0.0:
                torch.nn.utils.clip_grad_norm(net.parameters(), args.clip)
            optimizer.step()

    del net


def load_data(client, partition, args=None):
    """Load CIFAR-10 (training and test set)."""
    cifar_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    mnist_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.transforms.Normalize((0.5, ), (0.5, ))]
    )
    if args.datasets in ['cifar10', 'dirtycifar10']:
        trainset = CIFAR10("/data/cifar10_data/", train=True, download=False, transform=cifar_transform)

    if args.datasets == 'cifar100':
        trainset = CIFAR100("/data/cifar100_data/", train=True, download=True, transform=cifar_transform)

    elif args.datasets == 'mnist':
        trainset = MNIST("/data/", train=True, download=True, transform=mnist_transform)
        
    num_examples = {'trainset' : len(trainset)}

    # Data Partitioning
    temp = tuple([len(trainset) // client for i in range(client)])
    Worker_data = torch.utils.data.random_split(trainset, temp, torch.Generator().manual_seed(42))
    trainloader = torch.utils.data.DataLoader(Worker_data[partition], batch_size=64, shuffle=True, num_workers=2)

    if args.datasets == 'dirtycifar10':
        if partition in [5,6,7,8,9] :
            trainloader = torch.utils.data.DataLoader(Worker_data[partition], batch_size=64, shuffle=True, num_workers=2)

        elif partition in [0,1,2,3,4]:
            rand_index = torch.randperm(2500, generator=torch.Generator().manual_seed(94))
            subset_labels = [Worker_data[partition][i][1] for i in range(2500)]

            permuted_label = np.array(subset_labels)[rand_index]
            clean_labels = np.array([Worker_data[partition][i][1] for i in range(2500, 5000)])

            tensor_dataset = torch.cat([torch.unsqueeze(Worker_data[partition][i][0], dim=0) for i in range(5000)])
            label_concat = torch.from_numpy(np.hstack((permuted_label, clean_labels)))

            noisy_dataset = torch.utils.data.TensorDataset(tensor_dataset, label_concat)

            trainloader = torch.utils.data.DataLoader(noisy_dataset, batch_size=64, shuffle=True, num_workers=2)


    print(f'{partition}th client train data is {len(trainloader.dataset)}')
    
    return trainloader, num_examples


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def main():
    """Create model, load data, define Flower client, start Flower client."""

    parser = argparse.ArgumentParser(description="Flower-Client")
    parser.add_argument("--partition", type=int, choices=range(0, 100), required=True)
    parser.add_argument("--datasets", type=str, default='cifar100')
    parser.add_argument("--model", type=str, default='cnn')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=0.0)
    parser.add_argument("--num_gpu", type=int, default=2)
    parser.add_argument("--client", type=int, default=10)
    parser.add_argument("--local_ep", type=int, default=1)
    parser.add_argument("--strategy", type=str, default='fedavg')
    parser.add_argument("--gpu", type=str, default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.model == 'cnn':
        if args.datasets == 'mnist':
            net = mnist_Net().cuda()
        elif args.datasets in ['cifar10', 'dirtycifar10']:
            net = Net().cuda() 

    elif args.model == 'resnet18':
        if args.datasets in ['mnist', 'cifar10', 'dirtycifar10']: num_classes = 10
        elif args.datasets == 'cifar100': num_classes = 100
        net = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes).cuda()
    
    wandb.config.update(args)
    wandb.watch(net)

    # Load data (CIFAR-10) client, partition
    trainloader, num_examples = load_data(args.client*args.num_gpu, args.partition, args=args)

    # Flower client
    class CifarClient(fl.client.NumPyClient):
        def __init__(self, args):
            if args.model == 'cnn' :
                if args.datasets == 'mnist' :
                    net = mnist_Net().cuda()
                elif args.datasets in ['cifar10', 'dirtycifar10']:
                    net = Net().cuda()

            elif args.model == 'resnet18' : 
                self.net = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=100).cuda()

        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=args.local_ep, args=args)
            return self.get_parameters(), num_examples["trainset"], {}, # int(loss*1000)

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=CifarClient(args=args))


if __name__ == "__main__":
    main()
