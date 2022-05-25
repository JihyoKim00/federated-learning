import flwr as fl
import argparse
from typing import List, Tuple, Dict, Optional   
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import flwr as fl
import argparse
from typing import List, Tuple, Dict, Optional   
import torch
from torch.nn import GroupNorm 
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from collections import OrderedDict
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torch.nn as nn
import numpy as np  
import wandb 
from model import Net, mnist_Net
from torchvision.models import resnet18
import warnings 
import os 

warnings.filterwarnings('ignore')


def get_eval_fn(model, args=None):
    """Return an evaluation function for server-side evaluation."""
    
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.transforms.Normalize((0.5, ), (0.5, ))]
    )
    
    cifar_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )


    if args.datasets == 'mnist':
        testset = MNIST("/data/", train=False, download=False, transform=mnist_transform)
    
    if args.datasets == 'cifar10':
        testset = CIFAR10("/data/cifar10_data/", train=False, download=False, transform=cifar_transform)

    elif args.datasets == 'cifar100':
        testset = CIFAR100("/data/cifar/", train=False, download=False, transform=cifar_transform)


    testloader = DataLoader(testset, batch_size=64, num_workers=4, shuffle = False)  
    # The `evaluate` function will be called after every round
    
    
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters 
        state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
        model.load_state_dict(state_dict, strict=True)
        

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        loss_li = []
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels).item()
                loss_li.append(loss)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss = np.mean(loss_li)
        accuracy = correct / total
        wandb.log({"test_loss" : loss,
                    "test_accuracy" : accuracy })
          
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate
    
def get_parameters(model):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Flower-Server")
    parser.add_argument("--model", type=str, default='cnn')
    parser.add_argument("--datasets", type=str, default='cifar100')
    parser.add_argument("--fr_rate", type=float, default=0.0)
    parser.add_argument("--fr_val_rate", type=float, default=0.0)
    parser.add_argument("--round", type=int, default=100)
    parser.add_argument("--min_client", type=int, default=2)
    parser.add_argument("--min_ac", type=int, default=2)
    parser.add_argument("--strategy", type=str, default= 'fedavg')
    parser.add_argument("--gpu", type=str, default='4')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.model == 'cnn':
        if args.datasets == 'mnist':
            model = mnist_Net().cuda()
        elif args.datasets == 'cifar10':
            model = Net().cuda()

    elif args.model == 'resnet18':
        if args.datasets in ['mnist', 'cifar10']: num_classes = 10
        elif args.datasets == 'cifar100': num_classes = 100
        model = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes).cuda()
    
    model.eval()

    # Define strategy
    if args.strategy == 'fedavg' : 
        strategy = fl.server.strategy.FedAvg(
            fraction_fit = args.fr_rate,
            fraction_eval = args.fr_val_rate,
            min_fit_clients = args.min_client,
            min_eval_clients = args.min_client,
            min_available_clients = args.min_ac,
            eval_fn = get_eval_fn(model, args=args),
            initial_parameters = fl.common.weights_to_parameters(get_parameters(model))
        )

    elif args.strategy == 'fedadam' : 
        strategy = fl.server.strategy.FedAdam(
            fraction_fit = args.fr_rate,
            fraction_eval = args.fr_val_rate,
            min_fit_clients = args.min_client,
            min_eval_clients = args.min_client,
            min_available_clients = args.min_ac,
            eval_fn = get_eval_fn(model, args=args),
            initial_parameters = fl.common.weights_to_parameters(get_parameters(model)),
            eta  = 1e-2,
            eta_l = 1e-3,
            beta_1 = 0.9,
            beta_2 = 0.99,
        )

    elif args.strategy == 'fedyogi' : 
        strategy = fl.server.strategy.FedYogi(
            fraction_fit = args.fr_rate,
            fraction_eval = args.fr_val_rate,
            min_fit_clients = args.min_client,
            min_eval_clients = args.min_client,
            min_available_clients = args.min_ac,
            eval_fn = get_eval_fn(model, args=args),
            initial_parameters = fl.common.weights_to_parameters(get_parameters(model)),
            eta  = 1e-2,
            beta_1 = 0.9,
            beta_2 = 0.99,
        )
    
    # momentum == 0 
    elif args.strategy == 'fedadagrad' : 
        strategy = fl.server.strategy.FedAdagrad(
            fraction_fit = args.fr_rate,
            fraction_eval = args.fr_val_rate,
            min_fit_clients = args.min_client,
            min_eval_clients = args.min_client,
            min_available_clients = args.min_ac,
            eval_fn = get_eval_fn(model, args=args),
            initial_parameters = fl.common.weights_to_parameters(get_parameters(model)),
            eta  = 1e-2,
            eta_l  = 1e-2,
            tau  = 1e-9,
        )

    wandb.init(project="project name", entity="entity name", group='group name', job_type='job type name')
    wandb.config.update(args)

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": args.round,
                "fraction_rate" : args.fr_rate,
                "gpu number" : args.gpu},
        strategy=strategy,
    )
