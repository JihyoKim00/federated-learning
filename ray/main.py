import argparse
import os 
import flwr as fl
from flwr.common.typing import Scalar
import os
import ray
import torch
import torchvision
import time  
import numpy as np
import wandb 
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
from dataset_utils import getCIFAR10, getMNIST, do_fl_partitioning, get_dataloader
from utils import Net, Net2, train, test
from models import ResNet18

warnings.filterwarnings('ignore') 

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--datasets", type=str, default='cifar')
parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--client", type=int, default=100)
parser.add_argument("--local_ep", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=50)
parser.add_argument("--gpu", type=str, default='7')
parser.add_argument("--wb", type=bool, default='False')


# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, args=None):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        
        # instantiate model
        #self.net = Net()
        if str(args['datasets']) == 'cifar':
            self.net = Net()
        elif str(args['datasets']) == 'mnist': 
            self.net = Net2()
        
        #self.net = ResNet18()
        
        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

         
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        # print(f"fit() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get train loader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        train_loader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=int(config["batch_size"]),
            workers=num_workers,
            dataset = str(args['datasets']))
        # send model to device
        
        self.net.to(self.device)

        # train
        st = time.time()
        train(self.net, train_loader, epochs=int(args["local_ep"]), device=self.device)
        print(f'train time is {time.time()-st}')
        
        # return local model and statistics
        return self.get_parameters(), len(train_loader.dataset), {}

    def evaluate(self, parameters):

        # print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get train loader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        val_loader = get_dataloader(
            self.fed_dir, self.cid, is_train=False, batch_size=16, workers=num_workers
        )

        # send model to device
        self.net.to(self.device)

        # evaluate
        st = time.time()
        loss, accuracy = test(self.net, val_loader, device=self.device)
        print(f'loss is {loss}')
        print(f'test time is {time.time()-st}')
        # return statistics
        return float(loss), len(val_loader.dataset), {"accuracy": float(accuracy)}

def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "batch_size": str(64),
    }
    return config

def evaluate_config(rnd: int) -> Dict[str, str]: 
    
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}
    


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

def get_eval_fn(
    testset: torchvision.datasets.CIFAR10, args=None) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if str(args['datasets']) == 'cifar':
            net = Net()
        elif str(args['datasets']) == 'mnist': 
            net = Net2()

        #model = ResNet18()
        set_weights(net, weights)
        net.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=128)
        st = time.time()
        loss, accuracy = test(net, testloader, device=device)
        wandb.log({'test loss' : loss,
                   'test accuracy' : accuracy})
        

        print(f'global test time is {time.time()-st}')
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate

# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of clients. We refer to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a Ray-based simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise to each client's data.


if __name__ == "__main__":

    # parse input arguments
    args = vars(parser.parse_args())
    
    # setting the gpu 
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu'])

    pool_size = int(args['client'])  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": int(args['num_client_cpus'])  # each client will get allocated 1 CPUs
    }  
    
    # download CIFAR10 dataset
    if str(args['datasets']) == 'cifar' :
        train_path, testset = getCIFAR10()

    # download MNIST dataset 
    if str(args['datasets']) == 'mnist' :  
        train_path, testset = getMNIST()
    
    # add the FEMNIST dataset 
    
    # wandb option
    

    wandb.init(project="Federated_Learning", entity="sungchul_")
    wandb.config.update(args)
        
    #wandb.init(project="Federated_Learning", entity="sungchul_", mode='disabled')
    
    # data partitioning

    fed_dir = do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1)

    #model = Net()

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit= 0.1,
        fraction_eval = 0.5,
        min_fit_clients = 2,
        min_eval_clients = 2, 
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config, #send a configuration
        on_evaluate_config_fn = evaluate_config,
        eval_fn=get_eval_fn(testset, args=args),  # centralised testset evaluation of global model
    )
    
    #print(strategy)

    # create a single client instance
    def client_fn(cid: str):
        
        return CifarRayClient(cid, fed_dir, args=args)

    # (optional) specify ray config
    ray_config = {"include_dashboard": False}
    
    # start simulation        
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=int(args['num_rounds']),
        strategy=strategy,
        ray_init_args=ray_config,
    )    
