import sys

sys.path.insert(0, "../..")
import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import fire
import torch
from pyrographnets.models import GraphEncoder, GraphCore
from pyrographnets.blocks import (
    EdgeBlock,
    NodeBlock,
    GlobalBlock,
    AggregatingEdgeBlock,
    AggregatingGlobalBlock,
    AggregatingNodeBlock,
    Aggregator,
)
from pyrographnets.blocks import Flex
from pyrographnets.data import GraphBatch, GraphData, GraphDataLoader
from typing import List
from torch import nn
from pyrographnets.utils import pairwise
from typing import Tuple
import wandb
from typing import Dict
from data import generate_data, create_loader, CircuitGenerator
from tqdm.auto import tqdm
from pyrographnets.utils import _first


class MLPBlock(nn.Module):
    """A multilayer perceptron block."""

    def __init__(self, input_size: int, output_size: int = None):
        super().__init__()
        if output_size is None:
            output_size = input_size
        self.blocks = nn.Sequential(
            nn.Linear(input_size, output_size), nn.ReLU(), nn.LayerNorm(output_size)
        )

    def forward(self, x):
        return self.blocks(x)


class MLP(nn.Module):
    """A multilayer perceptron."""

    def __init__(self, *latent_sizes: List[int]):
        super().__init__()
        self.blocks = nn.Sequential(
            *[MLPBlock(n1, n2) for n1, n2 in pairwise(latent_sizes)]
        )

    def forward(self, x):
        return self.blocks(x)


class Network(torch.nn.Module):
    def __init__(
        self,
        latent_sizes=(128, 128, 1),
        depths=(1, 1, 1),
        pass_global_to_edge: bool = True,
        pass_global_to_node: bool = True,
    ):
        super().__init__()
        self.config = {
            "latent_size": {
                "node": latent_sizes[1],
                "edge": latent_sizes[0],
                "global": latent_sizes[2],
                "core_node_block_depth": depths[0],
                "core_edge_block_depth": depths[1],
                "core_global_block_depth": depths[2],
            },
            "node_block_aggregator": "add",
            "global_block_to_node_aggregator": "add",
            "global_block_to_edge_aggregator": "add",
            "pass_global_to_edge": pass_global_to_edge,
            "pass_global_to_node": pass_global_to_node,
        }
        self.encoder = GraphEncoder(
            EdgeBlock(Flex(MLP)(Flex.d(), latent_sizes[0])),
            NodeBlock(Flex(MLP)(Flex.d(), latent_sizes[1])),
            GlobalBlock(Flex(MLP)(Flex.d(), latent_sizes[2])),
        )

        edge_layers = [self.config["latent_size"]["edge"]] * self.config["latent_size"][
            "core_edge_block_depth"
        ]
        node_layers = [self.config["latent_size"]["node"]] * self.config["latent_size"][
            "core_node_block_depth"
        ]
        global_layers = [self.config["latent_size"]["global"]] * self.config[
            "latent_size"
        ]["core_global_block_depth"]

        self.core = GraphCore(
            AggregatingEdgeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), *edge_layers),
                    #                 torch.nn.Linear(latent_sizes[0], latent_sizes[0])
                )
            ),
            AggregatingNodeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), *node_layers),
                    #                 torch.nn.Linear(latent_sizes[1], latent_sizes[1])
                ),
                Aggregator(self.config["node_block_aggregator"]),
            ),
            AggregatingGlobalBlock(
                Flex(MLP)(Flex.d(), *global_layers),
                edge_aggregator=Aggregator(
                    self.config["global_block_to_edge_aggregator"]
                ),
                node_aggregator=Aggregator(
                    self.config["global_block_to_node_aggregator"]
                ),
            ),
            pass_global_to_edge=self.config["pass_global_to_edge"],
            pass_global_to_node=self.config["pass_global_to_node"],
        )

        self.decoder = GraphEncoder(
            EdgeBlock(Flex(MLP)(Flex.d(), latent_sizes[0], latent_sizes[0])),
            NodeBlock(Flex(MLP)(Flex.d(), latent_sizes[1], latent_sizes[1])),
            GlobalBlock(Flex(MLP)(Flex.d(), latent_sizes[2])),
        )

        self.output_transform = GraphEncoder(
            EdgeBlock(Flex(torch.nn.Linear)(Flex.d(), 1)),
            NodeBlock(Flex(torch.nn.Linear)(Flex.d(), 1)),
            GlobalBlock(Flex(torch.nn.Linear)(Flex.d(), 1)),
        )

    def forward(self, data, steps):
        # encoded
        e, x, g = self.encoder(data)
        data = GraphBatch(x, e, g, data.edges, data.node_idx, data.edge_idx)

        # graph topography data
        edges = data.edges
        node_idx = data.node_idx
        edge_idx = data.edge_idx
        latent0 = data

        meta = (edges, node_idx, edge_idx)

        outputs = []
        for _ in range(steps):
            # core processing step
            e = torch.cat([latent0.e, e], dim=1)
            x = torch.cat([latent0.x, x], dim=1)
            g = torch.cat([latent0.g, g], dim=1)
            data = GraphBatch(x, e, g, *meta)
            e, x, g = self.core(data)

            # decode
            data = GraphBatch(x, e, g, *meta)

            _e, _x, _g = self.decoder(data)
            decoded = GraphBatch(_x, _e, _g, *meta)

            # transform
            _e, _x, _g = self.output_transform(decoded)
            outputs.append(GraphBatch(_x, _e, _g, edges, node_idx, edge_idx))

        # revise connectivity

        return outputs


def create_data(train_size: int,
            dev_size: int,
            n_parts: int,
            train_part_range: Tuple[int, int],
            dev_part_range: Tuple[int, int]) -> Dict:
    circuit_gen = CircuitGenerator(n_parts)

    data = generate_data(circuit_gen, train_size, train_part_range, dev_size, dev_part_range)

    return data, circuit_gen

def to(batch, device):
    return GraphBatch(
        batch.x.to(device),
        batch.e.to(device),
        batch.g.to(device),
        batch.edges.to(device),
        batch.node_idx.to(device),
        batch.edge_idx.to(device)
    )

def train():
    """
    This is the documentation for the function.

    :param learning_rate:
    :return:
    """

    import wandb

    # Set up your default hyperparameters before wandb.init
    # so they get properly set in the sweep
    hyperparameter_defaults = dict(
        latent_size_0=254,
        latent_size_1=254,
        latent_size_2=128,
        latent_depth_0=3,
        latent_depth_1=3,
        latent_depth_2=3,
        processing_steps=5,
        eval_processing_steps=5,
        pass_global_to_node=True,
        pass_global_to_edge=True,
        batch_size=512,
        learning_rate=1e-3,
        weight_decay=1e-4,
        epochs=20,
        train_size=10000,
        dev_size=2000
    )

    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)
    # config = wandb.config
    config = hyperparameter_defaults

    # update config

    # Your model here ...
    device = "cuda:0"
    net = Network(
        (config['latent_size_0'], config['latent_size_1'], config['latent_size_2']),
        (config['latent_depth_0'], config['latent_depth_1'], config['latent_depth_2']),
        pass_global_to_edge=config['pass_global_to_edge'],
        pass_global_to_node=config['pass_global_to_node']
    )

    # create your data
    data, gen = create_data(config['train_size'],
                            config['dev_size'],
                            20, (2, 6), (2, 8))
    train_loader = create_loader(gen, data["train"], config['batch_size'], shuffle=True)
    eval_loader = create_loader(gen, data["train/dev"], None, shuffle=False)

    with torch.no_grad():
        batch, _ = _first(train_loader)
        print(batch.shape)
        out = net(batch, 3)
    net.to(device)


    assert list(net.parameters())
    optimizer = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loss_fn = torch.nn.MSELoss()
    log_every_epoch = 10
    for epoch in tqdm(range(config['epochs'])):
        net.train()

        running_loss = 0.
        for batch_idx, (train_data, target_data) in enumerate(train_loader):
            train_data.contiguous()
            target_data.contiguous()
            # TODO: why clone?
            target_data = target_data.clone()
            train_data = to(train_data, device)
            target_data = to(target_data, device)
            out = net(train_data, config['processing_steps'])
            assert out[-1].x.shape == target_data.x.shape

            optimizer.zero_grad()

            # TODO: the scale of the loss is proporational to the processing steps and batch_size, should this be normalized???
            loss = torch.tensor(0.).to(device)
            for _out in out:
                loss += loss_fn(_out.x, target_data.x)
            loss = loss / (target_data.x.shape[0] * 1. * len(out))

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        if epoch % log_every_epoch == 0:
            wandb.log({'train_loss': running_loss}, step=epoch)

        if epoch % log_every_epoch == 0:
            eval_loss = 0.
            net.eval()
            with torch.no_grad():
                for eval_data, eval_target in eval_loader:
                    eval_data = to(eval_data, device)
                    eval_target = to(eval_target, device)
                    eval_outs = net(eval_data, config['eval_processing_steps'])
                    # only take loss from last output
                    eval_loss += loss_fn(eval_outs[-1].x, eval_target.x) / eval_outs[-1].x.shape[0]
                wandb.log({'eval_loss': eval_loss / eval_data.x.shape[0] * 1000}, step=epoch)

if __name__ == "__main__":
    train()
