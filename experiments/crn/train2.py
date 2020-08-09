import sys
sys.path.insert(0, '../..')

import fire
import torch
from pyrographnets.models import GraphEncoder, GraphCore
from pyrographnets.blocks import EdgeBlock, NodeBlock, GlobalBlock, AggregatingEdgeBlock, AggregatingGlobalBlock, \
    AggregatingNodeBlock, Aggregator
from pyrographnets.blocks import Flex
from pyrographnets.data import GraphBatch, GraphData, GraphDataLoader
from typing import List
from torch import nn
from pyrographnets.utils import pairwise
import wandb

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

    def __init__(self, latent_sizes=(128, 128, 1),
                 depths=(1, 1, 1),
                 pass_global_to_edge: bool = True,
                 pass_global_to_node: bool = True):
        super().__init__()
        self.config = {
            'latent_size': {
                'node': latent_sizes[1],
                'edge': latent_sizes[0],
                'global': latent_sizes[2],
                'core_node_block_depth': depths[0],
                'core_edge_block_depth': depths[1],
                'core_global_block_depth': depths[2]
            },
            'node_block_aggregator': 'add',
            'global_block_to_node_aggregator': 'add',
            'global_block_to_edge_aggregator': 'add',
            'pass_global_to_edge': pass_global_to_edge,
            'pass_global_to_node': pass_global_to_node
        }
        self.encoder = GraphEncoder(
            EdgeBlock(Flex(MLP)(Flex.d(), latent_sizes[0])),
            NodeBlock(Flex(MLP)(Flex.d(), latent_sizes[1])),
            GlobalBlock(Flex(MLP)(Flex.d(), latent_sizes[2]))
        )

        edge_layers = [self.config['latent_size']['edge']] * self.config['latent_size']['core_edge_block_depth']
        node_layers = [self.config['latent_size']['node']] * self.config['latent_size']['core_node_block_depth']
        global_layers = [self.config['latent_size']['global']] * self.config['latent_size']['core_global_block_depth']

        self.core = GraphCore(
            AggregatingEdgeBlock(torch.nn.Sequential(
                Flex(MLP)(Flex.d(), *edge_layers),
                #                 torch.nn.Linear(latent_sizes[0], latent_sizes[0])
            )),
            AggregatingNodeBlock(torch.nn.Sequential(
                Flex(MLP)(Flex.d(), *node_layers),
                #                 torch.nn.Linear(latent_sizes[1], latent_sizes[1])
            ), Aggregator(self.config['node_block_aggregator'])),
            AggregatingGlobalBlock(Flex(MLP)(Flex.d(), *global_layers),
                                   edge_aggregator=Aggregator(self.config['global_block_to_edge_aggregator']),
                                   node_aggregator=Aggregator(self.config['global_block_to_node_aggregator'])),
            pass_global_to_edge=self.config['pass_global_to_edge'],
            pass_global_to_node=self.config['pass_global_to_node']
        )

        self.decoder = GraphEncoder(
            EdgeBlock(Flex(MLP)(Flex.d(), latent_sizes[0], latent_sizes[0])),
            NodeBlock(Flex(MLP)(Flex.d(), latent_sizes[1], latent_sizes[1])),
            GlobalBlock(Flex(MLP)(Flex.d(), latent_sizes[2]))
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

def train(
    learning_rate: float,
):
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
        batch_size=100,
        learning_rate=0.001,
        epochs=2,
    )

    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    # Your model here ...
    device = 'cuda:0'
    net = Network((254, 254, 128), (3, 3, 3), True, True)

    # Log metrics inside your training loop
    metrics = {'accuracy': accuracy, 'loss': loss}
    wandb.log(metrics)

if __name__ == '__main__':
    fire.Fire(start)