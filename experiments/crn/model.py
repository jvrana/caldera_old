from typing import List

import torch
from torch import nn

from caldera.blocks import AggregatingEdgeBlock
from caldera.blocks import AggregatingGlobalBlock
from caldera.blocks import AggregatingNodeBlock
from caldera.blocks import Aggregator
from caldera.blocks import EdgeBlock
from caldera.blocks import Flex
from caldera.blocks import GlobalBlock
from caldera.blocks import MLP
from caldera.blocks import NodeBlock
from caldera.data import GraphBatch
from caldera.models import GraphCore
from caldera.models import GraphEncoder
from caldera.utils import pairwise


class MLPBlock(nn.Module):
    """A multilayer perceptron block."""

    def __init__(self, input_size: int, output_size: int = None, dropout: float = None):
        super().__init__()
        if output_size is None:
            output_size = input_size
        args = [
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.LayerNorm(output_size),
        ]
        if dropout:
            args.append(nn.Dropout(dropout))
        self.blocks = nn.Sequential(*args)

    def forward(self, x):
        return self.blocks(x)


class MLP(nn.Module):
    """A multilayer perceptron."""

    def __init__(self, *latent_sizes: List[int], dropout: float = None):
        super().__init__()
        if dropout:
            self.blocks = nn.Sequential(
                *[MLPBlock(n1, n2, dropout) for n1, n2 in pairwise(latent_sizes)]
            )
        else:
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
        dropout: float = None,
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
            EdgeBlock(Flex(MLP)(Flex.d(), latent_sizes[0], dropout=dropout)),
            NodeBlock(Flex(MLP)(Flex.d(), latent_sizes[1], dropout=dropout)),
            GlobalBlock(Flex(MLP)(Flex.d(), latent_sizes[2], dropout=dropout)),
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
                    Flex(MLP)(Flex.d(), *edge_layers, dropout=dropout),
                    #                 torch.nn.Linear(latent_sizes[0], latent_sizes[0])
                )
            ),
            AggregatingNodeBlock(
                torch.nn.Sequential(
                    Flex(MLP)(Flex.d(), *node_layers, dropout=dropout),
                    #                 torch.nn.Linear(latent_sizes[1], latent_sizes[1])
                ),
                Aggregator(self.config["node_block_aggregator"]),
            ),
            AggregatingGlobalBlock(
                Flex(MLP)(Flex.d(), *global_layers, dropout=dropout),
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
            EdgeBlock(
                Flex(MLP)(Flex.d(), latent_sizes[0], latent_sizes[0], dropout=dropout)
            ),
            NodeBlock(
                Flex(MLP)(Flex.d(), latent_sizes[1], latent_sizes[1], dropout=dropout)
            ),
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
