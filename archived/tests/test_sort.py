import pytest

import numpy as np
import torch
import torch.nn as nn
import time
import itertools

torch.set_num_threads(4)

from pyrographnets.models import GraphEncoder, GraphCore
from pyrographnets.blocks import MLP, EdgeBlock, NodeBlock, GlobalBlock, AggregatingEdgeBlock, AggregatingGlobalBlock, AggregatingNodeBlock, Flex, Aggregator
from pyrographnets.data import GraphData, GraphBatch, GraphDataLoader


class EncoderProcessDecoder(torch.nn.Module):

    def __init__(self):
        super().__init__()

        layers = (16, 16, 1)

        self.encoder = GraphEncoder(
            EdgeBlock(MLP(1, layers[0])),
            NodeBlock(MLP(1, layers[1])),
            GlobalBlock(MLP(1, layers[2]))
        )

        self.core = GraphCore(
            AggregatingEdgeBlock(torch.nn.Sequential(
                Flex(MLP)(Flex.d(), layers[0]),
                #                 torch.nn.Linear(layers[0], layers[0])
            )),
            AggregatingNodeBlock(torch.nn.Sequential(
                Flex(MLP)(Flex.d(), layers[1]),
                #                 torch.nn.Linear(layers[1], layers[1])
            ), Aggregator('mean')),
            AggregatingGlobalBlock(Flex(MLP)(Flex.d(), layers[2]), Aggregator('add'), Aggregator('add'))
        )

        self.decoder = GraphEncoder(
            EdgeBlock(Flex(MLP)(Flex.d(), layers[0])),
            NodeBlock(Flex(MLP)(Flex.d(), layers[1])),
            GlobalBlock(Flex(MLP)(Flex.d(), layers[2]))
        )

        self.output_transform = GraphEncoder(
            EdgeBlock(Flex(torch.nn.Linear)(Flex.d(), 2)),
            NodeBlock(Flex(torch.nn.Linear)(Flex.d(), 2)),
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

        outputs = []
        for _ in range(steps):
            # core processing step
            e = torch.cat([latent0.e, e], dim=1)
            x = torch.cat([latent0.x, x], dim=1)
            g = torch.cat([latent0.g, g], dim=1)
            data = GraphBatch(x, e, g, edges, node_idx, edge_idx)
            e, x, g = self.core(data)

            # decode
            data = GraphBatch(x, e, g, edges, node_idx, edge_idx)
            _e, _x, _g = self.decoder(data)
            decoded = GraphBatch(_x, _e, _g, edges, node_idx, edge_idx)

            # transform
            _e, _x, _g = self.output_transform(decoded)
            outputs.append(GraphBatch(_x, _e, _g, edges, node_idx, edge_idx))
        return outputs


def graph_data_from_list(input_list):
    """Takes a list with the data, generates a fully connected graph with values of the list as nodes

    Parameters
    ----------
    input_list: list
        list of the numbers to sort

    Returns
    -------
    graph_data: dict
        Dict of entities for the list provided.
    """
    connectivity = torch.tensor(
        [el for el in itertools.product(range(len(input_list)), repeat=2)],
        dtype=torch.long,
    ).t()
    vdata = torch.tensor([[v] for v in input_list], dtype=torch.float)
    edata = torch.zeros(connectivity.shape[1], 1, dtype=torch.float)
    return (vdata, edata, connectivity)


def edge_id_by_sender_and_receiver(connectivity, sid, rid):
    """Get edge id from the information about its sender and its receiver.

    Parameters
    ----------
    metadata: list
        list of pgn.graph.Edge objects
    sid: int
        sender id
    rid: int
        receiver id

    Returns
    -------

    """
    return (connectivity[0, :] == sid).mul(connectivity[1, :] == rid).nonzero().item()


def create_target_data(vdata, edata, connectivity):
    """ Generate target data for training

    Parameters
    ----------
    input_data: list
        list of data to sort

    Returns
    -------
    res: dict
        dict of target graph entities
    """
    # two nodes might have true since they might have similar values
    min_val = vdata.min()

    # [prob_true, prob_false]
    target_vertex_data = torch.tensor(
        [[1.0, 0.0] if v == min_val else [0.0, 1.0] for v in vdata],
        dtype=torch.double
    )

    sorted_ids = vdata.argsort(dim=0).flatten()
    target_edge_data = torch.zeros(edata.shape[0], 2, dtype=torch.double)
    for sidx, sid in enumerate(sorted_ids):
        for ridx, rid in enumerate(sorted_ids):
            eid = edge_id_by_sender_and_receiver(connectivity, sid, rid)
            # we look for exact comparison here since we sort
            if sidx < len(sorted_ids) - 1 and ridx == sidx + 1:
                target_edge_data[eid][0] = 1.0
            else:
                target_edge_data[eid][1] = 1.0

    return target_vertex_data, target_edge_data


def generate_graph_batch(n_examples, sample_length):
    """ generate all of the training data

    Parameters
    ----------
    n_examples: int
        Num of the samples
    sample_length: int
        Length of the samples.
        # TODO we should implement samples of different lens as in the DeepMind example.
    Returns
    -------
    res: tuple
        (input_data, target_data), each of the elements is a list of entities dicts
    """

    input_data = [
        graph_data_from_list(np.random.uniform(size=sample_length))
        for _ in range(n_examples)
    ]
    target_data = [create_target_data(v, e, conn) for v, e, conn in input_data]

    return input_data, target_data


def batch_loss(outs, targets, criterion, batch_size, core_steps):
    """get the loss for the network outputs

    Parameters
    ----------
    outs: list
        list of lists of the graph network output, time is 0-th dimension, batch is 1-th dimension
    targets: list
        list of the graph entities for the expected output
    criterion: torch._Loss object
        loss to use
    Returns
    -------
    loss: float
        Shows how good your mode is.
    """
    loss = 0
    vsize = targets[0].shape[0] // batch_size
    esize = targets[1].shape[0] // batch_size
    for out in outs:
        for i in range(batch_size):
            loss += criterion(
                out[0][i * vsize : (i + 1) * vsize],
                targets[0][i * vsize : (i + 1) * vsize],
            )
        for i in range(batch_size):
            loss += criterion(
                out[1]["default"][i * esize : (i + 1) * esize],
                targets[1][i * esize : (i + 1) * esize],
            )

    return loss / core_steps / batch_size

def test_generate_graph_batch():


    all_train_data = []
    all_target_data = []

    for train_data, test_data in zip(*generate_graph_batch(10, 10)):

        test_node_attr, test_edge_attr, connectivity = train_data
        target_node_attr, target_edge_attr = test_data

        train_graph_data = GraphData(test_node_attr, test_edge_attr, torch.zeros(1, 1), connectivity)
        test_graph_data = GraphData(target_node_attr, target_edge_attr, torch.zeros(1, 1), connectivity)

        all_train_data.append(train_graph_data)
        all_target_data.append(test_graph_data)

    loader = GraphDataLoader(list(zip(all_train_data, all_target_data)), batch_size=32, shuffle=True)

    model = EncoderProcessDecoder()

    optimiser = torch.optim.Adam(lr=0.001, params=model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    for train_batch, target_batch in loader:

        optimiser.zero_grad()




#
# def run(args):
#     device = torch.device("cpu")
#     if args.cuda and torch.cuda.is_available():
#         device = torch.device("cuda")
#
#     model = EncoderCoreDecoder(
#         args.core_steps,
#         enc_vertex_shape=(1, 16),
#         core_vertex_shape=(32, 16),
#         dec_vertex_shape=(16, 16),
#         out_vertex_size=2,
#         enc_edge_shape=(1, 16),
#         core_edge_shape=(32, 16),
#         dec_edge_shape=(16, 16),
#         out_edge_size=2,
#         device=device,
#     )
#
#     optimiser = torch.optim.Adam(lr=0.001, params=model.parameters())
#     criterion = nn.BCEWithLogitsLoss()
#
#     train_input, train_target = generate_graph_batch(
#         args.num_train, sample_length=args.sample_length
#     )
#
#     eval_input, eval_target = generate_graph_batch(
#         args.num_train, sample_length=args.sample_length
#     )
#     train_input = list(batch_data(train_input))
#     train_target = [
#         torch.cat([el[0] for el in train_target]),
#         torch.cat([el[1] for el in train_target]),
#     ]
#     eval_target = [
#         torch.cat([el[0] for el in eval_target]),
#         torch.cat([el[1] for el in eval_target]),
#     ]
#     eval_input = list(batch_data(eval_input))
#     # if args.cuda and torch.cuda.is_available():
#     #     train_input[0] = train_input[0].to("cuda")
#     #     for k in train_input[1]:
#     #         train_input[1][k] = train_input[1][k].to("cuda")
#     #         train_input[2][k] = train_input[2][k].to("cuda")
#     #
#     #     eval_input[0] = eval_input[0].to("cuda")
#     #     for k in eval_input[1]:
#     #         eval_input[1][k] = eval_input[1][k].to("cuda")
#     #         eval_input[2][k] = eval_input[2][k].to("cuda")
#     #     train_target[0] = train_target[0].to("cuda")
#     #     train_target[1] = train_target[1].to("cuda")
#     #     eval_target[0] = eval_target[0].to("cuda")
#     #     eval_target[1] = eval_target[1].to("cuda")
#     #     model.to("cuda")
#     #
#     # for e in range(args.epochs):
#     #
#     #     st_time = time.time()
#     #     train_outs = model(*train_input, output_all_steps=True)
#     #     train_loss = batch_loss(
#     #         train_outs, train_target, criterion, args.num_train, args.core_steps
#     #     )
#     #     optimiser.zero_grad()
#     #     train_loss.backward()
#     #
#     #     optimiser.step()
#     #
#     #     end_time = time.time()
#     #     if args.verbose:
#     #         print("Epoch {} is done. {:.2f} sec spent.".format(e, end_time - st_time))
#     #
#     #     if e % args.eval_freq == 0 or e == args.epochs - 1:
#     #         model.eval()
#     #         eval_outs = model(*eval_input, output_all_steps=True)
#     #         eval_loss = batch_loss(
#     #             eval_outs, eval_target, criterion, args.num_eval, args.core_steps
#     #         )
#     #         print(
#     #             "Epoch %d, mean training loss: %f, mean evaluation loss: %f."
#     #             % (
#     #                 e,
#     #                 train_loss.item() / args.num_train,
#     #                 eval_loss.item() / args.num_eval,
#     #             )
#     #         )
#     #
#     #         args.writer.add_scalar('training loss',
#     #                         train_loss.item() / args.num_train, e)
#     #
#     #         args.writer.add_scalar('eval loss',
#     #                         eval_loss.item() / args.num_eval, e)
#     #         # plot_test(model, args.sample_length, args.cuda)
#     #         model.train()
#
# def test_main(new_writer):
#     class Args():
#         pass
#
#     args = Args()
#     args.sample_length = 10
#     args.num_train = 100
#     args.num_eval = 30
#     args.cuda = False
#     args.eval_freq = 1
#     args.core_steps = 10
#     args.verbose = True
#     args.epochs = 500
#     args.writer = new_writer('test_sort')
#
#     run(args)
