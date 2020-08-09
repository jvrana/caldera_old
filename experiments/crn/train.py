import sys

sys.path.insert(0, "../..")
import sys
import os

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import wandb
import torch
from pyrographnets.data import GraphBatch
from typing import Tuple
from typing import Dict
from data import generate_data, create_loader, CircuitGenerator
from model import Network
from summary import loader_summary
from tqdm.auto import tqdm
from pyrographnets.utils import _first
import pylab as plt
import pandas as pd
import seaborn as sns


def create_data(
    train_size: int,
    dev_size: int,
    n_parts: int,
    train_part_range: Tuple[int, int],
    dev_part_range: Tuple[int, int],
) -> Dict:
    circuit_gen = CircuitGenerator(n_parts)

    data = generate_data(
        circuit_gen, train_size, train_part_range, dev_size, dev_part_range
    )

    return data, circuit_gen


def to(batch, device):
    return GraphBatch(
        batch.x.to(device),
        batch.e.to(device),
        batch.g.to(device),
        batch.edges.to(device),
        batch.node_idx.to(device),
        batch.edge_idx.to(device),
    )


def plot(target_data, out):
    x = target_data.x.cpu().detach().numpy().flatten()
    y = out.x.cpu().detach().numpy().flatten()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    df = pd.DataFrame({"x": x, "y": y})
    ax = sns.scatterplot("x", "y", data=df, ax=ax)
    ax.set_ylim(-5, 20)
    ax.set_xlim(-5, 20)
    return ax, fig


def train(**kwargs):
    """
    This is the documentation for the function.

    :param learning_rate:
    :return:
    """

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
        pass_global_to_node=True,
        pass_global_to_edge=True,
        batch_size=512,
        learning_rate=1e-3,
        weight_decay=1e-2,
        epochs=100,
        train_size=10000,
        dev_size=2000,
        log_every_epoch=10,
        dropout=0.2,
    )

    # Pass your defaults to wandb.init

    hyperparameter_defaults.update(kwargs)
    wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    # update config

    # Your model here ...
    device = "cuda:0"
    net = Network(
        (config["latent_size_0"], config["latent_size_1"], config["latent_size_2"]),
        (config["latent_depth_0"], config["latent_depth_1"], config["latent_depth_2"]),
        pass_global_to_edge=config["pass_global_to_edge"],
        pass_global_to_node=config["pass_global_to_node"],
        dropout=config["dropout"],
    )

    # create your data
    data, gen = create_data(
        config["train_size"], config["dev_size"], 20, (2, 6), (8, 20)
    )
    train_loader = create_loader(gen, data["train"], config["batch_size"], shuffle=True)
    eval_loader = create_loader(gen, data["train/dev"], None, shuffle=False)

    wandb.config.update(
        {
            "train": {"loader_summary": loader_summary(train_loader)},
            "eval": {"loader_summary": loader_summary(eval_loader)},
        }
    )

    with torch.no_grad():
        batch, _ = _first(train_loader)
        print(batch.shape)
        out = net(batch, 3)
    net.to(device)

    assert list(net.parameters())
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = torch.nn.MSELoss()
    log_every_epoch = config["log_every_epoch"]
    for epoch in tqdm(range(config["epochs"])):
        net.train()

        running_loss = 0.0
        for batch_idx, (train_data, target_data) in enumerate(train_loader):
            train_data.contiguous()
            target_data.contiguous()
            # TODO: why clone?
            #             target_data = target_data.clone()
            train_data = to(train_data, device)
            target_data = to(target_data, device)
            out = net(train_data, config["processing_steps"])
            assert out[-1].x.shape == target_data.x.shape

            optimizer.zero_grad()

            # TODO: the scale of the loss is proporational to the processing steps and batch_size, should this be normalized???
            loss = torch.tensor(0.0).to(device)
            for _out in out:
                loss += loss_fn(_out.x, target_data.x)
            loss = loss / (target_data.x.shape[0] * 1.0 * len(out))

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        if epoch % log_every_epoch == 0:
            wandb.log({"train_loss": running_loss}, step=epoch)

        if epoch % log_every_epoch == 0:
            eval_loss = 0.0
            net.eval()
            with torch.no_grad():
                for eval_data, eval_target in eval_loader:
                    eval_data = to(eval_data, device)
                    eval_target = to(eval_target, device)
                    eval_outs = net(eval_data, config["processing_steps"])
                    # only take loss from last output
                    eval_loss += (
                        loss_fn(eval_outs[-1].x, eval_target.x)
                        / eval_outs[-1].x.shape[0]
                    )
                wandb.log(
                    {"eval_loss": eval_loss / eval_data.x.shape[0] * 1000}, step=epoch
                )
                wandb.log(
                    {
                        "edge_attr": wandb.Histogram(eval_outs[-1].e.cpu()),
                        "node_attr": wandb.Histogram(eval_outs[-1].x.cpu()),
                    },
                    step=epoch,
                )

        if epoch % log_every_epoch * 2 == 0:
            net.eval()
            with torch.no_grad():
                for eval_data, eval_target in eval_loader:
                    eval_data = eval_data.to(device)
                    eval_target = to(eval_target, device)
                predicted = net(eval_data, config["processing_steps"])[-1]
                ax, fig = plot(eval_target, predicted)
                wandb.log({"chart": fig}, step=epoch)


if __name__ == "__main__":
    train()
