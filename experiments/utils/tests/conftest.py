import random
import numpy
import torch
import pytest

@pytest.fixture(autouse=True)
def set_random_seed():
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)

from os.path import join, dirname, abspath
from pylab import plt
import functools


@pytest.fixture(autouse=True)
def patch_plt_show(request, monkeypatch):
    """Patches the plt.show() method with a save fig method in 'out'."""
    out = join(dirname(abspath(__file__)), "out")

    def _save_fig(filename=None):
        if filename is None:
            filename = "{}_{}.png".format(request.node.name, request.param_index)
        plt.savefig(join(out, filename), format="png", dpi=50)

    def _save_self_fig(_, filename=None):
        return functools.partial(_save_fig, filename=filename)

    print("PATCHING")
    monkeypatch.setattr(plt, "show", _save_fig)
    monkeypatch.setattr(plt.Figure, "show", _save_self_fig)
    print(plt.show)
    return _save_fig