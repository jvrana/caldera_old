import random

import numpy as np
import pytest
import torch

from experiments.utils.circuit_generator import CircuitGenerator


@pytest.mark.parametrize("n", range(2, 30, 3))
def test_init(n):
    generator = CircuitGenerator(n)


@pytest.mark.parametrize("n", range(2, 30, 3))
def test_random_circuit(n):
    generator = CircuitGenerator(n)
    circuit_generator = generator.iter_random_circuit(
        100, (1, n), annotate=True, cycles=False
    )
    for c in circuit_generator:
        pass


from pylab import plt


def test_plot_parts():
    generator = CircuitGenerator(20)
    x = np.repeat(np.expand_dims(np.linspace(0, 40, 30), 0), generator.n_parts, axis=0)
    y = generator.func(x, *tuple(np.expand_dims(generator.params.T, 2)))

    plt.plot(x.T, y.T)
    plt.show()
