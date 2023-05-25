# JaxPr Visualizer

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![License Badge](https://img.shields.io/badge/License-GPL3-red)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
![Python Version](https://img.shields.io/badge/python-3.10-blue)


This project provides interactive visualizations of JAX computations.
Composite nodes can be expanded using left-click and contracted using right-click.
Space pauses the simulation.

## Usage Example

```python
import jax.numpy as jnp

from jax import make_jaxpr
from jax.core import ClosedJaxpr

import jaxprvis as jvis

def visualize_loop_unroll_jaxpr(iter: int) -> ClosedJaxpr:
    example_args = jnp.zeros((2, 2))

    def f(x, y):
        z = x
        for i in range(iter):
            z = (z + y) / x
        return z

    closed_jaxpr: ClosedJaxpr = make_jaxpr(f)(example_args, example_args)
    jvis.visualize(closed_jaxpr)
```


## Installation

<details>
<summary align="center">For Users</summary>

### Install from Pypi:
```bash
#Create and activate virtual environment (if not present yet)
python -m venv --upgrade-deps ./venv
source ./venv/bin/activate
pip install --upgrade build
# Install your JAX version
#e.g. pip install --upgrade "jax[cpu]"

pip install --upgrade jaxprvis
```
### Install from source:
```bash
# Clone
git clone https://github.com/PSilvestre/jaxprvis
cd jaxprvis

#Create and activate virtual environment
python -m venv --upgrade-deps ./venv
source ./venv/bin/activate
pip install --upgrade build

python -m build
pip install -e .

```

</details>

<details>
<summary align="center">For Developers</summary>

Similar to the above, but instead install the dev requirements.
Run the following to install:

```bash
# Clone
git clone https://github.com/PSilvestre/jaxprvis
cd CORAL

#Create and activate virtual environment
python -m venv --upgrade-deps ./venv
source ./venv/bin/activate
pip install --upgrade build

python -m build
pip install -e ".[dev]"

#Run the tests
pytest
```

</details>

