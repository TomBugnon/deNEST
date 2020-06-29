[![Travis build badge](https://img.shields.io/travis/tombugnon/denest.svg?style=flat-square&maxAge=600)](https://travis-ci.org/tombugnon/denest)
[![Codecov badge](https://img.shields.io/codecov/c/github/tombugnon/denest?style=flat-square&maxAge=600)](https://codecov.io/gh/tombugnon/denest)
[![License badge](https://img.shields.io/github/license/tombugnon/denest.svg?style=flat-square&maxAge=86400)](https://github.com/tombugnon/denest/blob/develop/LICENSE)
![Python versions badge](https://img.shields.io/pypi/pyversions/pyphi.svg?style=flat-square&maxAge=86400)

# deNEST: A declarative frontend for NEST

deNEST is a Python library for specifying networks and running simulations
using [the NEST simulator](https://nest-simulator.org).

deNEST allows the user to concisely specify large-scale networks and
simulations in hierarchically-organized declarative parameter files.

From these parameter files, a network is instantiated in NEST (layers of neurons
and stimulation devices, their connections, and recorder devices), and a
simulation is run in sequential steps ("sessions"), during which the network
parameters can be modified and the network can be stimulated, recorded, etc.

Some advantages of the declarative approach:
- Parameters and code are separated
- Simulations are easier to reason about, reuse, and modify
- Parameters are more readable and succinct
- Parameter files can be easily version controlled and diffs are smaller and more interpretable
- Clean separation between the specification of the "network" (the simulated neuronal system) and the "simulation" (structured stimulation and recording of the network), which facilitates running different experiments using the same network
- Parameter exploration is more easily automated
- The complexity of interacting with NEST is hidden, which makes some tricky operations (such as connecting a `weight_recorder`) easy

## Documentation

Documentation and tutorials can be found at <http://denest.readthedocs.io>.


## Installation

See instructions [here](https://denest.readthedocs.io/en/latest/install.html).


## Credit

We are in the process of submitting a JOSS paper describing this package.

If you use it for your research, please be so kind as to check again later and
cite our article :)
