[![Travis build badge](https://img.shields.io/travis/tombugnon/denest.svg?style=flat-square&maxAge=600)](https://travis-ci.org/tombugnon/denest)
[![Codecov badge](https://img.shields.io/codecov/c/github/tombugnon/denest?style=flat-square&maxAge=600)](https://codecov.io/gh/tombugnon/denest)
[![License badge](https://img.shields.io/github/license/tombugnon/denest.svg?style=flat-square&maxAge=86400)](https://github.com/tombugnon/denest/blob/develop/LICENSE)
![Python versions badge](https://img.shields.io/pypi/pyversions/pyphi.svg?style=flat-square&maxAge=86400)

<!--lint disable list-item-indent-->
<!--lint disable list-item-content-indent-->
<!--lint disable list-item-bullet-indent-->

# deNEST: A declarative frontend for NEST

deNEST is a python library for specifying networks and running simulations using
[the NEST simulator](https://nest-simulator.org).

deNEST allows the user to fully specify large scale
networks and simulation characteristics in separate, trackable and
hierarchically organized declarative parameter files.

From those parameter files, a network is instantiated in NEST (neuron layers,
simulator layers, projections amongst layers), and a simulation is run in
multiple separate steps ("sessions") before which the network can be modified.

The declarative approach facilitates version control and sharing of the
parameter files, while decoupling the "network" and "simulation" parameters
facilitates running the same network in multiple conditions.


## Usage, Examples, and API documentation


Documentation and tutorials can be found [here](http://denest.readthedocs.io)


## Installation

Instructions [here](https://denest.readthedocs.io/en/latest/install.html)

## User group

TODO

## Credit

TODO
