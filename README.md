[![Travis build badge](https://img.shields.io/travis/tombugnon/denest.svg?style=flat-square&maxAge=600)](https://travis-ci.org/tombugnon/denest)
[![Codecov badge](https://img.shields.io/codecov/c/github/tombugnon/denest?style=flat-square&maxAge=600)](https://codecov.io/gh/tombugnon/denest)
[![License badge](https://img.shields.io/github/license/tombugnon/denest.svg?style=flat-square&maxAge=86400)](https://github.com/tombugnon/denest/blob/develop/LICENSE)
![Python versions badge](https://img.shields.io/pypi/pyversions/pyphi.svg?style=flat-square&maxAge=86400)

<!--lint disable list-item-indent-->
<!--lint disable list-item-content-indent-->
<!--lint disable list-item-bullet-indent-->

**NB**: Although it is public, this repository is not published yet,
and is under active development until the first release... soon :)

**Please don't use it for your work until it is stable and citable**... But please feel free to open issues or contact us if you have any suggestions to make this package better !

Upcoming changes before the v1.0 release:
  - Flexible setting of network and layer states from arrays
  - Improve the API for interactive use
  - Support all recorder and generator models
  - Support definition of connections from arrays/tables (maybe)
  - Tutorials and proper documentation
  - Improve test suite

# deNEST: A declarative frontend for NEST


deNEST is a python library for specifying networks and running simulations using
the NEST simulator (nest-simulator.org ).

deNEST allows the user to fully specify large scale
networks and simulation characteristics in separate, trackable and
hierarchically organized declarative parameter files.

From those parameter files, a network is instantiated in NEST (neuron layers,
simulator layers, connections amongst layers), and a simulation is run in
multiple separate steps ("sessions") before which the network can be modified.

The declarative approach facilitates version control and sharing of the
parameter files, while decoupling the "network" and "simulation" parameters
facilitates running the same network in multiple conditions.

## Installation

### Docker

A Docker image is provided with NEST 2.20 installed, based on
[nest-docker](https://github.com/nest/nest-docker).

1. From within the repo, build the image:
   ```bash
   docker build --tag denest .
   ```
2. Run an interactive container:
   ```bash
   docker run \
     -it \
     --name denest_simulation \
     --volume $(pwd):/opt/data \
     --publish 8080:8080 \
     denest \
     /bin/bash
   ```
3. Install deNEST within the container:
   ```bash
   pip install -e .
   ```
4. Use deNEST from within the container.

For more information on how to use the NEST Docker image, see
[nest-docker](https://github.com/nest/nest-docker).

### Local

1. Install NEST > v2.14.0 by following the instructions at <http://www.nest-simulator.org/installation/>
2. Set up a Python 3 environment and install deNEST with:
   ```python
   pip install "git+https://github.com/TomBugnon/deNEST@develop"
   ```


## Overview

#### Network

- We call **network** a full network in NEST, consisting of layers of units with
  specific models, connections of specific types with specific synapse models
  amongst these layers, population recorders (multimeters, spike detectors) and
  connection recorders (weight recorder).
  - The full network is represented in deNEST by the `Network()` class

- New NEST **models** (**neuron and generator model**, **synapse model** or
  **recorder model**) can be specified with arbitrary parameters. During network
  creation, models with specific parameters are created in NEST using a
  ``nest.CopyModel()`` or a ``nest.SetDefaults`` call.
  - Synapse models are represented by the ``SynapseModel`` class in deNEST. All
    other models are represented by the ``Model`` class.
  - Neuron and generator models are specified as leaves of the
    `network/neuron_models` parameter subtree (see section below)
  - Synapse models are specified as leaves of the `network/synapse_models`
    parameter subtree (see "Network parameters" section below)
  - Recorder models are specified as leaves of the `network/recorder_models`
    parameter subtree (see "Network parameters" section below)

- A **layer** is a NEST topological layer, created with a ``tp.CreateLayer``
  call in NEST. We call **population** all the nodes of the same model
  within a layer.
  - Layers can be of the type `InputLayer` when they are composed of generators.
    Input layers are automatically duplicated to contain parrot neurons, so that
    recording of generators' activity is possible.  Additionally, `InputLayer`s
    support setting the state of stimulators (see "session" description).
  - Layers are represented by the ``Layer`` or ``InputLayer`` class in deNEST
  - Layers are specified as leaves of the `network/layers` parameter
    subtree (see "Network parameters" section below)

- A **connection model** is a template specifying parameters passed to
  ``tp.ConnectLayers``, and that individual connections amongst populations can
  inherit from.
  - Connection models are represented by the ``ConnectionModel`` class in deNEST.
  - Connection models are specified as leaves of the `network/connection_models`
    parameter subtree (see "Network parameters" section below)

- A **connection** is an individual projection between layers or populations,
  created with a ``tp.ConnectLayers`` call. The parameters passed to
  ``tp.ConnectLayers`` are those of the "connection model" of the specific
  connection.
  - The list of all individual connections within the network is specified in
    the ``'connections'`` parameter of the `network/topology` parameter subtree
    (see "Network parameters" section below).

- A **population recorder** is a recorder connected to all the nodes of a given
  population. A **connection recorder** is a recorder connected to all the
  synapses of a given connection. Recorders with arbitrary parameters can be
  defined by creating "recorder models". However, currently only recorders based
  on the 'multimeter', 'spike_detector' and 'weight_recorder' NEST models are
  supported.
  - population and connection recorders are represented by the
    ``PopulationRecorder``  and ``ConnectionRecorder`` classes in
    deNEST.
  - The list of all population recorders and connection recorders are specified
    in the ``'population_recorders'`` and ``'connection_recorders'`` parameters
    of the `network/recorders` parameter subtree (See "Network parameters"
    section below)


#### Simulation

- A **session model** is a template specifying parameters inherited by
  individual sessions.
  - session models are specified as leaves of the `session_models` parameter
    subtree (see "Simulation parameters" section below)

- A **session** is a period of simulation of a network with specific inputs
  and parameters, and corresponds to a single ``nest.Simulate()`` call. The
  parameters used by a given session are inherited from its session model.
  - A session's parameters define the operations that may be performed before
    running it:
    - Modifying the state of some units within a popultion
    - Modifying the state of some synapses of a certain connection (TODO)
    - Setting the state of generators within ``InputLayer`` layers from arrays
    - Deactivate the recorders for that session.
  - Individual sessions are represented by the ``Session`` object in deNEST.
    (see "Simulation parameters" section below)

- A **simulation** is a full experiment. It is represented by the `Simulation()`
  object in deNEST, which contains a `Network` object and a list of `Session`
  objects.
  - The list of sessions run during a simulation is specified by the
    ``sessions`` parameter of the ``simulation`` parameter subtree (eg:
    sessions: ``['warmup', 'noise', 'grating', 'noise', 'grating']``) (see
    "Simulation parameters" section below)

#### Overview of a full simulation

A full deNEST simulation consists of the following steps:

1. **Initialize simulation** (``Simulation.__init__``)

    1. **Initialize kernel**: (``Simulation.init_kernel``)

        1. Set NEST kernel parameters
        2. Set random seeds for python's `random` module and NEST's random
        generator.

    1. **Create network**:

        1. Initialize the network objects (``Network.__init__``)
        2. Create the objects in NEST (``Network.__create__``)

    3. **Initialize the sessions** (``Session.__init__``)

    2. **Save the simulation's metadata**

        - Create the output directory
        - Save parameters
        - Save git hash
        - Save session times
        - Save network metadata (TODO)
        - Save session metadata (TODO)

2. **Run the simulation** (``Simulation.__run__``)

    1. Run each session in turn: (``Session.__run__``)

        1. Initialize session (``Session.initialize``)
            - (Possibly) reset the network
            - (Possibly) inactivate recorders for the duration of the session
            - (Possibly) Change some of the network's parameters:
              1. Change neuron parameters (``Network.change_unit_states``)
              1. Change synapse parameters (``Network.change_synapse_states``)
            - Set InputLayer's state from input arrays

        2. Call `nest.Simulate()`.

## Running deNEST

- From Python (_e.g._ in a Jupyter notebook):

  - Using the ``Simulation`` object to run the simulation step by step:

    ```python
    import denest

    # Path to the parameter files to use
    params_path = 'params/tree_paths.yml'

    # Override some parameters loaded from the file
    overrides = [

      # Maybe change the nest kernel's settings ?
      {'kernel': {'nest_params': {'local_num_threads': 20}}},

      # Maybe change a parameter for all the connections at once ?
      {'network': {'connection_models': {'nest_params': {
          'allow_autapses': true
      }}}},
    ]

    # Load the parameters
    params = denest.load_trees(params_path, *overrides)

    # Initialize the simulation
    sim = denest.Simulation(params, output_dir='output')

    # Run the simulation (runs all the sessions)
    sim.run()
    ```

  - Using the ``denest.run()`` function to run the full simulation at once:

    ```python
    import denest

    # Path to the parameter files to use
    params_path = 'params/tree_paths.yml'

    # Override parameters
    overrides = []

    denest.run(params_path, *overrides, output_dir=None)
    ```

- From the command line:

    ```bash
    python -m denest <tree_paths.yml> [-o <output_dir>]
    ```

## Loading simulation outputs

All simulation outputs, including the raw data from NEST, are saved in the directory specified by the ``output_dir`` parameter.

In particular,

- the ``data`` subdirectory contains the raw NEST data, and metadata files for each of the recorders. Each recorder is connected to a single population.

- the start and end time of each of the run sessions is saved at
  ``OUTPUT_DIR/session_times.yml``

The `denest.io.load` module contains utility functions to load the simulation outputs:

```python
import denest.io.load

from pathlib import Path

OUTPUT_DIR = Path('./output')  # Path to the simulation output directory

# Load the start and end time for each session
session_times = denest.io.load.load_session_times(OUTPUT_DIR)
print(session_times)  # {<session_name>: (<session_start>, <session_end>)}


## Load data from a specific recorder.

# All we need is the path to its metadata file
recorder_metadata_path = OUTPUT_DIR/'data/multimeter_l1_l1_exc.yml'

# All relevant information about this recorder and the population it's
# connected to are contained in its metadata file
recorder_metadata = denest.io.load.load_yaml(recorder_metadata_path)
print(f'Metadata keys: {recorder_metadata.keys()}')

# We can load the raw data as pandas dataframe
df = denest.io.load.load(recorder_metadata_path)
print(df[0:5])


## Load data for all recorders:

all_recorder_metadata_paths = denest.io.load.metadata_paths(OUTPUT_DIR)
for metadata_path in all_recorder_metadata_paths:
  print(f'Recorder: {metadata_path.name}')
  print(f'{denest.io.load.load(metadata_path)[0:5]}\n')
```

## Defining parameters

All parameters used by deNEST are specified in tree-like yaml files which are
converted to ``ParamsTree`` objects.

In this section, we describe the ``ParamsTree`` objects, the expected structure
of the full parameter tree interpreted by deNEST, and the expected formats and
parameters of each of the subtrees that define the various aspects of the
network and simulation.

#### Main parameter file

To facilitate defining parameters in separate files, ``denest.run`` and
``denest.load_trees`` take as input a path to a yaml file
 containing the relative paths of the tree-like yaml files to merge
so as to define the full parameter tree (see the ``params/tree_paths.yml`` file)


#### The ``ParamsTree`` class


The ``ParamsTree`` class is instantiated from tree-like nested dictionaries. At
each node, two reserved keys contain the node's data (called ``'params'`` and
``'nest_params'``). All the other keys are interpreted as named children nodes.

The ``'params'`` key contains data interpreted by deNEST, while the
``'nest_params'`` key contains data passed to NEST without modification.

The ``ParamsTree`` class offers a tree structure with two useful
characteristics:

- **Hierarchical inheritance of ancestor's data**: This provides a concise way
  of defining data for nested scopes. Data common to all leaves may be specified
  once in the root node, while more specific data may be specified further down
  the tree. Data lower within the tree overrides data higher in the tree.
  Ancestor nodes' ``params`` and ``nest_params`` are inherited independently.

- **(Horizontal) merging of trees**: ``ParamsTree`` objects can be merged
  horizontally. During the merging of multiple params trees, the  contents of
  the ``params`` and ``nest_params`` data keys of nodes at the same relative
  position are combined. This allows **splitting the deNEST parameter trees in
  separate files for convenience**, and **overriding the data of a node anywhere
  in the tree while preserving hierarchical inheritance**


##### An example parameter tree

Below is an example of a YAML file with a tree-like structure that can be loaded
and represented by the `ParamsTree` class:

```yaml
network:
  neuron_models:
    ht_neuron:
      params:                     # params common to all leaves
        nest_model: ht_neuron
      nest_params:                # nest_params common to all leaves
        g_KL: 1.0
      cortical_excitatory:
        nest_params:
          tau_spike: 1.75
          tau_m: 16.0
        l1_exc:                   # leaf
        l2_exc:                   # leaf
          nest_params:
            g_KL: 2.0     # Overrides ancestor's value
      cortical_inhibitory:
        nest_params:
          tau_m: 8.0
        l1_inh:                   # leaf
```

This file can be loaded into a ParamsTree structure. The leaves of the resulting
ParamsTree and their respective data (``params`` and ``nest_params``) are as
follows. Note the inheritance and override of ancestor data. The nested format
above is more compact and less error prone when there is a lot of shared
parameters between leaves.

```yaml
l1_exc:
  params:
    nest_model: ht_neuron
  nest_params:
    g_KL: 1.0
    tau_spike: 1.75
    tau_m: 16.0
l2_exc:
  params:
    nest_model: ht_neuron
  nest_params:
    g_KL: 2.0
    tau_spike: 1.75
    tau_m: 16.0
l1_inh:
  params:
    nest_model: ht_neuron
  nest_params:
    g_KL: 1.0
    tau_m: 8.0
```


#### Full parameter tree: expected structure

All the aspects of the overall simulation are specified in specific named
subtrees.

The overall `ParamsTree` passed to ``denest.Simulation()`` is expected to have no
data and the following ``ParamsTree`` children

  - ``simulation`` (``ParamsTree``). Defines input and output
      paths, and the simulation steps performed. The following
      parameters (`params` field) are recognized:

      - ``output_dir`` (str): Path to the output directory
          (default 'output').
      - ``input_dir`` (str): Path to the directory in which input files are
          searched for for each session. (default 'input')
      - ``sessions`` (list(str)): Order in which sessions are
          run. Elements of the list should be the name of
          session models defined in the ``session_models``
          parameter subtree (default [])

  - ``kernel`` (``ParamsTree``): Used for NEST kernel
      initialization. Refer to ``Simulation.init_kernel`` for a
      description of kernel parameters.

  - ``session_models`` (``ParamsTree``): Parameter tree, the
    leaves of which define session models. Refer to ``Sessions``
    for a description of session parameters.

  - ``network`` (``ParamsTree``): Parameter tree defining the
    network in NEST. Refer to `Network` for a full description of
    network parameters.


#### Network parameters

All network parameters are specified in the ``network`` subtree, used to
initialize the ``Network()`` object.

The ``network`` subtree should have no data, and the following ``ParamsTree``
children are expected:

  - ``neuron_models`` (``ParamsTree``). Parameter tree, the leaves
      of which define neuron models. Each leave is used to
      initialize a ``Model`` object

  - ``synapse_models`` (``ParamsTree``). Parameter tree, the
      leaves of which define synapse models. Each leave is used to
      initialize a ``SynapseModel`` object

  - ``layers`` (``ParamsTree``). Parameter tree, the leaves of
      which define layers. Each leave is used to initialize  a
      ``Layer`` or ``InputLayer`` object depending on the value of
      their ``type`` ``params`` parameter.

  - ``connection_models`` (``ParamsTree``). Parameter tree, the
      leaves of which define connection models. Each leave is used
      to initialize a ``ConnectionModel`` object.

  - ``recorder_models`` (``ParamsTree``). Parameter tree, the
      leaves of which define recorder models. Each leave is used
      to initialize a ``Model`` object.

  - ``topology`` (``ParamsTree``). ``ParamsTree`` object without
      children, the ``params`` of which may contain a
      ``connections`` key specifying all the individual
      population-to-population connections within the network as a
      list. ``Connection`` objects  are created from the
      ``topology`` ``ParamsTree`` object by the
      ``Network.build_connections`` method. Refer to this method
      for a description of the ``topology`` parameter.

  - ``recorders`` (``ParamsTree``). ``ParamsTree`` object without
      children, the ``params`` of which may contain a
      ``population_recorders`` and a ``connection_recorders`` key
      specifying all the network recorders. ``PopulationRecorder``
      and ``ConnectionRecorder`` objects  are created from the
      ``recorders`` ``ParamsTree`` object by the
      ``Network.build_recorders`` method. Refer to this
      method for a description of the ``recorders`` parameter.


### Inputs to the network

TODO
