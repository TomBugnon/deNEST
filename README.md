# NETS: Network Simulations (in NEST)

This project is a object-oriented front-end to the NEST simulation kernel (
nest-simulator.org ) which allows the user to fully specify large scale
networks and simulation characteristics in separate, trackable and
hierarchically organized parameter files. From those parameter files, NETS
instantiates in NEST the actual network (nodes, connections etc), and runs and
saves data for a particular simulation (stimulus, duration, parameter changes,
etc).

This declarative approach allows a decoupling of the network and simulation
parameters from the actual code and facilitates version control and sharing of
the parameter files.

Under the hood, NETS instantiates classes that represent specific aspects of the full network (layers, connections, ...) or of the simulation. This classes take as inputs subtrees of the whole parameter tree, translate those parameters to parameters understandable by NEST, instantiates the NEST object, etc. The main classes, which map to parameter subtrees, are described below.


## Definitions and objects

#### Network

**IMPORTANT**: We make the distinction between the "independent" parameters, that are specified in the parameter files, and the "dependent" parameters that are actually given to NEST during object instantiation. Please read carefully the documentation to know how parameters are interpreted, and possibly translated, by NETS.

- A **model** is a NEST model (neuron, synapse, generator or recorder) with a
  specific name and set of parameters. All models are copied from their
  respective "NEST model" using `nest.Copy()` with their specific parameters, or
  change the default parameters of their "NEST model" if they have the same
  name. NEST models can be either imported from external modules or built-in.
  The `model` NETS parameters specify the 
    - NB: Multiple models can have different names but the same NEST model and
    parameters.
    - Neuron and generator models are specified in the `network/neuron_models`
      parameter subtree.
    - Synapse models are specified in the `network/synapse_models` parameter
      subtree.
    - Recorder models are specified in the `network/recorders` parameter
      subtree.


- A **layer** is a NEST topological layer. A layer is akin to a grid of a
  certain dimension (nrows x ncols) and is composed of a given number of nodes
  at each grid _location_. Nodes at each location can be from different models.
  - NB: All units represented by NETS are organized in layers ! There can be no
    "floating" node, but there can be layers consisting in a single node.
  - NB: Only 2D layers are currently supported in NETS. Since there can be
    multiple units of the same population at the same grid location, unit
    locations are indexed as (row x col x unit_index) in the output.
  - Layers can be of the type `InputLayer` when they are composed of generators.
    (Think of the `InputLayer` as the retina.) Input layers are automatically
    duplicated to contain parrot neurons, so that recording of generators'
    activity is possible (via parrot neurons).  Additionally, the state of
    generators can be changed before each session to reflect the forecoming
    input stimulus (see "session" description).
  - Layers are specified in the `network/layers` parameter
    subtree.

- A **population** is a set of nodes of the same model within a specific layer.
  In practice, populations are `(layer, neuron_model)` tuples. 
  - Connections are typically defined between layers or populations (but not at
    a finer grain).
  - Each **recorder** created in NETS, and in NEST, maps to a specific
    population, so that it is possible to decide to record (or not) each population independently, and to
    record different variables for different populations (note, however, that it
    is not possible to record only part of a population.).
  - **IMPORTANT**: The layer parameters fully specify the populations of a
    network. The `network/populations` parameter subtree, the nodes of which are
    populations (rather than layers), is used exclusively to create or not
    recorders for each population. Note that this parameter subtree is redundant
    with the layer parameter subtree, and that both need to be consistent.


- We call **connection model** a set of parameters describing a certain type of
  projection between nodes or populations.
  The following types of connection models are handled so far:
    - _topological connections_: Connections to be created by
      `nest.topology.Connect()`. Topological connections are defined amongst
      populations.
  - Connection models are specified in the `network/connection_model` parameter
    subtree.


- We call __connection__ a specific projection between layers or populations.
  Each individual connection has a specific connection model, and optional
  parameters that can override the connection model's default.
  - The list of all individual connections is specified in the `network/topology` parameter
    subtree.

#### Simulation

- We call **network** a full NEST network, consisting of layers, connections,
  recorders, etc. It is represented by the `Network()` object, which contains all the objects representing the different aspects of a network.

- We call **session** a period of simulation of a network with specific
  inputs and parameters. Represented by the `Session()` object.
  - One can change the state of the network before each session by:
    - Modifying the state of all (or part of) the units within a population.
    - Modifying the state of all (or part of) the population-to-population
      connections (synapses).
  - One can change the state of the input layers of the network to mimic the
    presentation of a certain stimulus during that session. This can be done
    by specifying an **input** for that session. An input stimulus is an array
    from which the instantaneous poisson rate of each of the input layer's
    units is derived.
    - If the input has no time dimension, the poisson rate is constant for
      each unit.
    - If the input has a time dimension, the poisson rate of each unit varies
      with time accordingly.

- We call **simulation** a full experiment. Represented by a `Simulation()`
  object (which contains a `Network` object and a list of `Session` objects).

## Overview of a full simulation

A full NETS simulation consists of the following steps:

1. **Initialize kernel**:
    1. Set NEST kernel parameters
    2. Set random seeds for python's `random` module and NEST's random generator.
1. **Initialize network**:
    1. Initialize the `Network` object representing a NEST network: derive the dependent parameters from the independent parameters given in the parameter files.
    2. Create the network in NEST.
2. **Save the simulation's metadata**
    1. Create and/or clear the output directory
    1. Save parameters
    2. Save git hash
    3. Save network metadata (gid/location mappings, ...) (TODO)
3. **Run each session** in turn. For each session:
  1. Initialize the session:
    1. (Possibly) reset the network
    2. (Possibly) Change some of the network's parameters:
      1. Change neuron parameters for certain units of certain populations
      1. Change synapse parameters for certain synapse models.
    3. If there are any InputLayer in the network:
      1. Load the stimulus
      2. Set the input spike times or input rates to emulate the forecoming
         session stimulus.
  2. Call `nest.Simulate()`.
4. **Do some post-processing saves**
  1. Call `sim.save_data`:
    1. Save the session's times and stimuli
    2. Call `Network.save_data`:
        - Save the population rasters if the data was not cleared
        - Save the final state of synapses
  2. Save and plot other
    - Plot some connections
    - Dump some connections
    - ... any post-processing you like :)

## Parameters

#### `Params` parameter trees

Simulation parameter trees are specified as tree-like `Params` objects. The
hierarchical tree structure provides a concise means of specifying parameters
for nested 'scopes', which inherit parameters from ancestor scopes. The leaves
of the tree specify models, etc. that are instantiated in the simulation. 

##### Example parameter tree

Below is an example of a YAML file with a tree-like structure that can be loaded
and represented by the `Params` class:

```yaml
network:                # name of node (non-leaf)
  synapse_models:       # name of node (non-leaf)
    params:             # params common to all nodes below 'synapse_models'
      nest_model: static_synapse
    cortical:           # name of node (non-leaf)
      params:           # parameters common to all leaves below 'cortical' node
        target_neuron: bmt_neuron  # NEST model name
      AMPA_syn:         # name of node (non-leaf)
        params:         # parameters common to all leaves below 'AMPA_syn' node
          receptor_type: AMPA
        AMPA_syn:       # name of leaf
        AMPA_syn_to_MC: # name of leaf
      NMDA_syn:         # name of leaf
        params:         # params of the leaf
          receptor_type: NMDA
simulation:             # name of leaf
  params:               # parameters of 'simulation' leaf.
    local_num_threads: 1
```

Each node can have a `params` key containing a dictionary of parameters for that
node. All the keys of a node that are not named 'params' are children nodes. A
leaf is a node that doesn't have children (meaning, its only key is `params` if
it has any). The parameters of a node will be passed along to the children (and
possibly overwritten by them) until we reach a leaf.

In the example above, the leaves of the 'network' subtree and their respective
parameters are as follows:

```yaml
AMPA_syn:
    nest_model: static_synapse
    target_neuron: bmt_neuron
    receptor_type: AMPA
AMPA_syn_to_MC:
    nest_model: static_synapse
    target_neuron: bmt_neuron
    receptor_type: AMPA
NMDA_syn:
    nest_model: static_synapse
    target_neuron: bmt_neuron
    receptor_type: NMDA
```

#### Merging trees

`Params` objects can be merged horizontally, meaning the parameters (contents of
the `params` key) of nodes with the same name and at the same position in the
tree are combined in a `ChainMap` _before_ leaves inherit the parameters of
their parent nodes.

During a `nets.run()` call, the following parameters are combined (from lowest
to highest precedence):
- `constants.DEFAULT_PARAMS_PATH`: Default path to a parameter file to load.
- `run(input_path)`: Path to a parameter file to load.
- `run(overrides=<params_object>): A parameter object given directly to `run()`.
- CLI input: Parameters in the file given by the `--input=<path>` command-line
  flag.


#### Main parameter file

nets.run takes as input a path to a yaml file containing the relative paths to
the different parameter files to be merged. The main parameter file will look for example as follows, if the parameters are split amongst subdirectories and files:

```yaml
# List is read from last to first. param files towards the beginning take
# precedence over those towards the end if there are duplicate parameters.
- ./network/layers.yml
- ./network/synapse_models.yml
- ./network/neuron_models.yml
- ./network/connections.yml
- ./network/connection_models.yml
- ./simulation/populations.yml
- ./simulation/recorders.yml
- ./simulation/sessions_df.yml
- ./simulation/kernel_params.yml
- ./simulation/simulation_params.yml
```

Nets merges the parameter trees from all these files horizontally to generate the __Full parameter tree__ that fully describes the whole simulation.



### Full parameter tree: description of parameters

The final parameter tree (obtained after merging the parameter trees from all the parameter files) must have the following
subtrees/leaves:

NB: Parameters for which no default value is given are _mandatory_ and should be
defined in the final parameter tree.

- `kernel` (leaf): Contains the kernel parameters.
  - The following parameters are recognized:
    - `local_num_threads` (int): Passed to `nest.SetKernelStatus()`. (default 1)
    - `resolution` (float): Passed to `nest.SetKernelStatus()` (default `1.0`)
    - `print_time` (bool): Passed to `nest.SetKernelStatus()` (default `False`)
    - `overwrite_files` (bool): Passed to `nest.SetKernelStatus()` (default
      `True`)
    - `nest_seed` (int): Used to set general and thread-specific seeds in NEST
      (default from `nets.constants`)
    - `python_seed` (int): Used to set seed in `random` and `NumPy.random`
      modules (default from `nets.constants`)
    - `extension_modules` (list of str): List of external modules to install.
      Each module is passed to `nest.Install()` (default `[]`)

- `simulation` (leaf): Contains the simulation parameters.
  - The following parameters are recognized:
    - `output_dir` (str or None): Path to the output directory. Can be
      overwritten by the 'output' CLI kwarg, by the `__init__.run()`
      'output_dir' kwarg or by the `Simulation.__init__()` 'output_dir' kwarg.
      (default from `nets.constants`)
    - `input_dir` (str or None): Path to the input. Can be overwritten by the
      'output' CLI kwarg, by the `__init__.run()` 'output_dir' kwarg or by the
      `Simulation.__init__()` 'output_dir' kwarg. Please see the 'Input'
      section for details on how the input is loaded for each session.
      (default from `nets.constants`)
    - `clear_output_dir` (bool): If true, the contents of the subdirectories of
      `output_dir` listed in the CLEAR_SUBDIRS constant (defined in `save.py`)
      are deleted during a `Simulation.save()` call before any saving of output. 
      (default `False`)
        - If set to `False`, it is possible that an output directory contains data
          from a previous simulation.
        - If you add subdirectories to the main output_dir don't forget to
          update the `CLEAR_SUBDIRS` variable accordingly.
    - `dump_connections` (bool): If true, the unit-to-unit synapses are dumped
      during a `__init__.run()` call. Modify the `dump_connection`
      connection_model parameter to dump only a subset of the connections.
      (default `False`)
    - `plot_connections` (bool): If true, population-to-population connections
      are plotted during a `__init__.run()` call. Modify the `plot_connection`
      connection_model parameters to plot only a subset of the connections.
      (default `False`)
    - `dump_connection_numbers` (bool): If true, the number of incoming
      connections by population for each connection type is dumped during a
      `__init__.run()` call. (default `False`)

- `sessions` (subtree): Contains the parameters for each session and the list of
  sessions to be run in order. The sessions are represented by the leaves of
  that subtree. The following parameters are recognized:
  - `order` (list of str): List of sessions to be run during simulation.
    (default `[]`)
  - `reset_network` (bool): Whether to call `nest.ResetNetwork()` before running
    the session (default `False`)
  - `session_input` (bool): Absolute or relative path to a NumPy array defining
    the stimulus for that session. Possibly ignored if the `input_path` session
    parameter points to a NumPy array. Please refer to the 'Inputs' paragraph for details on how this parameter is interpreted.
    (default `None`)
  - `time_per_frame` (float): Number of milliseconds during which each 'frame'
    of the input movie is shown to the network. (default `1.0`)
  - `simulation_time` (float): Duration of the session, in ms. The stimulus is
    generated by repeating each element of the session input array for
    `time_per_frame` milliseconds, and the array of repeated elements is
    trimmed or repeated such that the stimulus lasts `simulation_time`
    (mandatory).
    - Example: if the session input array has two frames, time_per_frame = 2 (ms) and simulation_time = 7 (ms), the effective input at each millisecond will be [frame 1, frame 1, frame 2, frame 2, frame 1, frame 1, frame 2]
  - `input_rate_scale_factor` (float or int): Scaling factor to obtain the
    Poisson firing rates in Hz from array values of the stimulus array. The
    corresponding firing rate in Hz for a given array value is equal to
    <value> * <session_input_rate_scale_factor> * <layer_input_scale_factor>.
    See `input_rate_scale_factor` in the layer parameters and the "Input"
    documentation paragraph. (default 1.0)
  - `unit_changes` (list): List of dictionaries describing the parameter changes
    to apply to a proportion of units of specific populations. (default `[]`).
    Each `unit_change` dictionary should be of the form::
        {
            'layers': <layer_name_list>,
            'layer_type': <layer_type>,
            'population': <pop_name>,
            'change_type': <change_type>,
            'proportion': <prop>,
            'filter': <filter>,
            'params': {<param_name>: <param_value>,
                       ...}
        }
    where:
    ``<layer_name_list>`` (default None) is the list of name of the
        considered layers. If not specified or empty, changes are
        applied to all the layers of type <layer_type>.
    ``<layer_type>`` (default None) is the name of the type of
        layers to which the changes are applied. Should be 'Layer'
        or 'InputLayer'. Used only if <layer_name> is None.
    ``<population_name>`` (default None) is the name of the
        considered population in each layer. If not specified,
        changes are applied to all the populations.
    ``<change_type>`` ('constant' or 'multiplicative'). If
        'multiplicative', the set value for each parameter is the
        product between the preexisting value and the given value.
        If 'constant', the given value is set without regard for the
        preexisting value. (default: 'constant')
    ``<prop>`` (default 1) is the proportion of units of the
        considered population on which the filter is applied. The
        changes are applied on the units that are randomly selected
        and passed the filter.
    ``filter`` (default {}) is a dictionary defining the filter
        applied onto the proportion of randomly selected units of
        the population. The filter defines an interval for any unit
        parameter. A unit is selected if all its parameters are
        within their respectively defined interval. The parameter
        changes are applied only on the selected units.
        The ``filter`` dictionary is of the form:
            {
                <unit_param_name_1>:
                    'min': <float_min>
                    'max': <float_max>
                <unit_param_name_2>:
                    ...
            }
        Where <float_min> and <float_max> define the (inclusive)
        min and max of the filtering interval for the considered
        parameter (default resp. -inf and +inf)
      ``'params'`` (default {}) is the dictionary of parameter changes
          applied to the selected units.
  - `synapse_changes` (list): List of dictionaries describing the parameter
    changes to apply to population-to-population connections within specific
    populations. (default `[]`). TODO: Document

- `network` (subtree): Should contain the following subtrees:
  - `layers` (subtree): Defines layer parameters. Each leaf is a layer. The
    following parameters are recognized for each layer-leaf:
    - `type` (str): `'InputLayer'`, `'Layer'` or `None` (default `'Layer'`).
    - `nrows`, `ncols` (int): Number of rows/columns for the layer (mandatory)
    - `edge_wrap` (bool): Whether the layer is wrapped in NEST. (default
      `False`)
    - `visSize` (bool): NEST physical 'extent' of each the layer's side. We only
      consider square layers for now (possible TODO) (mandatory)
    - `populations` (dict): Non-empty dictionary containing the neuron model
      (keys) and the number of units of that model at each grid location
      (values). (mandatory)
    - `area` (str): Area of the layer. (default `None`)
    - `input_rate_scale_factor` (float or int): Scaling factor to obtain the
      Poisson firing rates in Hz from array values of the stimulus array. The
      corresponding firing rate in Hz for a given array value is equal to
      <value> * <session_input_rate_scale_factor> * <layer_input_scale_factor>.
      See `input_rate_scale_factor` in the session parameters and the "Input"
      documentation paragraph. (mandatory for `InputLayer` layers)
  - `neuron_models` (subtree): Defines neuron models. Each leaf is a neuron
    model. All parameters of a leaf are passed to ``nest.CopyModel()`` __except
    the following__:
    - `nest_model` (str): Base NEST model for the neuron model. (mandatory)
  - `synapse_models` (subtree): Defines synapse models. Each leaf is a synapse
    model. All parameters of a leaf are passed to ``nest.CopyModel()`` without
    change __except the following__:
    - `nest_model` (str): Base NEST model for the synapse model. (mandatory)
    - `receptor_type` (str): Name of the receptor_type. Used to derive the
      port-in of the connection. If specified, a `target_neuron` parameter is
      expected. (default `None`)
    - `target_neuron` (str): Name of the target neuron. Used to derive the
      port-in of the connection. Ignored if `receptor_type` is unspecified.
      (default `None`)
  - `connection_models` (subtree): Defines the connection models. Each leaf is a
    connection model. All parameters are interpreted as "NEST parameters" and
    passed to NEST __except the following parameters__:
    - `type` (str). Type of the connection. Only 'topological' connections are
      recognized (default `'topological'`).
    - `dump_connection` (bool): Whether connections of that connection model
      should be dumped during a `Simulation.dump_connections()` call. (default
      `False`)
    - `plot_connection` (bool): Whether connections of that connection model
      should be plotted during a `Simulation.plot_connections()` call. (default
      `True`)
    - All other parameters (`kernel`, `mask`, `weights`...) define the
      topological connections and will be passed to
      nest.topology.ConnectLayers() without modification.
  - `topology` (leaf): Should contain a `connections` parameter consisting in a
    list of dictionary describing population(/layer)-to-population(/layer)
    connections. Each item specifies a list of source layers and target layers.
    Actual connections are created for all the source_layer x target_layer
    combinations.
    Each item should be a dictionary with the following fields:
    - `source_layers` (list[str]): List of source layers. Individual connections
      are created for all source_layer x target_layer combinations,. (mandatory)
    - `source_population` (str or None): source pop for each of the source_layer x
      target_layer combination. If not specified, each connection originates
      from all the populations in the source layer.
      (default `None`)
    - `source_layers` (list[str]): List of target layers. Individual connections
      are created for all source_layer x target_layer combinations,. (mandatory)
    - `target_population` (str or None): target pop for each of the source_layer x
      target_layer combination. If not specified, each connection targets
      all the populations in the target layer.
      (default `None`)
    - `connection_model` (str): Name of the connection model. (mandatory)
    - `nest_params` (dict or None): NEST parameters for the specific connections
      represented by the item. Takes precedence over the connection model's
      NEST parameters
    - `params` (dict or None): Non-NEST parameters for the specific connections
      represented by the item. Takes precedence over the connection model's
      non-NEST parameters
  - `recorders` (subtree): Defines recorder models. Each leaf is a recorder
    model. All parameters of a leaf are passed to ``nest.CopyModel()`` or
    ``nest.SetDefaults()``except the following:
    - `nest_model` (str): Base NEST model for the recorder model.
  - `populations` (subtree): Defines the populations (= models within a specific
    layer). Used to define which population is recorded by what type of
    recorder. Leaves are individual populations. The name of populations-leaves
    should be the corresponding neuron model. The following parameters are
    expected or recognized:
    - `layer` (str): Name of the layer for that population. (mandatory)
    - `recorders` (dict): Dictionary of which the keys are the models of
      recorders that will be created and connected to that population, and the
      values are dictionary (possibly empty) containing overriding parameters
      for the corresponding recorder.
    - **NB**: Make sure that no population is missing compared to the `'layer'`
      parameters. (TODO: Add check)


### Inputs to the network

The input 'shown' to the network (that is, used to set the firing rates of the
InputLayer during a forecoming session) is a NumPy array of dimensions
compatible with the network's `InputLayer`'s dimension. The NumPy array should
be 3D with the following dimensions: `(frames, rows, columns)`
- _frames_ : If the stimulators are spike_generators, successive 'frames' are
    shown to the InputLayer, each for a certain duration. If the stimulators
    are poisson_generators, only the first frame is shown to the network.
- _rows_ : should be greater than the number of rows of the `InputLayer`.
- _columns_ : should be greater than the number of columns of the `InputLayer`

##### From input arrays to firing rates

If InputLayer stimulators are poisson generators, the rates are set according
proportionally to the first 'frame' of the input array. The other frames are
ignored.
If InputLayer stimulators are spike generators, the instantaneous rates at each
time are set proportionally to the frame corresponding to that time. Each frame
is 'shown' to the layer for a certain duration (`time_per_frame` parameter).
In both cases __the scaling factor from np-array values to firing rates is equal to__:

<session_input_rate_scale_factor> * <layer_input_rate_scale_factor>, where <session_input_rate_scale_factor> and <layer_input_rate_scale_factor> are the `input_rate_scale_factor` parameters from the session and layer parameters.

#### Loading of the input arrays

The input stimulus "shown" to the network is loaded using the following steps
during the initialization of each session:

1. (option 1): If the `input_path` simulation parameter is an absolute path
    pointing towards a NumPy array, this array will be loaded for all sessions.
    Otherwise...
2. (option 2): If the `session_input` session parameter for the considered
    session is an absolute path pointing towards a NumPy array, it will be
    loaded for that session. Otherwise...
3. (option 3): If the `input_path` simulation parameter is a path pointing
    towards a directory, and the `session_input` session parameter can be
    interpreted as a relative path, from the `input_path` directory, pointing
    towards a NumPy array, it will be loaded for the session.

In summary, if an absolute path to an array is specified in the command line
`input` optional argument, it will be used for all sessions. Otherwise, a good
approach is to specify the absolute path to an 'input directory' in the
nets.constants, leave the `'input_path'` simulation parameter otherwise
unspecified and specify the relative path (from the `'input directory'`) to each
session's input in the session parameters.

### Duration of a session

The simulation time is equal to the 'simulation_time' session
parameter.
The stimulus is
    generated by repeating each element of the session input array for
    `time_per_frame` milliseconds, and the array of repeated elements is
    trimmed or repeated such that the stimulus lasts `simulation_time`
    (mandatory).
    - Example: if the session input array has two frames, time_per_frame = 2 (ms) and simulation_time = 7 (ms), the effective input at each millisecond will be [frame 1, frame 1, frame 2, frame 2, frame 1, frame 1, frame 2]


## Run the simulation

### To run **from Python** (_e.g._ in a Jupyter notebook):

#### Using the ``Simulation`` object to run the simulation step by step:

```python
import nets

# The simulation parameter file to use.
params_path = 'params/default.yml'

# User defined list of tree-like overrides
# First override is ignored (just an example)
override1 = {}
# Second override sets the number of NEST threads
override2 = {'kernel': {'params': {'local_num_thread': 10}}}
# Leftmost is applied last and precedes.
overrides = [override2, override1]

# Load the parameters and apply the overrides
params = nets.load_params(params_path, *overrides)

# You can also override the input and output settings in the parameter file by
# passing them as kwargs when you create the simulation:
input_dir = None  # Path to an input directory. NOTE: This can also be a path
                  # to a saved NumPy array.
output_dir = None # Path to output directory.

# Create the simulation.
sim = nets.Simulation(params, input_path=input_path, output_dir=output_dir)

# Run the simulation.
sim.run()

# Save the results, dump the connections, etc.
sim.save()
sim.dump()
```

#### Using the ``nets.run()`` function to run the full simulation at once:

```python
import nets

# The simulation parameter file to use.
params_path = 'params/default.yml'

# User defined list of tree-like overrides
# Leftmost is applied last and precedes.
overrides = []

# The __init__.run() function knows which steps to run or not by reading the
# 'simulation' parameters.
nets.run(params_path, *overrides, input_dir=None, output_dir=None)
```

**NB**:
- `DEFAULT_PARAMS_PATH` from `nets.constants` parameters are applied when
  loading the parameters during a `__init__.run()` call.
- Control which stimulation steps are run during a `__init__.run()` call by
  setting the simulation parameters.

### To run directly **from the command line**:

A command line package call is equivalent to a `__init__.run()` call. Notes in
the above paragraph apply.

To run a simulation from the command line:
```bash
python -m nets <param_file.yml> [-i <input>] [-o <output>]
```

## Outputs of the simulation

All the simulations outputs are saved in subdirectories of the directory
specified in the `output_dir` simulation parameter.

The `nets.save` module contains utility functions to load the simulation outputs.

The main output consists in the raw data recorded by each recorder (spike detector, multimeter or weight_recorder), as saved by NEST, and the recorder metadata which contains all the necessary information to postprocess the raw data. Both are saved by NETS in the 'data' subdirectory of the output directory. As a reminder, each recorder is paired to a single population, and the recorder metadata contains information about that population. 

Below is an exemple of how to use the `nets.save` utility functions to load the recorded activity as pandas arrays after the simulation has ended:


```python
import nets.save
from pathlib import Path

OUTPUT_DIR = Path('./output')
data_path = OUTPUT_DIR/'data'

## We can load the start and end time for each sessions:
session_times = nets.save.load_session_times(OUTPUT_DIR) # {<session>: range(<session_start>, <session_end>)}


## Get the names of all the recorders
recorder_type = 'multimeter'
# recorder_type = 'spike_detector'
# recorder_type = 'weight_recorder'
all_recorder_metadata_filenames = nets.save.load_metadata_filenames(OUTPUT_DIR, recorder_type)

## To load the data for a recorder, all we need is the path to its metadata file.
metadata_filename = all_recorder_metadata_filenames[0]
# All the information about this recorder and the population it's connected to are contained in its metadata file
mm_metadata = nets.save.load_yaml(data_path/metadata_filename)
print(mm_metadata.keys())
# load the data as panda dataframe
mm_df = nets.save.load(data_path/metadata_filename)


# We can also use some more advanced options for loading:
mm_df = nets.save.load(
    data_path/metadata_filename,
    assign_locations=True, # add x,y,z columns
    usecols=['gid', 'time', 'V_m'], # Ignore some columns
    filter_ratio={
       'gid': 0.5,
       'time': 0.5,
    }, # Load data for half of all unique GIDs and timestamps
    filter_type={
       'gid': 'random',
       'time': 'even',
    }, # GIDs are sampled randomly and timestamps evenly
)
```
