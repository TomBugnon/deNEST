# NETS: Network Simulations (in NEST)


## Network

##### Definitions

- We call __model__ a NEST model (whether of a neuron or synapse) with a specific
    name and set of parameters. All models are copied from their respective
    "NEST model" using `nest.Copy()` with their specific parameters. NEST models
    can be either imported from external modules or built-in.
    - models should have a different name from NEST models.
    - multiple models can have different names but the same NEST model and
        parameters.


- We call __layer__ a NEST topological layer. A layer is akin to a grid of a
    certain dimension (nrows x ncols) and is composed of a given number of
    nodes at each grid _location_. Nodes at each location can be from different
    models.
    - Please use 2D, regular layers only
    - Layers can be of the type `InputLayer` when composed of generators. Input
        layers are automatically duplicated to contain parrot neurons, so that
        recording of generators' activity is possible (via parrot neurons).
        Additionally, the state of generators can be changed before each session
        to reflect the forecoming input stimulus (see 'session' paragraph)


- We call __population__ a set of nodes of the same model within a specific
    layer. In practice, populations are (layer, neuron_model) tuples.
    - Connections are defined between layers or populations (but not at a finer
        grain)
    - Each recorder created in NEST corresponds to a specific population,
    so it is possible to decide to record (or not) each population
    independently, and to record different variables for different populations.


- We call __connection_model__ a set of parameters describing a certain type of
    projection between node populations. The following types of
    connection_models are handled so far:
    - _topological connections_: Connections to be created by
        nest.topology.Connect()
    - _connections from file_: Connections between individual units loaded from
        a dump from a previous simulation. Arbitrary units to units connections
        are handled with this type of connections.
    - _rescaled (topological) connections_: "Rescaled" topological connections
        from a dump from a previous simulation. This allows one to change the
        spatial profile of topological connections between two networks while
        keeping the same number of outgoing connections per node.


- We call __connection__ a specific projection between layers or populations.
    Each individual connection has a specific connection model.

## Simulation

##### Definitions

- We call __network__ a full NEST network, consisting of layers, connections,
    etc. Represented by the `Network()` object.


- We call __session__ a period of simulation of a network with specific
    background conditions/inputs. Represented by the `Session()` object.
    - One can change the state of the network before each session by:
        - Modifying the state of all (or part of) the units within a population.
        - Modifying the state of all (or part of) the population-to-population
            connections (synapses).
    - One can change the state of the input layers of the network to mimic the
        presentation of a certain stimulus during that session. This can be done
        by specifying an __input__ for that session. An input stimulus is an
        array from which the instantaneous poisson rate of each of the input
        layer's units is derived.
        - If the input has no time dimension, the poisson rate is constant for
            each unit.
        - If the input has a time dimension, the poisson rate of each unit
            varies with time accordingly.


- We call __simulation__ a full experiment. Represented by a `Simulation()`
    object (which contains a `Network` object and a list of `Session` objects).

##### Overview of a full simulation

1. __Initialize kernel__:
    1. Set NEST kernel
    2. Set random seeds
1. __Initialize network__:
    1. Initialize the `Network` object representing a NEST network: derive
        the dependent parameters from the independent parameters given in
        parameter.
    2. Create the network in NEST
3. __Run each session__ in turn. For each session:
    1. Initialize the session:
        1. Load the stimulus
        2. (Possibly) reset the network
        3. Change the network's dynamic variables
        4. Set the input spike times or input rates to emulate the forecoming
            session stimulus.
    2. Call nest.Simulate()
4. __Save the simulation__:
    1. Create or clear the output directory
    2. Save and do some stuff.
        - Format and save the recorders' data
        - Save some raster plots
        - Plot some connections
        - Dump some connections
        - ... any post-processing you like :)

## Parameters

##### `Params` parameter trees

We define parameters as ``Params`` tree-like objects. The tree is a concise way
of writing default parameters for different 'scopes'.The full simulation is
defined by a merged ``Params`` tree structure. The merged parameters originate
from either direct `overrides` (tree-like) or files containing paths to
tree-like yaml files.

##### Example parameter tree

Below is an example of a yaml file with a tree-like structure recognized by the
`Params` class:

```yaml
network: # name of node (non-leaf)
  synapse_models: # name of node (non-leaf)
    params: # params common to all leaves below 'synapse_models' node
      nest_model: static_synapse
    cortical: # name of node(non-leaf)
      params: # parameters common to all leaves below 'cortical' node
        target_neuron: bmt_neuron
        weight: 1.0
      AMPA_syn: # name of node (non-leaf)
        params: # parameters common to all leaves below 'AMPA_syn' node
          receptor_type: AMPA
        AMPA_syn: # name of leaf
        AMPA_syn_to_MC: # name of leaf
            params: # params of the leaf
                weight: 2.0
      NMDA_syn: # name of leaf
        params: # params of the leaf
          receptor_type: NMDA
simulation: # name of leaf
  params: # parameters of 'simulation' leaf.
    local_num_threads: 1
```

Each node can have a `params` key containing a dictionary of parameters for that
node. All the keys of a node that are not named 'params' are children nodes.
A leaf is a node that doesn't have children (meaning, its only key is `params`
if it has any). The parameters of a node will be passed along to the children
(and possibly overwritten by them) until we reach a leaf.

In the example above, the leaves of the 'network' subtree and their respective
parameters are as following:
```yaml
AMPA_syn:
    nest_model: static_synapse
    target_neuron: bmt_neuron
    receptor_type: AMPA
    weight: 1.0
AMPA_syn_to_MC:
    nest_model: static_synapse
    target_neuron: bmt_neuron
    receptor_type: AMPA
    weight: 2.0
NMDA_syn:
    nest_model: static_synapse
    target_neuron: bmt_neuron
    receptor_type: NMDA
    weight: 1.0
```

##### Merging trees

``Params`` object are merged horizontally meaning the parameters (contents of
    the `params` key) of nodes with the same name and at the same position in
    the tree are combined in a ChainMap _before_ leaves inherit the parameters
    of their parent nodes.

During a `nets.run()` call, the following parameters are combined
(from lowest to highest precedence):
- ``DEFAULT_PARAMS_PATH``: Path to USER defined default parameters, possibly
    specified in `user_config.py` (lowest precedence)
- ``"path" parameters``: Parameters from the main simulation parameter file,
    passed to the main simulation parameters passed during a direct
    `run()` call or a CLI package call.
- ``USER_OVERRIDES``: Tree-like overrides defined in `user_config.py`.
- ``overrides``: Overrides passed by USER during a direct `run()` call. During a
    CLI package call, these overrides possibly contain the `input` and `output`
    CLI optional arguments. (highest precedence)

##### Full parameter tree

The final parameter tree (obtained after merging) is expected to have the
following subtrees/leaves:

NB: Parameters for which no default value is given are _mandatory_ and should
be defined in the final parameter tree.

- `kernel` (leaf): Contains the kernel parameters.
    - The following parameters are recognized:
        - `local_num_threads` (int): Passed to `nest.SetKernelStatus()`.
            (default 1)
        - `resolution` (float): Passed to `nest.SetKernelStatus()` (default 1.)
        - `print_time` (bool): Passed to `nest.SetKernelStatus()` (default False)
        - `overwrite_files` (bool): Passed to `nest.SetKernelStatus()` (default
            True)
        - `nest_seed` (int): Used to set general and thread-specific seeds in
            NEST (default from `user_config.py`)
        - `python_seed` (int): Used to set seed in `random` and `numpy.random`
            modules (default from `user_config.py`)
        - `extension_modules` (list of str): List of external modules to
            install. Each module is passed to `nest.Install()` (default [])


- `simulation` (leaf): Contains the simulation parameters.
    - The following parameters are recognized:
        - `output_dir` (str or None): Path to the output directory. Can be
            overwritten by the 'output' CLI kwarg, by the `__init__.run()`
            'output_dir' kwarg or by the `Simulation.__init__()` 'output_dir'
            kwarg. (default from `user_config.py`)
        - `input_dir` (str or None): Path to the input. Can be overwritten by
            the 'output' CLI kwarg, by the `__init__.run()` 'output_dir' kwarg or
            by the `Simulation.__init__()` 'output_dir' kwarg. Please see the
            'Input' paragraph for details on how the input is loaded for each
            session. (default from `user_config.py`)
        - `dry_run` (bool): If true, `Simulation.run()` is not called in
            `__init__.run()`. This means that the simulation is initialized and
            saved as usual, but no nest.Simulate() call is performed and the
            sessions are not initialized. (default False)
        - `clear_output_dir` (bool): If true, the contents of the subdirectories
            of `output_dir` listed in the CLEAR_SUBDIRS constant (defined in
            `save.py`) are deleted during a `Simulation.save()` call before any
            saving of output. (default false)
            - If set to False, it is possible that an output directory contains
            data from a previous simulation.
            - If you add subdirectories to the main output_dir don't forget to
            update the CLEAR_SUBDIRS variable accordingly.
        - `save_simulation` (bool): If true, `Simulation.save()` is called in
            `__init__.run()` (default true)
        - `delete_raw_dir` (bool): If true, the raw data output by NEST is
            deleted during a `Simulation.save()` call. (default false)
        - `save_nest_raster` (bool): If true, NEST raster plots are generated
            during a `Simulation.save()` (default true)
        - `dump_connections` (bool): If true, the unit-to-unit synapses are
            dumped during a `__init__.run()` call. Modify the `plot_connection`
            connection_model parameters to dump only a subset of the
            connections. (default false)
        - `plot_connections` (bool): If true, population-to-population
            connections are plotted during a `__init__.run()` call. Modify the
            `plot_connection` connection_model parameters to plot only a subset
            of the connections. (default false)
        - `dump_connection_numbers` (bool): If true, the number of incoming
            connections by population for each connection type is dumped during
            a `__init__.run()` call.
            (default false)


- `sessions` (subtree): Contains the parameters for each session and the list
    of sessions to be run in order. The sessions are represented by the leaves
    of that subtree. The following parameters are recognized:
    - `order` (list of str): List of sessions to be run during simulation.
        (default [])
    - `reset_network` (bool): Whether to call `nest.ResetNetwork()` before
        running the session (default false)
    - `session_input` (bool): Absolute or relative path to a numpy array
        defining the stimulus for that session. Possibly ignored if the
        `input_path` session parameter points to a numpy array. Please refer to
        the 'Inputs' paragraph. (default None)
    - `time_per_frame` (float): Number of milliseconds during which each
        'frame' of the input movie is shown to the network. (default 1.)
    - `max_session_sim_time` (float): Maximum duration of the session, in ms.
        Actual simulation time of the session is the minimum between this
        parameter and the total duration of the input movie (nframes x
        time_per_frame). (default float('inf'))
    - `unit_changes` (list): List of dictionaries describing the parameter
        changes to apply to a proportion of units of specific populations.
        (default [])
    - `synapse_changes` (list): List of dictionaries describing the parameter
        changes to apply to population-to-population connections within specific
        populations. (default [])


- `network` (subtree): Should contain the following subtrees:
    - `layers` (subtree): Defines layer parameters. Each leaf is a layer. The
        following parameters are recognized for each layer-leaf:
        - `type` (str): 'InputLayer', 'Layer' or None (default 'Layer').
        - `nrows`, `ncols` (int): Number of rows/columns for the layer
            (mandatory)
        - `edge_wrap` (bool): Whether the layer is wrapped in NEST. (default
            false)
        - `visSize` (bool): NEST physical 'extent' of each the layer's side.
            We only consider square layers for now (possible TODO) (mandatory)
        - `populations` (dict): Non-empty dictionary containing the neuron model
            (keys) and the number of units of that model at each grid location
            (values). (mandatory)
        - `area` (str): Area of the layer. (default None)
        - `input_rate_scale_factor` (float or int): Scaling factor for the poisson rate
            applied to an InputLayer to reflect a stimulus. (mandatory for
            InputLayers)
        - `weight_gain`: Scaling factor for the weight of connections of which
            the considered layer is the _source_. (default 1.)
        - `scale_kernels_masks_to_extent`: Whether the kernels and masks of
            connections of which the considered layer is the pooling layer are
            considered to be expressed in 'number of units', rather than
            physical extent. (default True).
            - __Important__: Please note that by default the kernel and masks
            are interpreted as being expressed in 'number of units' rather than
            physical extent and are accordingly scaled before being passed to
            NEST.
        - `scale_input_weight`: Whether the weight of connections originating
            from an InputLayer is scaled by number of layers within an
            InputLayer. (default false)
    - `neuron_models` (subtree): Defines neuron models. Each leaf is a neuron
        model. All parameters of a leaf are passed to ``nest.CopyModel()``
        except the following:
        - `nest_model` (str): Base NEST model for the neuron model. (mandatory)
    - `synapse_models` (subtree): Defines synapse models. Each leaf is a synapse
        model. All parameters of a leaf are passed to ``nest.CopyModel()``
        without change except the following:
        - `nest_model` (str): Base NEST model for the synapse model. (mandatory)
        - `receptor_type` (str): Name of the receptor_type. Used to derive the
            port-in of the connection. If specified, a `target_neuron` parameter
            is expected. (default None)
        - `target_neuron` (str): Name of the target neuron. Used to derive the
            port-in of the connection. Ignored if `receptor_type` is
            unspecified. (default None)
    - `connection_models` (subtree): Defines the connection models. Each leaf
        is a connection model. The following parameters are recognized:
        - `type` (str). Type of the connection. Recognized types are
            'topological', 'from_file', 'rescaled'. (default 'topological').
        - `source_dir` (str). Path to the directory containing the previously
            dumped connections. Used (and mandatory) only if the connection is
            of type 'from_file' or 'rescaled'. (default None)
        - `scale_factor` (float). Scaling factor for the kernel and masks.
            (default 1.)
        - `dump_connection` (bool): Whether connections of that connection model
            should be dumped during a `Simulation.dump_connections()` call.
            (default false)
        - `plot_connection` (bool): Whether connections of that connection model
            should be plotted during a `Simulation.plot_connections()` call.
            (default true)
        - The other parameters define the topological or rescaled (topological)
            connections and will be passed to nest.topology.ConnectLayers() (for
            topological) connections, possibly after weight, kernel and mask
            scaling. These parameters are ignored in the case of connections
            'from_file', and interpreted similarly in 'rescaled connections' as
            in topological connections.
        - __Important__:
            - For each connection model, if the  `scale_kernels_masks_to_extent`
            parameter of the pooling layer is true, the masks and kernel sizes
            specified in the parameters (and scaled by the `scale_factor`) are
            interpreted as "number_of_units" rather than physical extent.
    - `topology` (leaf): Should contain a `connection` parameter consisting in
        a list of individual population(/layer)-to-population(/layer)
        connections. Each individual connection should be a dictionary with the
        following fields:
        - source_layer (str): source layer. (mandatory)
        - source_population (str or None): source pop. If not specified, the
            connection originates from all the populations in the source layer.
            (default None)
        - target_layer (str): target layer. (mandatory)
        - target_population (str or None): Target pop. If not specified, the
            connection targets all the populations in the target layer. (default
            None)
        - connection (str): Name of the connection model. (mandatory)
    - `recorders` (subtree): Defines recorder models. Each leaf is a recorder
        model. All parameters of a leaf are passed to ``nest.CopyModel()``
        except the following:
        - `nest_model` (str): Base NEST model for the recorder model.
    - `populations` (subtree): Defines the populations (= models within a
        specific layer). Used to define which population is recorded by what
        type of recorder. Leaves are individual populations. The name of
        populations-leaves should be the corresponding neuron model. The
        following parameters are expected or recognized:
        - `layer` (str): Name of the layer for that population. (mandatory)
        - `recorders` (dict): Dictionary of which the keys are the models of
            recorders that will be created and connected to that population,
            and the values are dictionary (possibly empty) containing
            overriding parameters for the corresponding recorder.
        - __NB__: Make sure that no population is missing compared to the
            'layer' parameters.



## Inputs to the network

##### Input array

The input 'shown' to the network (that is, used to set the firing rates during
    a forecoming session) is a numpy array of dimensions compatible with the
    network's InputLayer's dimension. The numpy array should be 4D with the
    following dimensions: `(rows, columns, filters, frames)`
- _rows_ : should be greater than the number of rows of the InputLayer
- _columns_ : should be greater than the number of columns of the InputLayer
- _filter_ : should be greater than the number of Layers in the InputLayer.
    Please ignore this dimension in general cases but make sure your array is
    4D. This dimension would correspond to different filters to be applied to
    an image to mimic different 'filterings' by the retina, LGN, V1 etc in an
    object recognition network.
- _frames_ : Successive 'frames' are shown to the InputLayer, each for a certain
    duration.

#### Loading of the input arrays

The input stimulus "shown" to the network is loaded using the following steps
during the initialization of each session:

1. If the `input_path` simulation parameter is an absolute path pointing towards
    a numpy array, this array will be loaded for all sessions. Otherwise...
2. If the `session_input` session parameter for the considered session is an
absolute path pointing towards a numpy array, it will be loaded for that
session. Otherwise...
3. If the `input_path` simulation parameter is a path pointing towards a
directory, and the `session_input` session parameter can be interpreted as a
relative path, from the `input_path` directory, pointing towards a numpy array,
it will be loaded for the session.
4. Just for reference, there is another type of loading that is specific to our
usage and doesn't need to be developed upon right now. We will probably migrate
the corresponding code outside of the package in the future.

In summary, if an absolute path to an array is specified in the command line
`input` optional argument, it will be used for all sessions. Otherwise, a good
approach is to specify the absolute path to an 'input directory' in the
user_config.py, leave the 'input_path' simulation parameter otherwise
unspecified and specify the relative path (from the 'input directory') to each
session's input in the session parameters.

## Run the simulation

#### To run **from Python** (_e.g._ in a Jupyter notebook):

**Using the ``Simulation`` object to run the simulation step by step:**

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

**Using the ``nets.run()`` function to run the full simulation at
once:**

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

NB:
- `DEFAULT_PARAMS_PATH` and `USER_OVERRIDES` from `user_config.py` parameters
    are applied when loading the parameters during a `__init__.run()` call.
- Control which stimulation steps are run during a `__init__.run()` call by
    setting the simulation parameters.


#### To run directly **from the command line**:

A command line package call is equivalent to a `__init__.run()` call. Notes in
the above paragraph apply.

To run a simulation from the command line:
```bash
python -m nets <param_file.yml> [-i <input>] [-o <output>]
```

## Outputs of the simulation

All the simulations outputs are saved in subdirectories of the directory
specified in the `output_dir` simulation parameter.
