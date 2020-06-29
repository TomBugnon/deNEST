Overview
========

Definitions
~~~~~~~~~~~

Network
-------

  * We call **network** a full network in NEST, consisting of layers of units with
    specific models, projections of specific types with specific synapse models
    amongst these layers, population recorders (multimeters, spike detectors) and
    projection recorders (weight recorder).

    * The full network is represented in deNEST by the `Network()` class

  * New NEST **models** (**neuron and generator model**, **synapse model** or
    **recorder model**) can be specified with arbitrary parameters. During network
    creation, models with specific parameters are created in NEST using a
    ``nest.CopyModel()`` or a ``nest.SetDefaults`` call.

    * Synapse models are represented by the ``SynapseModel`` class in deNEST. All other models are represented by the ``Model`` class. * Neuron and generator models are specified as leaves of the `network/neuron_models` parameter subtree (see section below)
    * Synapse models are specified as leaves of the `network/synapse_models` parameter subtree (see "Network parameters" section below)
    * Recorder models are specified as leaves of the `network/recorder_models` parameter subtree (see "Network parameters" section below)

  * A **layer** is a NEST topological layer, created with a ``tp.CreateLayer``
    call in NEST. We call **population** all the nodes of the same model
    within a layer.

    * Layers can be of the type ``InputLayer`` when they are composed of
      generators. An extra population of parrot neurons can be automatically
      created and connected one-to-one to the generators, such that recording of
      generators' activity is possible. Additionally, ``InputLayer`` support
      shifting the ``origin`` flag of stimulators at the start of a Session.
    * Layers are represented by the ``Layer`` or ``InputLayer`` class in deNEST
    * Layers are specified as leaves of the `network/layers` parameter subtree (see "Network parameters" section below)

  * A **projection model** is a template specifying parameters passed to
    ``tp.ConnectLayers``, and that individual projections amongst populations can
    inherit from.

    * Projection models are represented by the ``ProjectionModel`` class in deNEST.
    * Projection models are specified as leaves of the `network/projection_models` parameter subtree (see "Network parameters" section below)

  * A **projection** is an individual projection between layers or populations,
    created with a ``tp.ConnectLayers`` call. The parameters passed to
    ``tp.ConnectLayers`` are those of the "projection model" of the specific
    projection.

    * The list of all individual projections within the network is specified in the ``'projections'`` parameter of the `network/topology` parameter subtree (see "Network parameters" section below).

  * A **population recorder** is a recorder connected to all the nodes of a given
    population. A **projection recorder** is a recorder connected to all the
    synapses of a given projection. Recorders with arbitrary parameters can be
    defined by creating "recorder models". However, currently only recorders based
    on the 'multimeter', 'spike_detector' and 'weight_recorder' NEST models are
    supported.

    * population and projection recorders are represented by the
      ``PopulationRecorder``  and ``ProjectionRecorder`` classes in
      deNEST.

    * The list of all population recorders and projection recorders are specified
      in the ``'population_recorders'`` and ``'projection_recorders'`` parameters
      of the `network/recorders` parameter subtree (See "Network parameters"
      section below)

Simulation
----------

* A **session model** is a template specifying parameters inherited by
  individual sessions.

  * session models are specified as leaves of the ``session_models`` parameter
    subtree (see "Simulation parameters" section below)

* A **session** is a period of simulation of a network with specific inputs
  and parameters, and corresponds to a single ``nest.Simulate()`` call. The
  parameters used by a given session are inherited from its session model.

  * A session's parameters define the operations that may be performed before
    running it:

    1. Modifying the state of some units (using the ``Network.set_state`` method)
    2. (Possibly) shift the ``origin`` flag for the ``InputLayer`` stimulators
    3. (Possibly) deactivate the recorders for that session by setting their
       ``start`` flag to the end of the session

  * Individual sessions are represented by the ``Session`` object in deNEST.
    (see "Simulation parameters" section below)

* A **simulation** is a full experiment. It is represented by the `Simulation()`
  object in deNEST, which contains a `Network` object and a list of `Session`
  objects.

  * The list of sessions run during a simulation is specified by the
    ``sessions`` parameter of the ``simulation`` parameter subtree (eg:
    ``sessions: ['warmup', 'noise', 'grating', 'noise', 'grating']``) (see
    "Simulation parameters" section below)


Overview of a full simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A full deNEST simulation consists of the following steps:

1. **Initialize simulation** (``Simulation.__init__``)

   1. **Initialize kernel**: (``Simulation.init_kernel``)
      1. Set NEST kernel parameters
      2. Set seed for NEST's random generator.
   2. **Create network**:
      1. Initialize the network objects (``Network.__init__``)
      2. Create the objects in NEST (``Network.__create__``)
   3. **Initialize the sessions** (``Session.__init__``)
   4. **Save the simulation's metadata**
      * Create the output directory
      * Save the full simulation parameter tree
      * Save git hash
      * Save session times
      * Save network metadata (TODO)
      * Save session metadata (TODO)

2. **Run the simulation** (``Simulation.__run__``). This runs each session in
   turn (``Session.__run__``) :

   1. Initialize session (``Session.initialize``)

      - (Possibly) reset the network
      - (Possibly) inactivate recorders for the duration of the session
      - (Possibly) shift the `origin` of stimulator devices to the start of the session
      - (Possibly) Change some of the network's parameters using the ``Network.set_state`` method

        1. Change neuron parameters
        2. Change synapse parameters

   2. Call `nest.Simulate()`.


Specifying the simulation parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All parameters used by deNEST are specified in tree-like YAML files which are
converted to ``ParamsTree`` objects.

In this section, we describe the ``ParamsTree`` objects, the expected structure
of the full parameter tree interpreted by deNEST, and the expected formats and
parameters of each of the subtrees that define the various aspects of the
network and simulation.

Main parameter file
-------------------

To facilitate defining parameters in separate files, ``denest.run`` and
``denest.load_trees`` take as input a path to a YAML file containing the
relative paths of the tree-like YAML files to merge so as to define the full
parameter tree (see the ``params/tree_paths.yml`` file)


The ``ParamsTree`` class
------------------------

The ``ParamsTree`` class is instantiated from tree-like nested dictionaries. At
each node, two reserved keys contain the node's data (called ``'params'`` and
``'nest_params'``). All the other keys are interpreted as named children nodes.

The ``'params'`` key contains data interpreted by deNEST, while the
``'nest_params'`` key contains data passed to NEST without modification.

The ``ParamsTree`` class offers a tree structure with two useful
characteristics:

* **Hierarchical inheritance of ancestor's data**: This provides a concise way
  of defining data for nested scopes. Data common to all leaves may be specified
  once in the root node, while more specific data may be specified further down
  the tree. Data lower within the tree overrides data higher in the tree.
  Ancestor nodes' ``params`` and ``nest_params`` are inherited independently.

* **(Horizontal) merging of trees**: ``ParamsTree`` objects can be merged
  horizontally. During the merging of multiple params trees, the  contents of
  the ``params`` and ``nest_params`` data keys of nodes at the same relative
  position are combined. This allows **splitting the deNEST parameter trees in
  separate files for convenience**, and **overriding the data of a node anywhere
  in the tree while preserving hierarchical inheritance**


An example parameter tree
"""""""""""""""""""""""""

Below is an example of a YAML file with a tree-like structure that can be loaded
and represented by the `ParamsTree` class:

.. code-block:: yaml

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

This file can be loaded into a ParamsTree structure. The leaves of the resulting
ParamsTree and their respective data (``params`` and ``nest_params``) are as
follows. Note the inheritance and override of ancestor data. The nested format
above is more compact and less error prone when there are a lot of shared
parameters between leaves.

.. code-block:: yaml

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


Full parameter tree: expected structure
---------------------------------------

All the aspects of the overall simulation are specified in specific named
subtrees.

The overall `ParamsTree` passed to ``denest.Simulation()`` is expected to have
no data and the following ``ParamsTree`` children:

  * ``simulation`` (``ParamsTree``). Defines input and output paths, and the simulation steps performed. The following parameters (`params` field) are recognized:

      * ``output_dir`` (str): Path to the output directory (default 'output').
      * ``input_dir`` (str): Path to the directory in which input files are searched for for each session. (default 'input')
      * ``sessions`` (list(str)): Order in which sessions are run. Elements of the list should be the name of session models defined in the ``session_models`` parameter subtree (default [])

  * ``kernel`` (``ParamsTree``): Used for NEST kernel initialization. Refer to ``Simulation.init_kernel`` for a description of kernel parameters.

  * ``session_models`` (``ParamsTree``): Parameter tree, the leaves of which define session models. Refer to ``Sessions`` for a description of session parameters.

  * ``network`` (``ParamsTree``): Parameter tree defining the network in NEST. Refer to `Network` for a full description of network parameters.


``"network"`` parameter tree: expected structure
------------------------------------------------

All network parameters are specified in the ``network`` subtree, used to
initialize the ``Network()`` object.

The ``network`` subtree should have no data, and the following ``ParamsTree``
children are expected:

  * ``neuron_models`` (``ParamsTree``). Parameter tree, the leaves of which define neuron models. Each leave is used to initialize a ``Model`` object

  * ``synapse_models`` (``ParamsTree``). Parameter tree, the leaves of which define synapse models. Each leave is used to initialize a ``SynapseModel`` object

  * ``layers`` (``ParamsTree``). Parameter tree, the leaves of which define layers. Each leave is used to initialize  a ``Layer`` or ``InputLayer`` object depending on the value of their ``type`` ``params`` parameter.

  * ``projection_models`` (``ParamsTree``). Parameter tree, the leaves of which define projection models. Each leave is used to initialize a ``ProjectionModel`` object.

  * ``recorder_models`` (``ParamsTree``). Parameter tree, the leaves of which define recorder models. Each leave is used to initialize a ``Model`` object.

  * ``topology`` (``ParamsTree``). ``ParamsTree`` object without children, the ``params`` of which may contain a ``projections`` key specifying all the individual population-to-population projections within the network as a list. ``Projection`` objects  are created from the ``topology`` ``ParamsTree`` object by the ``Network.build_projections`` method. Refer to this method for a description of the ``topology`` parameter.

  * ``recorders`` (``ParamsTree``). ``ParamsTree`` object without children, the ``params`` of which may contain a ``population_recorders`` and a ``projection_recorders`` key specifying all the network recorders. ``PopulationRecorder`` and ``ProjectionRecorder`` objects  are created from the ``recorders`` ``ParamsTree`` object by the ``Network.build_recorders`` method. Refer to this method for a description of the ``recorders`` parameter.


Running a deNEST Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* From Python (_e.g._ in a Jupyter notebook):

  * Using the ``Simulation`` object to run the simulation step by step:

    .. code-block:: python

       import denest

       # Path to the parameter files to use
       params_path = 'params/tree_paths.yml'

       # Override some parameters loaded from the file
       overrides = [

         # Maybe change the nest kernel's settings ?
         {'kernel': {'nest_params': {'local_num_threads': 20}}},

         # Maybe change a parameter for all the projections at once ?
         {'network': {'projection_models': {'nest_params': {
             'allow_autapses': true
         }}}},
       ]

       # Load the parameters
       params = denest.load_trees(params_path, *overrides)

       # Initialize the simulation
       sim = denest.Simulation(params, output_dir='output')

       # Run the simulation (runs all the sessions)
       sim.run()

  * Using the ``denest.run()`` function to run the full simulation at once:

    .. code-block:: python

       import denest

       # Path to the parameter files to use
       params_path = 'params/tree_paths.yml'

       # Override parameters
       overrides = []

       denest.run(params_path, *overrides, output_dir=None)


* From the command line:

    .. code-block:: bash

       python -m denest <tree_paths.yml> [-o <output_dir>]
