Overview
========

Definitions
~~~~~~~~~~~

Network

.. code-block::

   *
     We call **network** a full network in NEST, consisting of layers of units with
     specific models, projections of specific types with specific synapse models
     amongst these layers, population recorders (multimeters, spike detectors) and
     projection recorders (weight recorder).


     * The full network is represented in deNEST by the ``Network()`` class

   *
     New NEST **models** (\ **neuron and generator model**\ , **synapse model** or
     **recorder model**\ ) can be specified with arbitrary parameters. During network
     creation, models with specific parameters are created in NEST using a
     ``nest.CopyModel()`` or a ``nest.SetDefaults`` call.


     * Synapse models are represented by the ``SynapseModel`` class in deNEST. All
       other models are represented by the ``Model`` class.
     * Neuron and generator models are specified as leaves of the
       ``network/neuron_models`` parameter subtree (see section below)
     * Synapse models are specified as leaves of the ``network/synapse_models``
       parameter subtree (see "Network parameters" section below)
     * Recorder models are specified as leaves of the ``network/recorder_models``
       parameter subtree (see "Network parameters" section below)

   *
     A **layer** is a NEST topological layer, created with a ``tp.CreateLayer``
     call in NEST. We call **population** all the nodes of the same model
     within a layer.


     * Layers can be of the type ``InputLayer`` when they are composed of generators.
       Input layers are automatically duplicated to contain parrot neurons, so that
       recording of generators' activity is possible.  Additionally, ``InputLayer``\ s
       support setting the state of stimulators (see "session" description).
     * Layers are represented by the ``Layer`` or ``InputLayer`` class in deNEST
     * Layers are specified as leaves of the ``network/layers`` parameter
       subtree (see "Network parameters" section below)

   *
     A **projection model** is a template specifying parameters passed to
     ``tp.ConnectLayers``\ , and that individual projections amongst populations can
     inherit from.


     * Projection models are represented by the ``ProjectionModel`` class in deNEST.
     * Projection models are specified as leaves of the ``network/projection_models``
       parameter subtree (see "Network parameters" section below)

   *
     A **projection** is an individual projection between layers or populations,
     created with a ``tp.ConnectLayers`` call. The parameters passed to
     ``tp.ConnectLayers`` are those of the "projection model" of the specific
     projection.


     * The list of all individual projections within the network is specified in
       the ``'projections'`` parameter of the ``network/topology`` parameter subtree
       (see "Network parameters" section below).

   *
     A **population recorder** is a recorder connected to all the nodes of a given
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
       of the ``network/recorders`` parameter subtree (See "Network parameters"
       section below)

   Simulation
   ~~~


* A **session model** is a template specifying parameters inherited by
  individual sessions.


* session models are specified as leaves of the ``session_models`` parameter
  subtree (see "Simulation parameters" section below)


* A **session** is a period of simulation of a network with specific inputs
  and parameters, and corresponds to a single ``nest.Simulate()`` call. The
  parameters used by a given session are inherited from its session model.


*
  A session's parameters define the operations that may be performed before
  running it:


  * Modifying the state of some units within a popultion
  * Modifying the state of some synapses of a certain projection (TODO)
  * Setting the state of generators within ``InputLayer`` layers from arrays
  * Deactivate the recorders for that session.

*
  Individual sessions are represented by the ``Session`` object in deNEST.
  (see "Simulation parameters" section below)


* A **simulation** is a full experiment. It is represented by the ``Simulation()``
  object in deNEST, which contains a ``Network`` object and a list of ``Session``
  objects.


* The list of sessions run during a simulation is specified by the
  ``sessions`` parameter of the ``simulation`` parameter subtree (eg:
  sessions: ``['warmup', 'noise', 'grating', 'noise', 'grating']``\ ) (see
  "Simulation parameters" section below)

Overview of a full simulation
:raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ :raw-html-m2r:`<del>~</del>`\ ~~~~

A full deNEST simulation consists of the following steps:

   **Initialize simulation** (\ ``Simulation.__init__``\ )

.. code-block::

     **Initialize kernel**\ : (\ ``Simulation.init_kernel``\ )


     #. Set NEST kernel parameters
     #. Set seed for NEST's random generator.

.. code-block::

     **Create network**\ :


     #. Initialize the network objects (\ ``Network.__init__``\ )
     #. Create the objects in NEST (\ ``Network.__create__``\ )

.. code-block::

     **Initialize the sessions** (\ ``Session.__init__``\ )

.. code-block::

     **Save the simulation's metadata**


     * Create the output directory
     * Save parameters
     * Save git hash
     * Save session times
     * Save network metadata (TODO)
     * Save session metadata (TODO)


**Run the simulation** (\ ``Simulation.__run__``\ )

.. code-block::

     Run each session in turn: (\ ``Session.__run__``\ )


     #.
        Initialize session (\ ``Session.initialize``\ )


        * (Possibly) reset the network
        * (Possibly) inactivate recorders for the duration of the session
        * (Possibly) Change some of the network's parameters:

          #. Change neuron parameters (\ ``Network.change_unit_states``\ )
          #. Change synapse parameters (\ ``Network.change_synapse_states``\ )

        * Set InputLayer's state from input arrays

     #.
        Call ``nest.Simulate()``.
