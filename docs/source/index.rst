.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   install
   overview
   example-params

deNEST
======

**deNEST is a Python library for specifying networks and running simulations using the NEST simulator**
(https://nest-simulator.org).

deNEST allows the user to concisely specify large-scale networks and
simulations in hierarchically-organized **declarative parameter files**.

From these parameter files, a network is instantiated in NEST (neurons &
their projections), and a simulation is run in sequential steps ("sessions"),
during which the network parameters can be modified and the network can be
stimulated, recorded, etc.

Some advantages of the declarative approach:

* Parameters and code are separated
* Simulations are easier to reason about, reuse, and modify
* Parameters are more readable and succinct
* Parameter files can be easily version controlled and diffs are smaller and more interpretable
* Clean separation between the specification of the "network" (the simulated
  neuronal system) and the "simulation" (structured stimulation and recording
  of the network), which facilitates running different experiments using the
  same network
* Parameter exploration is more easily automated

To learn how to use deNEST, please see the :doc:`overview` section and the Jupyter notebook tutorials:

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/quickstart
   tutorials/network_object
   tutorials/modify_network
   tutorials/simulation_object
   tutorials/load_simulation_output
   tutorials/perform_parameter_exploration
   tutorials/version_control_params

.. include:: ./install.rst
.. include:: ./overview.rst
.. include:: ./example-params.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   user-interface
