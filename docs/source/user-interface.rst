User interface
==============

These are the main functions and classes that comprise deNEST's user
interface.

.. currentmodule:: denest

.. autosummary::
   load_trees
   run
   Simulation
   Session
   Network
   network.Layer
   ParamsTree

.. automodule:: denest
   :members:
   :ignore-module-all:

.. autoclass:: Simulation
   :members:
   :exclude-members: total_time, install_module

.. autoclass:: Network
   :members:
   :exclude-members: build_named_leaves_dict, get_population_recorders, get_projection_recorders, 

.. autoclass:: denest.network.Layer
   :members:
   :exclude-members: position

.. autoclass:: ParamsTree
   :members:
   :exclude-members: position
