.. deNEST documentation master file, created by
   sphinx-quickstart on Mon Jun 22 18:23:47 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*****
deNEST: A declarative frontend for specifying networks and running simulations in NEST
*****

deNEST is a python library for specifying networks and running simulations using
the NEST simulator (nest-simulator.org ).

deNEST allows the user to fully specify large scale
networks and simulation characteristics in separate, trackable and
hierarchically organized declarative parameter files.

From those parameter files, a network is instantiated in NEST (neuron layers,
simulator layers, projections amongst layers), and a simulation is run in
multiple separate steps ("sessions") before which the network can be modified.

The declarative approach facilitates version control and sharing of the
parameter files, while decoupling the "network" and "simulation" parameters
facilitates running the same network in multiple conditions.

A parameter tree specifying a full simulation may look like the following:

.. code-block:: yaml

  params: {}
  nest_params: {}
  session_models:
    params:
      reset_network: false
      record: true
      shift_origin: false
    nest_params: {}
    even_rate:
      params:
        simulation_time: 50.0
        unit_changes:
        - layers:
          - input_layer
          population_name: input_exc
          change_type: constant
          from_array: false
          nest_params:
            rate: 100.0
      nest_params: {}
    warmup:
      params:
        reset_network: true
        record: false
        simulation_time: 50.0
        unit_changes:
        - layers:
          - l1
          population_name: null
          change_type: constant
          from_array: false
          nest_params:
            V_m: -70.0
        - layers:
          - input_layer
          population_name: input_exc
          change_type: constant
          from_array: false
          nest_params:
            rate: 100.0
      nest_params: {}
    arbitrary_rate:
      params:
        simulation_time: 50.0
        unit_changes:
        - layers:
          - input_layer
          population_name: input_exc
          change_type: constant
          from_array: true
          nest_params:
            rate: ./input_layer_rates_5x5x1.npy
      nest_params: {}
  simulation:
    params:
      sessions:
      - warmup
      - even_rate
      - arbitrary_rate
      output_dir: ./output
      input_dir: ./params/input
    nest_params: {}
  kernel:
    params:
      extension_modules: []
      nest_seed: 94
    nest_params:
      local_num_threads: 20
      resolution: 1.0
      print_time: true
      overwrite_files: true
  network:
    params: {}
    nest_params: {}
    neuron_models:
      params: {}
      nest_params: {}
      ht_neuron:
        params:
          nest_model: ht_neuron
        nest_params:
          g_peak_NaP: 0.5
          g_peak_h: 0.0
          g_peak_T: 0.0
          g_peak_KNa: 0.5
          g_KL: 1.0
          E_rev_NaP: 55.0
          g_peak_AMPA: 0.1
          g_peak_NMDA: 0.15
          g_peak_GABA_A: 0.33
          g_peak_GABA_B: 0.0132
          instant_unblock_NMDA: true
          S_act_NMDA: 0.4
          V_act_NMDA: -58.0
        cortical_inhibitory:
          params: {}
          nest_params:
            theta_eq: -53.0
            tau_theta: 1.0
            tau_spike: 0.5
            tau_m: 8.0
          l1_inh:
            params: {}
            nest_params: {}
          l2_inh:
            params: {}
            nest_params: {}
        cortical_excitatory:
          params: {}
          nest_params:
            theta_eq: -51.0
            tau_theta: 2.0
            tau_spike: 1.75
            tau_m: 16.0
          l1_exc:
            params: {}
            nest_params: {}
          l2_exc:
            params: {}
            nest_params: {}
      input_exc:
        params:
          nest_model: poisson_generator
        nest_params: {}
    layers:
      params:
        type: null
      nest_params:
        rows: 5
        columns: 5
        extent:
        - 8.0
        - 8.0
        edge_wrap: true
      input_area:
        params:
          type: InputLayer
          add_parrots: true
        nest_params: {}
        input_layer:
          params:
            populations:
              input_exc: 1
          nest_params: {}
      l1_area:
        params: {}
        nest_params: {}
        l1:
          params:
            populations:
              l1_exc: 2
              l1_inh: 1
          nest_params: {}
      l2_area:
        params: {}
        nest_params: {}
        l2:
          params:
            populations:
              l2_exc: 2
              l2_inh: 1
          nest_params: {}
    synapse_models:
      params: {}
      nest_params: {}
      static_synapse:
        params:
          nest_model: static_synapse_lbl
        nest_params: {}
        input_synapse_NMDA:
          params:
            target_neuron: ht_neuron
            receptor_type: NMDA
          nest_params: {}
        input_synapse_AMPA:
          params:
            target_neuron: ht_neuron
            receptor_type: AMPA
          nest_params: {}
      ht_synapse:
        params:
          nest_model: ht_synapse
          target_neuron: ht_neuron
        nest_params: {}
        GABA_B_syn:
          params:
            receptor_type: GABA_B
          nest_params: {}
        AMPA_syn:
          params:
            receptor_type: AMPA
          nest_params: {}
        GABA_A_syn:
          params:
            receptor_type: GABA_A
          nest_params: {}
        NMDA_syn:
          params:
            receptor_type: NMDA
          nest_params: {}
    topology:
      params:
        projections:
        - source_layers:
          - input_layer
          source_population: parrot_neuron
          target_layers:
          - l1
          target_population: l1_exc
          projection_model: input_projection_AMPA
        - source_layers:
          - input_layer
          source_population: parrot_neuron
          target_layers:
          - l1
          target_population: l1_inh
          projection_model: input_projection_AMPA
        - source_layers:
          - input_layer
          source_population: parrot_neuron
          target_layers:
          - l1
          target_population: l1_inh
          projection_model: input_projection_NMDA
        - source_layers:
          - l1
          source_population: l1_exc
          target_layers:
          - l1
          target_population: l1_exc
          projection_model: horizontal_exc
        - source_layers:
          - l1
          source_population: l1_exc
          target_layers:
          - l1
          target_population: l1_inh
          projection_model: horizontal_exc
        - source_layers:
          - l1
          source_population: l1_exc
          target_layers:
          - l2
          target_population: l2_exc
          projection_model: FF_exc
        - source_layers:
          - l1
          source_population: l1_exc
          target_layers:
          - l2
          target_population: l2_inh
          projection_model: FF_exc
      nest_params: {}
    recorder_models:
      params: {}
      nest_params:
        record_to:
        - file
        - memory
        withgid: true
        withtime: true
      spike_detector:
        params:
          nest_model: spike_detector
        nest_params: {}
      weight_recorder:
        params:
          nest_model: weight_recorder
        nest_params:
          record_to:
          - file
          - memory
          withport: false
          withrport: true
      multimeter:
        params:
          nest_model: multimeter
        nest_params:
          interval: 1.0
          record_from:
          - V_m
    recorders:
      params:
        population_recorders:
        - layers: []
          populations: []
          model: multimeter
        - layers:
          - l2
          populations:
          - l2_inh
          model: multimeter
        - layers: null
          populations:
          - l2_exc
          model: multimeter
        - layers:
          - l1
          populations: null
          model: multimeter
        - layers: null
          populations: null
          model: spike_detector
        projection_recorders:
        - source_layers:
          - input_layer
          source_population: parrot_neuron
          target_layers:
          - l1
          target_population: l1_exc
          projection_model: input_projection_AMPA
          model: weight_recorder
        - source_layers:
          - l1
          source_population: l1_exc
          target_layers:
          - l1
          target_population: l1_exc
          projection_model: horizontal_exc
          model: weight_recorder
      nest_params: {}
    projection_models:
      params:
        type: topological
      nest_params:
        allow_autapses: false
        allow_multapses: false
        allow_oversized_mask: true
      horizontal_inh:
        params: {}
        nest_params:
          connection_type: divergent
          synapse_model: GABA_A_syn
          mask:
            circular:
              radius: 7.0
          kernel:
            gaussian:
              p_center: 0.25
              sigma: 7.5
          weights: 1.0
          delays:
            uniform:
              min: 1.75
              max: 2.25
      input_projection:
        params: {}
        nest_params:
          connection_type: convergent
          mask:
            circular:
              radius: 12.0
          kernel: 0.8
          weights: 1.0
          delays:
            uniform:
              min: 1.75
              max: 2.25
        input_projection_AMPA:
          params: {}
          nest_params:
            synapse_model: input_synapse_AMPA
        input_projection_NMDA:
          params: {}
          nest_params:
            synapse_model: input_synapse_NMDA
      horizontal_exc:
        params: {}
        nest_params:
          connection_type: divergent
          synapse_model: AMPA_syn
          mask:
            circular:
              radius: 12.0
          kernel:
            gaussian:
              p_center: 0.05
              sigma: 7.5
          weights: 1.0
          delays:
            uniform:
              min: 1.75
              max: 2.25
      FF_exc:
        params: {}
        nest_params:
          connection_type: convergent
          synapse_model: AMPA_syn
          mask:
            circular:
              radius: 12.0
          kernel: 0.8
          weights: 1.0
          delays:
            uniform:
              min: 1.75
              max: 2.25


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   install.rst
   overview.rst

.. toctree::
    :maxdepth: 4
    :hidden:
    :caption: API Reference
    :glob:

    api/*

# .. toctree::
#    :maxdepth: 1
#    :hidden:
#    :caption: User Interface
#
#    user-interfaces.rst
#    array.rst
#    bag.rst
#    dataframe.rst
#    delayed.rst
#    futures.rst
#    Machine Learning <https://ml.dask.org>
#    best-practices.rst
#    api.rst
#
# .. toctree::
#    :maxdepth: 1
#    :hidden:
#    :caption: Scheduling
#
#    scheduling.rst
#    Distributed Scheduling <https://distributed.dask.org/>
#
# .. toctree::
#    :maxdepth: 1
#    :hidden:
#    :caption: Diagnostics
#
#    understanding-performance.rst
#    graphviz.rst
#    diagnostics-local.rst
#    diagnostics-distributed.rst
#    debugging.rst
#
# .. toctree::
#    :maxdepth: 1
#    :hidden:
#    :caption: Help & reference
#
#    develop.rst
#    changelog.rst
#    configuration.rst
#    configuration-reference.rst
#    educational-resources.rst
#    presentations.rst
#    cheatsheet.rst
#    spark.rst
#    caching.rst
#    graphs.rst
#    phases-of-computation.rst
#    remote-data-services.rst
#    gpu.rst
#    cite.rst
#    funding.rst
#    logos.rst
#
# .. _`Anaconda Inc`: https://www.anaconda.com
# .. _`3-clause BSD license`: https://github.com/dask/dask/blob/master/LICENSE.txt
#
# .. _`#dask tag`: https://stackoverflow.com/questions/tagged/dask
# .. _`GitHub issue tracker`: https://github.com/dask/dask/issues
# .. _`gitter chat room`: https://gitter.im/dask/dask
# .. _`xarray`: https://xarray.pydata.org/en/stable/
# .. _`scikit-image`: https://scikit-image.org/docs/stable/
# .. _`scikit-allel`: https://scikits.appspot.com/scikit-allel
# .. _`pandas`: https://pandas.pydata.org/pandas-docs/version/0.17.0/
# .. _`distributed scheduler`: https://distributed.dask.org/en/latest/
