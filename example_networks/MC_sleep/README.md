## Default parameter files for the network described in :

"Sleep and wake in a model of the thalamocortical system with Martinotti cells", Tom Bugnon, William G. P. Mayner, Chiara Cirelli and Giulio Tononi

### Running the simulation

1. Install NEST v2.20

2. Make the updated ht_neuron model available to NEST by installing the extension module at `../extension_modules`

2. Install the `MC_sleep` branch of deNEST with `pip install git+https://github.com/tombugnon/denest.git@MC_sleep`

3. From this directory, run with `python run.py`. This will run the list of sessions described in `./params/sim/simulation.yml`: 

  - "warmup",
  - "noise" (corresponding to the "spontaneous wake" mode),
  - "grating" (mimicking presentation of a drifting grating),
  - "sleep_transition" (at the start of which the wake-to-sleep parameter changes are applied)
  - "sleep" (corresponding to the "sleep" mode)

The session parameters are described in `./params/sim/sessions.yml`. Output is saved by default at at `./output`.

See <denest.readthedocs.io> for more information on how to run the network in multiple conditions


### All simulations described in the paper:

Parameter files for all the simulations described in the paper are available at <https://github.com/TomBugnon/MC_sleep_replicate>
