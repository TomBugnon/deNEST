# Spiking VisNet


### Before running the simulation:

**1: Preprocess some stimuli**:  
-> their dimension (resolution and
    number of filters) should be compatible with the network you are going to run.

To preprocess: (see *spiking_visnet/preprocess/README.md*)

```bash
python3 -m spiking_visnet.preprocess -p <preprocessing_params> -n <simulation_params> [-input <input_dir>]
```
eg:
```bash
python3 -m spiking_visnet.preprocess -p spiking_visnet/preprocess/params/default.yml -n params/default.yml
```

**2: Change the session's stimuli**  
Manually modify the session parameter tree so that sessions use stimulus sequences compatible with the network's dimensions.  
You can use the default stimulus sequence created during preprocessing:  
eg: `session_stims: 'stim_df_set_df_res_100x100_contrastnorm_filter_o2.yml` in *sessions_df.yml*


### To run the simulation:

To run **from Python** (_e.g._ in a Jupyter notebook):

```python
import nest
import spiking_visnet

# Load the network parameters
params = spiking_visnet.load_params('params/default.yml')

# Initializes the network in NEST's kernel with the given parameter file
network = spiking_visnet.init(params)
# Create a simulation object (a series of sessions)
simulation = spiking_visnet.Simulation(params, network)
# Runs all the sessions
simulation.run(run)
# Et voilà !
```

To run directly **from the command line**:

```bash
  python -m spiking_visnet <param_file.yml>
```
eg:
```bash
  python -m spiking_visnet params/default.yml
```
