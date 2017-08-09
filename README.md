# Spiking VisNet


### Before running the simulation:

**1: Preprocess some stimuli**:  
-> their dimension (resolution and
    number of filters) should be compatible with (eg superior to that of) the network you are going to run.

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

Alternatively, you can also specify the input that will be used for all sessions as a command line argument. In that case, you don't need to modify the session parameters file. 

### To run the simulation:

To run **from Python** (_e.g._ in a Jupyter notebook):

```python
import nest
import spiking_visnet

# USER: Specify the path to a simulation parameter file
params_path = 'params/default.yml'

# USER can introduce the command line arguments to the tree
user_input = None # Path to input np-array. If specified, overwrites param file default
user_savedir = None # Path to savedir. overwrites default if specified

# Load the network parameters
params = spiking_visnet.load_params(path)
# Incorporate user defined arguments to the params tree
incorporate_user_args(params, user_input=user_input, savedir=user_savedir)
# Initializes the network in NEST's kernel with the given parameter file
network = spiking_visnet.init(params)
# Create a simulation object (a series of sessions)
simulation = spiking_visnet.Simulation(network, params)
# Runs all the sessions
simulation.run(run)
# Save the results
spiking_visnet.save_all(network, params)
# Et voilà ! Great success.
```

To run directly **from the command line**:

```bash
    python -m spiking_visnet <param_file.yml> [-i <input>] [-s <savedir>]
```
eg:
```bash
  python -m spiking_visnet params/default.yml -i my_np_array.npy -s my_saving_directory
```
