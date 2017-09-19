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
import spiking_visnet

# USER: Specify the path to a simulation parameter file
params_path = 'params/default.yml'

# USER can specified input and output that overwrite those in parameter files.
input_ = None # Path to input np-array or input directory. Overwrites params.
output_dir = None # Path to output directory. Overwrites params.

params = spiking_visnet.load_params(params_path)
sim = spiking_visnet.Simulation(params, input_dir=input_dir, output_dir=output_dir)
sim.run()
sim.save()
```

To run directly **from the command line**:

```bash
    python -m spiking_visnet <param_file.yml> [-i <input>] [-o <output>]
```
eg:
```bash
  python -m spiking_visnet params/default.yml -i my_np_array.npy -o output
```
