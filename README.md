# Spiking VisNet

## Before running the simulation

### Preprocess stimuli  
Note: the stimuli dimension (resolution and number of filters) should be
compatible with (_i.e._, greater than the dimensions of) your network.

To preprocess, run
```bash
python3 -m spiking_visnet.preprocess \
        -p <preprocessing_params> \
        -n <simulation_params> \
        [-input <input_dir>]
```
For example:
```bash
python3 -m spiking_visnet.preprocess \
        -p spiking_visnet/preprocess/params/default.yml \
        -n params/default.yml
```

See `spiking_visnet/preprocess/README.md` for details.

### Change the session's stimuli
Manually modify the session parameters so that sessions use stimulus sequences
compatible with the network's dimensions. You can use the default stimulus
sequence created during preprocessing; _e.g._ put the following in
`sessions_df.yml`:
```yaml
session_stims: 'stim_df_set_df_res_100x100_contrastnorm_filter_o2.yml`
```

Alternatively, you can also specify the input that will be used for all
sessions as a command line argument. In that case, you don't need to modify
the session parameters file.

## Run the simulation

To run **from Python** (_e.g._ in a Jupyter notebook):
```python
import spiking_visnet

# The simulation parameter file to use.
params_path = 'params/default.yml'

# Load the parameters.
params = spiking_visnet.load_params(params_path)

# You can override the intput and output settings in the parameter file by
# passing them when you create the simulation:

input_dir = None  # Path to an input directory. NOTE: This can also be a path
                  # to a saved NumPy array.
output_dir = None # Path to output directory.

# Create the simulation.
sim = spiking_visnet.Simulation(params, input_dir=input_dir, output_dir=output_dir)

# Run the simulation.
sim.run()

# Save the results.
sim.save()
```

To run directly **from the command line**:
```bash
python -m spiking_visnet <param_file.yml> [-i <input>] [-o <output>]
```
For example:
```bash
mkdir output
python -m spiking_visnet params/default.yml -i my_input_array.npy -o output
```
