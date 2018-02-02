# Spiking VisNet

## Parameters

The full simulation is defined by a merged parameter tree. Parameters originate
from either `overrides` (tree-like) or files containing
paths to tree-like files.

During a `nets.run()` call, the following parameters are combined
(from lowest to highest precedence):
- ``DEFAULT_PARAMS_PATH``: Path to USER defined default parameters, possibly
    specified in `user_config.py` (lowest precedence)
- ``path``: Path to the main simulation parameters passed during a direct
    `run()` call or a CLI package call.
- ``overrides``: Overrides passed by USER during a direct `run()` call. During a
    CLI package call, these overrides possibly contain the input and output CLI
    optional arguments.
- ``USER_OVERRIDES``: Tree-like overrides defined in `user_config.py`. (highest precedence)

## Inputs to the network

__Note__: the stimuli dimension (resolution and number of filters) should be
compatible with (i.e., greater than the dimensions of) your network. The
expected dimension of the arrays is: __time x nfilters x nrows x ncols__.

The `input_dir` simulation parameter can be either of the following:
- A path to an array file: In which case this is the stimulus showed to the
network for all the sessions.
- A path to a directory: In which case the movie presented to the network during
a session is a 'stimulus set', the concatenation of arrays whose paths are
obtained from the `session_stims` session parameter. The input directory
`input_dir` should then have the expected structure and the `session_stims` key
should contain the name of a file existing in the 'stimuli' subdirectory of
`input_dir`

##### For 'Stimulus set' input (as opposed to numpy): (TODO update)

###### Preprocessing stimuli

TODO: Update

To preprocess, run
```bash
python3 -m nets.preprocess \
        -p <preprocessing_params> \
        -n <simulation_params> \
        [-input <input_dir>]
```
For example:
```bash
python3 -m nets.preprocess \
        -p nets/preprocess/params/default.yml \
        -n params/default.yml
```

See `nets/preprocess/README.md` for details.

###### Change the session's stimuli

Manually modify the session parameters so that sessions use stimulus sequences
compatible with the network's dimensions. You can use the default stimulus
sequence created during preprocessing; _e.g._ put the following in
`sessions_df.yml`:

```yaml
session_stims: 'stim_df_set_df_res_100x100_contrastnorm_filter_o2.yml'
```

## Run the simulation

#### To run **from Python** (_e.g._ in a Jupyter notebook):

**Using the ``Simulation`` object to run the simulation step by step:**

```python
import nets

# The simulation parameter file to use.
params_path = 'params/default.yml'

# User defined list of tree-like overrides
overrides = []

# Load the parameters.
params = nets.load_params(params_path, *overrides)

# You can override the input and output settings in the parameter file by
# passing them when you create the simulation:

input_dir = None  # Path to an input directory. NOTE: This can also be a path
                  # to a saved NumPy array.
output_dir = None # Path to output directory.

# Create the simulation.
sim = nets.Simulation(params, input_dir=input_dir, output_dir=output_dir)

# Run the simulation.
sim.run()

# Save the results, dump the connections, etc.
sim.save()
sim.dump()
```

**Using the ``nets.run()`` function to run the full simulation at
once:**

```python
import nets

# The simulation parameter file to use.
params_path = 'params/default.yml'

# User defined list of tree-like overrides
overrides = []

nets.run(params_path, *overrides, input_dir=None, output_dir=None)
```

Note:
- `DEFAULT_PARAMS_PATH` and `USER_OVERRIDES` parameters are applied.
- Control which stimulation steps are ran in the simulation parameters.


#### To run directly **from the command line**:

```bash
python -m nets <param_file.yml> [-i <input>] [-o <output>]
```

Note:
- `DEFAULT_PARAMS_PATH` and `USER_OVERRIDES` parameters are applied.
- Calls `nets.run()`.
