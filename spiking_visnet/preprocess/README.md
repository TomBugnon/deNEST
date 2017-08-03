# Preprocessing of raw input movies

## To run from the command line:

```
python3 -m spiking_visnet.preprocess [-i <input_dir>] -p <preprocessing_params> -n <simulation_params>
```

### Args:

- `<input_dir>` :  Path to the directory in which all the movie inputs are saved.
    input_dir should at least contain the subdirectory *raw_input*.  
    -> __Default from config file__

- `<preprocessing_params>`: path to the yml file containing the preprocessing parameters.  
    -> eg: __'preprocess/params/default.yml'__

- `<network_params>`: path to the simulation parameter file containing then network.    
    Used to obtain the set of filters applied to the raw input, and the resolution to which it is downsampled.  
    -> eg: __'params/default.yml'__

### Effects
- Preprocess all the movies in *`input_dir/raw_input`*  
  Save in *`<input_dir>/preprocessed_input/<preprocessing_subdir>`*,  
  Where `<preprocessing_subdir>` is a string describing the preprocessing pipeline.

  If *`<input_dir>/preprocessed_input/<preprocessing_dir>`* already exists, the preprocessing is only done on the movies that haven't been processed already.

- If some input sets are defined in *`<input_dir>/raw_input_sets`*, the corresponding sets are created in *`<input_dir>/preprocessed_input_sets`*

- Finally, creates a default stimulus yaml file for this preprocessing pipeline in *`<input_dir>/stimuli`* that
    defines the input during a session.


### Usage example:
```
python3 -m spiking_visnet.preprocess -i input_dir spiking_visnet/preprocess/params/default.yml -n params/default.yml
```

##### Before running the network:

Make sure that the session parameters of the simulation point to a stimulus file compatible with the network dimensions.
