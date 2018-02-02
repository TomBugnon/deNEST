# Preprocessing of input

To run from the command line:
```bash
python3 -m spiking_visnet.preprocess \
        [-i <input_dir>] \
        -p <preprocessing_params> \
        -n <simulation_params>
```

### Arguments

- `<input_dir>`: Path to the directory in which all the movie inputs are saved.
  The directory must contain the subdirectory `raw_input`.  
- `<preprocessing_params>`: Path to the YAML file containing the preprocessing parameters.
  _Example:_ `'preprocess/params/default.yml'`
- `<network_params>`: Path to the simulation parameter file containing then
  network. Used to obtain the set of filters applied to the raw input, and
  the resolution to which it is downsampled.
  _Example:_ `'params/default.yml'`

### Effects

The command above
1. Preprocesses all the input in `<input_dir>/raw_input`  
2. Saves the results in
   `<input_dir>/preprocessed_input/<preprocessing_subdir>`, where
   `<preprocessing_subdir>` is a string describing the preprocessing pipeline
   that was used. If `<input_dir>/preprocessed_input/<preprocessing_dir>`
   already exists, the preprocessing is only done on the input that hasn't
   already been processed.
3. If input sets are defined in `<input_dir>/raw_input_sets`, the corresponding
   sets are created in `<input_dir>/preprocessed_input_sets`.
4. Finally, a default stimulus YAML file is created for this preprocessing
   pipeline in `<input_dir>/stimuli` that defines the input during a session.

### Example

```bash
python3 -m spiking_visnet.preprocess \
        -i input_dir spiking_visnet/preprocess/params/default.yml \
        -n params/default.yml
```

### NOTE:

Make sure that the session parameters of the simulation point to a stimulus
file compatible with the network dimensions.
