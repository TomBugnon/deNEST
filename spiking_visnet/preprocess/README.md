# Preprocessing of raw input movies

To run from the command line,

```
python3 -m spiking_visnet.preprocess -i <input_dir> -p <preprocessing_params> -n <network_params>
```

where:

- `<input_dir>` : the (relative or absolute) path to the directory in which all
 	the movie inputs are saved. input_dir should at least contain the subdirectory *raw_input*.

- `<preprocessing_params>`: path to the yml file containing the preprocessing parameters. eg: _preprocess/params/default.yml_
- `<network_params>`: path to the file containing the parameters defining a network. eg: _params/default.yml_

This command will preprocess all the movies in the subdirectory:
*`<input_dir>`/raw_input* and add their preprocessed version to the subdirectory
*`input_dir`/preprocessed_input/`<preprocessing_dir>`*, where `<preprocessing_dir>` is a string depending on the preprocessing parameters.

If *`input_dir`/preprocessed_input/`<preprocessing_dir>`* already exists, the preprocessing is only done on the movies that haven't been processed already.

Finally, if some input sets are defined in *`input_dir`/raw_input_sets*, the corresponding sets are definied in *`input_dir`/preprocessed_input_sets*

Example:
```
python3 -m spiking_visnet.preprocess -i /Users/Tom/temp/input_dir -p spiking_visnet/preprocess/params/default.yml -n params/default.yml
```
