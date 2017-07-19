# Spiking VisNet

To run from Python (_e.g._ in a Jupyter notebook):

```python
import nest
import spiking_visnet

# This initializes the network in NEST's kernel with the given parameter file
network = spiking_visnet.init('params/default.yml')

# Simulate the network for 1000 ms
nest.Simulate(1000)
```

To run directly from the command line:

```bash
  python -m spiking_visnet <param_file.yml> <time_in_ms>
```
