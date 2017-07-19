# Spiking VisNet

To run from Python (_e.g._ in a Jupyter notebook):

```python
import nest
import spiking_visnet

# load the full parameter tree
params = spiking_visnet.load_params('params/default.yml')

# This initializes the network in NEST's kernel with the given parameter file
network = spiking_visnet.init(params)
# This creates a simulation object (a series of sessions)
simulation = spiking_visnet.simulation.Simulation(params)
# This runs all the sessions
simulation.run()
# Et voilà !

```

To run directly from the command line:

```bash
  python -m spiking_visnet <param_file.yml>
```
