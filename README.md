# Spiking VisNet

To run from Python (_e.g._ in a Jupyter notebook):

```python
import nest
import spiking_visnet

# Load the network parameters
params = spiking_visnet.load_params('params/default.yml')

# Initializes the network in NEST's kernel with the given parameter file
network = spiking_visnet.init(params)
# Create a simulation object (a series of sessions)
simulation = spiking_visnet.simulation.Simulation(params)
# Runs all the sessions
simulation.run()
# Et voilà !
```

To run directly from the command line:

```bash
  python -m spiking_visnet <param_file.yml>
```
