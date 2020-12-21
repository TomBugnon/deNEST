## NEST extensions

NB: This Makefile was tested on Ubuntu 18.04 and may require modification on
different systems. Refer to
https://nest.github.io/nest-simulator/extension_modules for indications on how
to install an extension module

#### Use your own module/extension in a simulation:

0- Install NEST==2.20 and make sure the nest variables are sourced (check with `$ which nest`)

1- Install the module in NEST:
  - Run `make` in this directory.
  - (Run `make clean` to remove the build directory.)

2- Make sure the kernel parameters include the module. eg:

  ```
  kernel:
    params:
      ...
      extension_modules: ['htneuronmodule']
    ```

The module will be installed during the kernel initialization.
