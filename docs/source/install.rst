Installation
============

Local
~~~~~

1. Install NEST >= v2.14.0, <3.0 by following the instructions at http://www.nest-simulator.org/installation/.

2. Set up a Python 3 environment and install deNEST with:

   .. code-block:: bash

      pip install denest


Docker
~~~~~~

A Docker image is provided with NEST 2.20 installed, based on
`nest-docker <https://github.com/nest/nest-docker>`_.

1. From within the repo, build the image:

   .. code-block:: bash

      docker build --tag denest .

2. Run an interactive container:

   .. code-block:: bash

      docker run \
        -it \
        --name denest_simulation \
        --volume $(pwd):/opt/data \
        --publish 8080:8080 \
        denest \
        /bin/bash

3. Install deNEST within the container:

   .. code-block:: bash

      pip install -e .

4. Use deNEST from within the container.

For more information on how to use the NEST Docker image, see `nest-docker
<https://github.com/nest/nest-docker>`_.
