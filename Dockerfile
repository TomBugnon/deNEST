FROM nestsim/nest:2.20.0

RUN apt-get update && apt-get install -y --no-install-recommends \
  # For `make`
  build-essential \
  # For `codecov`
  curl \
  # Required by deNEST
  git
