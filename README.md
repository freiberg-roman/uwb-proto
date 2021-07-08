# Ultrawideband Map of Measurement Noise prototype implementation

Simple numyp and scipy implementation as a proof of concept for uwb noise map usage to improve location.
Whole procedure is explained in the [UWB](https://ieeexplore.ieee.org/document/6514113) paper.

### Installation

Python 3.9+ is required.
Optional: Create conda environment with 

    conda create -n uwb python=3.9
    # switch to environment
    conda activate uwb

Standard installation with pip

    git clone https://github.com/freiberg-roman/uwb-proto.git
    cd uwb
    pip install -e ".[dev]"

Test installation by running
    
    python -m uwb.examples.main
    # or
    pytest

### Usage

Check out the notebooks for example use cases.
