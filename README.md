# SpikeML

![Build Status](https://img.shields.io/github/workflow/status/yourusername/spikeml/Python%20package?label=build&logo=github)
![PyPI](https://img.shields.io/pypi/v/spikeml?logo=python&label=PyPI)

A Python framework for spiking neural networks, with stochastic dynamics applied to embodied cognitive modeling, and bio-inspired Machine Learning.

### Goals

Main goals of this project are:
    - Support development and testing-validation of theories of stochastic and chaotic neural dynamics, brain function, and embodied-adaptive behavior
    - Provide alternative path to scalable Machine-Learning using model inspired in biological system

### Key Motivations

Key ideas push forward:
    - Stochastic neural dynamics is the most promissing approach to understand brain function and adaptive behavior
    - Backprogation trained neural network models are a poor model of natural brain
    - Mean-firing-rate models don't capture important aspects of neural dynamics, some of which are sometimes required for correct function
    - Detailed models of neurons or synapsis are not required to capture functional behavior

## Features

- Spiking neuron models with stochastic dynamics
- Embodied cognition simulations
- Cognitive modeling tools
- Experiment data export
- Experiment data visualization
- Extensible Pythonic API

## Components


## Installation
Pre-release (TestPyPI):
```bash
pip install --index-url https://test.pypi.org/simple/spikeml
```

Stable release (PyPI):
```bash
pip install spikeml
```

# TODO:

- Framework:
  - Results.render()
  - SSensor homeostasis
  - .render({'t0':0, 't':1000}) + plot axis
  - FanConcat test

- Experiments:
  - xx

## License
MIT License
