# PSA Simulation

This repository contains a pressure swing adsorption (PSA) cycle simulator with optional process optimization support.

## Requirements

Install Python dependencies with:

```
pip install -r requirements.txt
```

> **Note:** The optimization mode depends on [`cyipopt`](https://github.com/mechmotum/cyipopt), which in turn requires a working IPOPT installation. Ensure IPOPT is available on your system before running the optimizer.

## Usage

Run the baseline PSA evaluation:

```
python Simulation.py
```

Optimize the high, intermediate, and purge pressures (using IPOPT):

```
python Simulation.py --optimize
```

Add `--no-plots` to skip generating concentration profile figures. Targets for purity and recovery can be customised with `--purity-target` and `--recovery-target`.
