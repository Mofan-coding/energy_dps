# Capacity expansion under uncertainty project

This project applies direct policy search (DPS) to long-term energy system planning.

## Structure of the project

- .venv/: is where the environment energy_dps_env is stored
- energySim/: is where the helper functions are
    - _check_energy_sim.py: is used to show how the energy system model is as close as possible to results obtained in Way et al., Joule, 2022
    - _energy_sim.py: contains the class EnergyModel (all the functions to simulate and plot)
    - _energy_sim_params.py: hardcodes the parameters used in Way et al., Joule, 2022
    - _policy.py: implementation of policies and training algorithms (in this case Separable Natural Evolutionary Strategies - Wierstra et al., 2014)
    - fast_transition_policy_new_.pth (sample policy output, torch file)
    - histElec.txt: parameter of EnergyModel, cost of grid expansion assuming continuing historical trend
- figures/: folder where gifs are saved
- LICENSE (MIT - Carnegie - to be changed?)
- requirements.txt (python environment)
- testing_results.py: run this to simulate the differences in cost between static and adaptive
- training_new_policies.py: run this script for training (uses 4 cpus as of now, be careful)

## check that the model runs and results can be obtained
1) create the environment, in the terminal from the main folder:
```
python -m venv .venv/energy_dps_env
```
or, using an absolute path:
```
python -m venv PATH_TO_ENV
```

2) activate the environment and install the requirements, from the terminal:
```
source PATH_TO_ENV/bin/activate
```
in the case you followed above:
```
source .venv/energy_dps_env/bin/activate (use this!)
```
followed by:
```
pip install -r requirements.txt
```
3) in the main folder run the following command from the terminal:
```
python testing_results.py
```