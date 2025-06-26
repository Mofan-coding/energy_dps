#This script loads a trained policy and evaluates its performance under uncertainty.
# It runs multiple simulations using the saved policy and reports the results.
# Usage: Run this script after training a policy to test and analyze its effectiveness.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import energySim._energy_sim_model as _energy_sim_model
import energySim._energy_sim_params as _energy_sim_params

label = '062501'
scenario = 'fast'
nsim = 10  # Number of evaluation runs
savegif = True

model = _energy_sim_model.EnergyModel(
    EFgp=_energy_sim_params.scenarios[scenario][0],
    slack=_energy_sim_params.scenarios[scenario][1],
    costparams=_energy_sim_params.costsAssumptions['Way et al. (2022)'],
)

policy_path = f'results/{label}_{scenario}_policy.pth'
model.mode = 'policy'
model.policy.load(policy_path)

costs = []
for n in range(nsim):
    print(f"Simulating run {n+1}/{nsim} ...")
    costs.append(1e-12 * model.simulate())
    if savegif and n == 0:
        model.make_gif(f'{label}_{scenario}_policy_run{n}')

df = pd.DataFrame({'Scenario': scenario, 'Net Present Cost [trillion USD]': costs})

# Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(data=df, y='Net Present Cost [trillion USD]')
plt.title(f'Policy Evaluation: {scenario} (label {label})')
plt.ylabel('Net Present Cost [trillion USD]')
plt.tight_layout()
plt.savefig(f'results/{label}_{scenario}_policy_boxplot.png')
plt.show()

# Optionally save results
df.to_csv(f'results/{label}_{scenario}_policy_eval.csv', index=False)