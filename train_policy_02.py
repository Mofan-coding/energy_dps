# This script trains a new adaptive policy for energy system planning under uncertainty.
# It saves the trained policy (.pth file) to the results folder for later evaluation.
# Usage: Run this script to generate a new policy for a given scenario and label.

# iter 控制“进化算法优化多少轮”
#batch_size 控制“每轮用多少随机轨迹评估 policy”

import os
import energySim._energy_sim_model as _energy_sim_model
import energySim._energy_sim_params as _energy_sim_params

label = '081002'
scenario = 'slow transition'

model = _energy_sim_model.EnergyModel(
    EFgp=_energy_sim_params.scenarios[scenario][0],
    slack=_energy_sim_params.scenarios[scenario][1],
    costparams=_energy_sim_params.costsAssumptions['Way et al. (2022)'],
)

model.mode = 'policy'
model.policy.train(label, iter=500, batch_size=100, popsize=16, dist=True) #pop size不大于 cpus-per-task
#model.policy.train(label, iter = 2, batch_size = 2,popsize = 2, dist = True)
os.makedirs('results', exist_ok=True)
policy_path = f'results/{label}_{scenario}_policy.pth'
model.policy.save(policy_path)
print(f"Policy saved to {policy_path}")
