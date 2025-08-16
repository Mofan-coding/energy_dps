
import os
import json
import numpy as np
import energySim._energy_sim_model as _energy_sim_model
import energySim._energy_sim_params as _energy_sim_params

label = '081201'
scenario = 'fast transition'
policy_path = f'results/{label}_{scenario}_policy.pth'
nsim = 100
gt_clip = 0.3
hidden_size = 16
input_norm = False



model = _energy_sim_model.EnergyModel(
    EFgp=_energy_sim_params.scenarios[scenario][0],
    slack=_energy_sim_params.scenarios[scenario][1],
    costparams=_energy_sim_params.costsAssumptions['Way et al. (2022)'],
    gt_clip=gt_clip,
    hidden_size=hidden_size,
    input_norm=input_norm
)
model.mode = 'policy'
model.policy.load(policy_path)

all_inputs = []

orig_get_action = model.policy.get_action
def logging_get_action(pol_input):
    arr = np.asarray(pol_input, dtype=float)
    if np.all(np.isfinite(arr)) and arr.shape == (5,):
        all_inputs.append(arr.tolist())
    return orig_get_action(pol_input)

model.policy.get_action = logging_get_action

np.random.seed(0)
for n in range(nsim):
    _ = model.simulate()

all_inputs = np.asarray(all_inputs, dtype=float)  # [N, 5]
print("Collected samples:", all_inputs.shape[0])

# 可选：winsorize 1% 极值，避免尾部拉坏尺度
low, high = np.quantile(all_inputs, [0.01, 0.99], axis=0)
all_inputs_clipped = np.clip(all_inputs, low, high)
print('low:',low)
print('high:', high)

means = all_inputs_clipped.mean(axis=0)
stds  = all_inputs_clipped.std(axis=0, ddof=0)

print("Policy input means:", means)
print("Policy input stds :", stds)
print("Quantiles (1%,25%,50%,75%,99%):\n",
      np.quantile(all_inputs, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0))

