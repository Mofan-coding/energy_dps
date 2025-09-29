
import os
import json
import numpy as np
import energySim._energy_sim_model as _energy_sim_model
import energySim._energy_sim_params as _energy_sim_params

label = '092801'
scenario = 'fast transition'
policy_path = f'results/{label}_{scenario}_policy.pth'
nsim = 1
gt_clip = 1
hidden_size = 2
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
solar_inputs = []
call_count = 0

elec_techs = ['coal electricity', 'gas electricity', 'nuclear electricity', 
              'hydroelectricity', 'biopower electricity', 'wind electricity', 
              'solar pv electricity'] 

orig_get_action = model.policy.get_action

def logging_get_action(pol_input):
    global call_count
    arr = np.asarray(pol_input, dtype=float)
    if np.all(np.isfinite(arr)) and arr.shape == (5,):

        tech_idx = call_count % 7

        if tech_idx == 6: # solar 是第7个技术，索引为6
            year = model.y0 + (call_count // 7)

            gt = orig_get_action(pol_input)

            gt_final = gt
            current_share = model.q['solar pv electricity'][year-model.y0] / model.elec[year-model.y0]
        
            if current_share > 1.0:

                gt_final = min(gt, 0.0001)
                print(gt_final)
            

            solar_inputs.append(arr.tolist())
            print(f"Solar year {year}: inputs={arr}")
            print(f"  Tech cost: {arr[0]:.3f}")
            print(f"  Cum prod/10: {arr[1]:.3f}")
            print(f"  Time progress: {arr[2]:.3f}")
            print(f"  Supply-demand: {arr[3]:.3f}")
            print(f"  Tech share: {arr[4]:.3f}")
            
            # 修复：处理 gt 可能是数组的情况
            if isinstance(gt, np.ndarray):
                if gt.size == 1:
                    print(f"  Growth rate output: {gt.item():.3f}")
                else:
                    print(f"  Growth rate output: {gt}")
            else:
                print(f"  Growth rate output: {gt_final:.3f}")

        call_count +=1

        all_inputs.append(arr.tolist())
    return orig_get_action(pol_input)

model.policy.get_action = logging_get_action

np.random.seed(0)
for n in range(nsim):
    print(f"\n=== Simulation {n+1} ===")
    _ = model.simulate()

all_inputs = np.asarray(all_inputs, dtype=float)  # [N, 5]
solar_inputs = np.asarray(solar_inputs, dtype=float)

print("Collected samples:", all_inputs.shape)
#rint(all_inputs)

# # 可选：winsorize 1% 极值，避免尾部拉坏尺度
# low, high = np.quantile(all_inputs, [0.01, 0.99], axis=0)
# all_inputs_clipped = np.clip(all_inputs, low, high)
# print('low:',low)
# print('high:', high)

# means = all_inputs_clipped.mean(axis=0)
# stds  = all_inputs_clipped.std(axis=0, ddof=0)

# print("Policy input means:", means)
# print("Policy input stds :", stds)
# print("Quantiles (1%,25%,50%,75%,99%):\n",
#       np.quantile(all_inputs, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0))

