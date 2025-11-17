
import os
import json
import numpy as np
import energySim._energy_sim_model as _energy_sim_model
import energySim._energy_sim_params as _energy_sim_params

label = '092901'
scenario = 'fast transition'
policy_path = f'results/{label}_{scenario}_policy.pth'
nsim = 100
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
electrolyzer_inputs = []
call_count = 0

elec_techs = ['coal electricity', 'gas electricity', 'nuclear electricity', 
              'hydroelectricity', 'biopower electricity', 'wind electricity', 
              'solar pv electricity', 'electrolyzer'] 

orig_get_action = model.policy.get_action

def logging_get_action(pol_input):
    global call_count
    arr = np.asarray(pol_input, dtype=float)
    if np.all(np.isfinite(arr)) and arr.shape == (5,):

        tech_idx = call_count % 8

        if tech_idx == 7: # elecrrolzyer 索引
            year = model.y0 + (call_count // 8)

            gt_raw = orig_get_action(pol_input)

            # 提取标量值
            if isinstance(gt_raw, np.ndarray):
                gt = gt_raw.item()  # 将(1,)数组转换为标量
            else:
                gt = float(gt_raw)

       
            if year == 2090: 
                # 获取当前electrolyzer容量
                current_electrolyzer = model.q['electrolyzers'][model.y-model.y0]

                #solar_inputs.append(arr.tolist())
                print(f"year {year}: inputs={arr}")
                print(f"  Tech cost: {arr[0]:.3f}")
                # print(f"  Cum prod/10: {arr[1]:.3f}")
                # print(f"  Time progress: {arr[2]:.3f}")
                # print(f"  Supply-demand: {arr[3]:.3f}")
                # print(f"  Tech share: {arr[4]:.3f}")
                print(f"  Growth rate output: {gt:.3f}")
   
                
                 # 计算下一年的electrolyzer容量
                gt = min(1.0, gt)
                next_electrolyzer = current_electrolyzer * (1 + gt)
                print(f"  Next year electrolyzer capacity: {next_electrolyzer:.6f}")

        call_count +=1

        all_inputs.append(arr.tolist())
    return orig_get_action(pol_input)

model.policy.get_action = logging_get_action

np.random.seed(0)
for n in range(nsim):
    print(f"\n=== Simulation {n+1} ===")
    call_count = 0 
    _ = model.simulate()

all_inputs = np.asarray(all_inputs, dtype=float)  # [N, 5]
#solar_inputs = np.asarray(solar_inputs, dtype=float)

#print("Collected samples:", all_inputs.shape)
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

