# for a given policy, plot the heatmap
# y axis: cost; x axis: cumulative production



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib, os
import energySim._energy_sim_model as _energy_sim_model
import energySim._energy_sim_params as _energy_sim_params
import numpy as np
import copy

matplotlib.rc('savefig', dpi=300)
sns.set_style('ticks')
sns.set_context('talk')
matplotlib.rc('font',
                **{'family':'sans-serif',
                   'sans-serif':'Helvetica'})


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
    # get original action
    action = orig_get_action(pol_input)

    # record inputs if valid 5-dim finite vector
    if arr.shape == (5,) and np.all(np.isfinite(arr)):
        all_inputs.append(arr.tolist())

    # advance call counter
    call_count += 1

    return action


model.policy.get_action = logging_get_action

np.random.seed(0)
for n in range(nsim):
    print(f"\n=== Simulation {n+1} ===")
    call_count = 0 
    _ = model.simulate()

all_inputs = np.asarray(all_inputs, dtype=float)  # [N, 5]


def extract_year_across_sims(all_inputs, target_year, y0, nsim, ntech=8):
    N = all_inputs.shape[0]
    assert N % (nsim * ntech) == 0, "all_inputs length not divisible by nsim*ntech"
    n_years = N // (nsim * ntech)
    arr = all_inputs.reshape(nsim, n_years, ntech, 5)
    years = np.arange(y0, y0 + n_years)
    yi = int(np.where(years == target_year)[0][0])
    return arr[:, yi, :, :]  # (nsim, ntech, 5)

def summarize_year_features(inputs_year, winsor=(0.01, 0.99)):
    # inputs_year: (nsim, ntech, 5)
    X = inputs_year.reshape(-1, inputs_year.shape[-1])  # (nsim*ntech, 5)
    if winsor is not None:
        qlo, qhi = np.quantile(X, winsor, axis=0)
        X = np.clip(X, qlo, qhi)
    means = X.mean(axis=0)
    stds  = X.std(axis=0, ddof=0)
    qs = np.quantile(X, [0.01, 0.25, 0.5, 0.75, 0.99], axis=0)  # shape (5q, 5feat)
    cols = ['mean', 'std', 'q01', 'q25', 'q50', 'q75', 'q99']
    data = np.vstack([means, stds, qs]).T
    import pandas as pd
    feat_names = ['log10_cost','log10_cumprod_div10','time_norm','grid_balance_x10','tech_share']
    return pd.DataFrame(data, index=feat_names, columns=cols)

# # Examples: 2030 and 2090
# inputs_2030_all = extract_year_across_sims(all_inputs, 2030, y0=model.y0, nsim=nsim)
# inputs_2050_all = extract_year_across_sims(all_inputs, 2050, y0=model.y0, nsim=nsim)
# inputs_2090_all = extract_year_across_sims(all_inputs, 2090, y0=model.y0, nsim=nsim)

# stats_2030 = summarize_year_features(inputs_2030_all, winsor=(0.01, 0.99))
# stats_2050 = summarize_year_features(inputs_2050_all, winsor=(0.01, 0.99))
# stats_2090 = summarize_year_features(inputs_2090_all, winsor=(0.01, 0.99))

# print("2030 feature stats across all sims and techs:")
# print(stats_2030.round(3))
# print("\n2090 feature stats across all sims and techs:")
# print(stats_2090.round(3))
# print("\n2050 feature stats across all sims and techs:")
# print(stats_2050.round(3))

############### Make Policy Heatmap

def plot_policy_heatmap_feature_domain(
    policy,
    time_norm,
    grid_balance_x10,
    tech_share,
    cost_feat_range=(0.5, 2.0),   # y轴：已在特征域（不再取log）
    cp10_range=(0.0, 0.3),        # x轴：log10(cumprod)/10（特征域）
    nbins=100,
    cmap='viridis',
    vmin=None,
    vmax=None,
    title=None,
    savepath=None,
    show_linear_cost_ticks=False  # 可选：右侧显示线性cost刻度(10**y)
):
    """
    在特征域上画policy热图：
      y轴 = log10_cost(特征) ∈ cost_feat_range
      x轴 = log10_cumprod_div10(特征) ∈ cp10_range
      固定其它输入：time_norm, grid_balance_x10, tech_share
      颜色 = policy.get_action([log10_cost, cp10, time_norm, grid_balance_x10, tech_share])
    """
    y_vals = np.linspace(cost_feat_range[0], cost_feat_range[1], nbins)  # 特征域：log10_cost
    x_vals = np.linspace(cp10_range[0], cp10_range[1], nbins)            # 特征域：log10_cumprod/10
    heat = np.empty((nbins, nbins), dtype=float)

    for iy, logc in enumerate(y_vals):
        for ix, cp10 in enumerate(x_vals):
            inp = np.array([logc, cp10, time_norm, grid_balance_x10, tech_share], dtype=float)
            a = policy.get_action(inp)
            heat[iy, ix] = float(np.asarray(a).reshape(-1)[0])

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    im = ax.imshow(
        heat,
        origin='lower',
        aspect='auto',
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        cmap=cmap,
        vmin=(vmin if vmin is not None else np.nanmin(heat)),
        vmax=(vmax if vmax is not None else np.nanmax(heat)),
    )
    ax.set_xlabel('log10(cumulative production) (feature)')
    ax.set_ylabel('log10(unit cost) (feature)')
    ax.set_title(title or f'Policy heatmap | t={time_norm:.2f}, grid={grid_balance_x10:.2f}, share={tech_share:.2f}')
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label('Policy action (growth rate)')

    if show_linear_cost_ticks:
        sec = ax.secondary_yaxis('right', functions=(lambda y: 10**y, lambda yl: np.log10(yl)))
        sec.set_ylabel('Unit cost (linear)')

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    return fig, ax

model.policy.get_action = orig_get_action
fig, ax = plot_policy_heatmap_feature_domain(
    policy=model.policy,
    time_norm=0.875,  #2030 = 0.125; 2090 = 0.875; 2050 = 0.375
    grid_balance_x10= 1.7,
    tech_share=0.12,
    cost_feat_range=(0.5, 2.0),
    cp10_range=(0.0, 0.35),
    nbins=100,
    cmap='Blues',#
    title='Policy heatmap (feature domain) Year = 2090',
    savepath=f'results/figures/{label}/heatmap2090_feature_domain.png',
    show_linear_cost_ticks=False
)