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

label = '020502'
scenario = 'slow transition'
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

# elec_techs = ['coal electricity', 'gas electricity', 'nuclear electricity', 
#               'hydroelectricity', 'biopower electricity', 'wind electricity', 
#               'solar pv electricity', 'electrolyzer'] 
elec_techs = ['coal electricity',
              'gas electricity','nuclear electricity',
              'hydroelectricity','biopower electricity',
              'wind electricity','solar pv electricity','SMR electricity',
              'SMR2 electricity',
              'electrolyzers']

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




# fig, ax = plot_policy_heatmap_feature_domain(
#     policy=model.policy,
#     time_norm=0.375,  #2030 = 0.125; 2040 = 0.25，2050 = 0.375， 2070 = 0.625；  2090 = 0.875; 
#     grid_balance_x10= 1.2,
#     tech_share=0.12,
#     cost_feat_range=(0.5, 2.0),
#     cp10_range=(0.0, 0.35),
#     nbins=100,
#     cmap='Blues',#
#     title='Policy heatmap (feature domain) Year = 2050',
#     savepath=f'results/figures/{label}/heatmap2050_feature_domain.png',
#     show_linear_cost_ticks=False
# )


#### new function, show original unit cost and production


def plot_policy_heatmap_raw_domain_from_feature_ranges(
    policy,
    time_norm,
    grid_balance_x10,
    tech_share,
    cost_feat_range=(0.5, 2.0),   # feature: log10(cost)
    cp10_range=(0.0, 0.35),       # feature: log10(prod)/10
    nbins=120,
    cmap="viridis",
    vmin=None,
    vmax=None,
    title=None,
    savepath=None,
    x_log=True,
    y_log=True,
    show_feature_ticks=False,     # optional: add top/right axes in feature units
):
    """
    Paper-friendly heatmap in ORIGINAL (raw) units for interpretability, while feeding the policy
    the SAME feature transforms used in training.

    Feature definitions (must match your policy input):
      logc  = log10(unit_cost)
      cp10  = log10(cumulative_production)/10

    Axes shown (raw domain):
      x-axis: cumulative production z (raw)
      y-axis: unit cost c (raw)

    The raw-domain ranges are derived EXACTLY from the provided feature ranges:
      c in [10**cost_feat_range[0], 10**cost_feat_range[1]]
      z in [10**(10*cp10_range[0]), 10**(10*cp10_range[1])]
    """

    # --- derive raw ranges from feature ranges (exact inverse mapping) ---
    c_min = 10 ** float(cost_feat_range[0])
    c_max = 10 ** float(cost_feat_range[1])

    z_min = 10 ** (10.0 * float(cp10_range[0]))
    z_max = 10 ** (10.0 * float(cp10_range[1]))

    # --- build grids in raw domain ---
    if y_log:
        y_vals = np.logspace(np.log10(c_min), np.log10(c_max), nbins)
    else:
        y_vals = np.linspace(c_min, c_max, nbins)

    if x_log:
        x_vals = np.logspace(np.log10(z_min), np.log10(z_max), nbins)
    else:
        x_vals = np.linspace(z_min, z_max, nbins)

    heat = np.empty((nbins, nbins), dtype=float)

    # --- evaluate policy on the grid ---
    # IMPORTANT: feed the policy the feature inputs (logc, cp10), not raw values.
    for iy, c in enumerate(y_vals):
        logc = np.log10(c)
        for ix, z in enumerate(x_vals):
            cp10 = np.log10(z) / 10.0
            inp = np.array([logc, cp10, time_norm, grid_balance_x10, tech_share], dtype=float)
            a = policy.get_action(inp)
            heat[iy, ix] = float(np.asarray(a).reshape(-1)[0])

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    im = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        cmap=cmap,
        vmin=(vmin if vmin is not None else np.nanmin(heat)),
        vmax=(vmax if vmax is not None else np.nanmax(heat)),
    )

    ax.set_xlabel("Cumulative production ")
    ax.set_ylabel("Unit cost ")
    ax.set_title(title or f"Policy heatmap | t={time_norm:.2f}, grid={grid_balance_x10:.2f}, share={tech_share:.2f}")

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label("Policy action (growth rate)")

    # Optional: show feature-domain ticks on top/right axes
    if show_feature_ticks:
        # top axis: cp10 (feature) from raw z
        secx = ax.secondary_xaxis(
            "top",
            functions=(lambda z: np.log10(z) / 10.0, lambda cp10: 10 ** (10.0 * cp10)),
        )
        secx.set_xlabel("log10(cumulative production)/10 (feature)")

        # right axis: log10(cost) (feature) from raw c
        secy = ax.secondary_yaxis(
            "right",
            functions=(lambda c: np.log10(c), lambda logc: 10 ** logc),
        )
        secy.set_ylabel("log10(unit cost) (feature)")

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    return fig, ax



# fig, ax = plot_policy_heatmap_raw_domain_from_feature_ranges(
#     policy= model.policy,
#     time_norm=0.375,
#     grid_balance_x10= 1.2,
#     tech_share=0.12,
#     cost_feat_range=(0.5, 2.0),
#     cp10_range=(0.0, 0.35),
#     nbins=120,
#     cmap = 'Blues',
#     title = 'Policy heatmap (original domain) Year = 2050',
#     savepath=f'results/figures/{label}/heatmap2050_original_domain.png',
#     x_log=True,
#     y_log=True,
#     show_feature_ticks=False  # optional
    
# )



from matplotlib.colors import PowerNorm


def plot_policy_heatmap_raw_domain_paper(
    policy,
    time_norm,
    grid_balance_x10,
    tech_share,
    cost_feat_range=(0.5, 2.0),     # feature: log10(cost)
    cp10_range=(0.0, 0.35),         # feature: log10(prod)/10
    nbins=160,
    x_log=True,
    y_log=True,
    cmap="Blues",
    # --- contrast / normalization ---
    use_quantile_vlims=True,
    q_low=0.05,
    q_high=0.95,
    vmin=None,
    vmax=None,
    use_power_norm=True,
    gamma=0.7,                      # <1 emphasizes high end, >1 emphasizes low end
    # --- contours (decision boundaries) ---
    add_contours=True,
    contour_levels=(0.2, 0.5, 1.0),
    contour_color="k",
    contour_lw=1.0,
    contour_alpha=0.9,
    contour_labels=True,
    # --- axes helpers ---
    show_feature_ticks=True,         # top/right feature axes
    # --- labeling / output ---
    title=None,
    savepath=None,
    dpi=300,
    return_data=False,               # optionally return heat + axes grids
):
    """
    Paper-friendly heatmap in ORIGINAL (raw) units with log axes, plus:
      - quantile-based color limits (to avoid 'all blue' saturation),
      - optional PowerNorm (contrast enhancement),
      - optional contour lines (decision boundaries),
      - optional feature-domain secondary axes.

    Policy inputs match your training feature transform:
      logc = log10(unit_cost)
      cp10 = log10(cumulative_production)/10
      inp = [logc, cp10, time_norm, grid_balance_x10, tech_share]

    Raw domain ranges are derived EXACTLY from feature ranges:
      c in [10**cost_feat_range[0], 10**cost_feat_range[1]]
      z in [10**(10*cp10_range[0]), 10**(10*cp10_range[1])]
    """

    # --- inverse map: feature ranges -> raw ranges ---
    c_min = 10 ** float(cost_feat_range[0])
    c_max = 10 ** float(cost_feat_range[1])
    z_min = 10 ** (10.0 * float(cp10_range[0]))
    z_max = 10 ** (10.0 * float(cp10_range[1]))

    # --- grids in raw domain (log-spaced recommended) ---
    y_vals = (np.logspace(np.log10(c_min), np.log10(c_max), nbins) if y_log
              else np.linspace(c_min, c_max, nbins))
    x_vals = (np.logspace(np.log10(z_min), np.log10(z_max), nbins) if x_log
              else np.linspace(z_min, z_max, nbins))

    heat = np.empty((nbins, nbins), dtype=float)

    # --- evaluate policy ---
    for iy, c in enumerate(y_vals):
        logc = np.log10(c)
        for ix, z in enumerate(x_vals):
            cp10 = np.log10(z) / 10.0
            inp = np.array([logc, cp10, time_norm, grid_balance_x10, tech_share], dtype=float)
            a = policy.get_action(inp)
            heat[iy, ix] = float(np.asarray(a).reshape(-1)[0])

    # --- choose vmin/vmax ---
    if use_quantile_vlims and (vmin is None or vmax is None):
        vmin_q = float(np.quantile(heat, q_low))
        vmax_q = float(np.quantile(heat, q_high))
        # avoid degenerate limits
        if np.isclose(vmin_q, vmax_q):
            vmin_q, vmax_q = float(np.nanmin(heat)), float(np.nanmax(heat))
        vmin = vmin if vmin is not None else vmin_q
        vmax = vmax if vmax is not None else vmax_q
    else:
        vmin = vmin if vmin is not None else float(np.nanmin(heat))
        vmax = vmax if vmax is not None else float(np.nanmax(heat))

    # --- normalization (contrast enhancement) ---
    norm = None
    if use_power_norm:
        # PowerNorm requires vmin < vmax; clamp if needed
        if vmax <= vmin:
            vmax = vmin + 1e-9
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    # --- plotting ---
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # Use pcolormesh so contours align perfectly in log axes
    X, Y = np.meshgrid(x_vals, y_vals)
    im = ax.pcolormesh(
        X, Y, heat,
        shading="auto",
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
    )

    ax.set_xlabel("Cumulative production (raw)")
    ax.set_ylabel("Unit cost (raw)")
    ax.set_title(title or f"Policy heatmap | Year(norm)={time_norm:.2f}, grid={grid_balance_x10:.2f}, share={tech_share:.2f}")

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    # --- contours (decision boundaries) ---
    if add_contours:
        # For contour, pass the same X,Y coordinates and heat.
        cs = ax.contour(
            X, Y, heat,
            levels=list(contour_levels),
            colors=contour_color,
            linewidths=contour_lw,
            alpha=contour_alpha,
        )
        if contour_labels:
            ax.clabel(cs, inline=True, fontsize=12, fmt="%.2g")

    # --- colorbar ---
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label("Policy action (growth rate)")
    if use_quantile_vlims:
        # small note in colorbar to avoid confusion
        cbar.ax.set_title(f"clip {int(q_low*100)}–{int(q_high*100)}%", fontsize=8, pad=6)

    # --- feature-domain secondary axes (top/right) ---
    if show_feature_ticks:
        # top: cp10 = log10(z)/10
        secx = ax.secondary_xaxis(
            "top",
            functions=(lambda z: np.log10(z) / 10.0, lambda cp10: 10 ** (10.0 * cp10)),
        )
        secx.set_xlabel("log10(cumulative production)/10 (feature)")

        # right: logc = log10(c)
        secy = ax.secondary_yaxis(
            "right",
            functions=(lambda c: np.log10(c), lambda logc: 10 ** logc),
        )
        secy.set_ylabel("log10(unit cost) (feature)")

    # --- save ---
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if return_data:
        return fig, ax, heat, x_vals, y_vals
    return fig, ax




fig, ax = plot_policy_heatmap_raw_domain_paper(
    policy= model.policy,
    time_norm=0.375,            # e.g., Year 2050 normalized
    grid_balance_x10=1.2,
    tech_share=0.12,
    cost_feat_range=(0.5, 2.0),
    cp10_range=(0.0, 0.35),
    nbins=160,
    cmap="Blues",
    use_quantile_vlims=False,   # key: avoids "all blue"
    q_low=0.05,
    q_high=0.95,
    vmin=-0.1,       # ⭐️强制 color scale
    vmax=1.4,
    use_power_norm=False,
    gamma=0.7,
    add_contours=True,
    contour_levels=(0.2, 0.5, 1.0),
    show_feature_ticks=False,
    title = 'Year = 2050',
    savepath=f'results/figures/{label}/heatmap2050_paper_original_domain.png',
)



