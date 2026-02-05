# simulate a trained policy
# compare with exogenous 


# 0820
# test for lowest cost and highest cost scenario

# 结果：1. boxplot图： 不同 scenario（No Transition, Slow Transition, Fast Transition）和不同策略（exogenous, policy）下的 Net Present Cost 分布（箱线图）
# 2. GIF动画和面积图 ： 在if scneario == 'fast' 更改成slow/no

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

## set to True to run new simulations
simulate = True
# select the number of cost projection simulations
# needed to explore parameters' uncertainty
# used only if new simulations are run

nsim =100
label = '122201'
sim_scenario = 'fast transition'

gt_clip = 1
hidden_size = 2
input_norm = False

savegif = True #individual simulation dynamics 


# create labels for different cost assumptions
labels = ['Way et al. (2022)']

# define colors for technologies
# techcolors = ['black', 'saddlebrown', 'darkgray', 'saddlebrown', 'darkgray',
#               'magenta', 'royalblue', 'forestgreen', 'deepskyblue',
#               'orange', 'pink', 'plum', 'lawngreen', 'burlywood'] 

techcolors = ['black', 'saddlebrown', 'darkgray', 'saddlebrown', 'darkgray',
              'magenta', 'royalblue', 'forestgreen', 'deepskyblue',
              'orange', 'steelblue', 'pink', 'plum', 'lawngreen', 'burlywood']
# resimulate only if required
if simulate:
    np.random.seed(0)

    # create dictionary to store total costs
    tcosts = {}

    # for each label, create an empty dictionary to store costs
    for l in labels:
        tcosts[l] = {}
        tcosts[l+' - decision rule'] = {}

    # create empty list to store technology expansion
    techExp = []

    scenario = sim_scenario

    # create empty list to store total costs
    for l in labels:
        tcosts[l][scenario] = []
        tcosts[l+' - decision rule'][scenario] = []

    # pass input data to model
    print("building the model...", 'scenario')
    model = _energy_sim_model.EnergyModel(\
                EFgp = _energy_sim_params.scenarios[scenario][0],
                slack = _energy_sim_params.scenarios[scenario][1],
                costparams = _energy_sim_params.costsAssumptions['Way et al. (2022)'],
                gt_clip=gt_clip,
                hidden_size=hidden_size,
                input_norm=input_norm)

    ######## simulate model


    # set simulation mode
    model.mode = 'exogenous'
    print('start:',model.mode)


    np.random.seed(0)

    all_costs_exo = [] #total system cost
    all_q_exo = []  # tech generation
    all_c_exo = [] # unit cost time series
    all_omega_exo = [] # learning rate
    
    all_C_exo = []        # 各技术的年度成本
    all_totalCost_exo = [] # 年度总成本
    all_gridInv_exo = []   # 电网投资成本
    all_elec_exo = []      # 电力需求
    
    for n in range(nsim):
        # for each cost assumption, compute total costs
        # and append it to the dictionary
        # 1e-12 is used to convert from USD to trillion USD
        for l in labels:
            #print("simulating...",n)
            cost = 1e-12 * model.simulate()
            tcosts[l][scenario].append(cost)
            all_costs_exo.append(cost)
            all_q_exo.append(copy.deepcopy(model.q))
            all_c_exo.append(copy.deepcopy(model.c))
            all_omega_exo.append(copy.deepcopy(model.omega))  # 记录本次sample的learning rate参数

            all_C_exo.append(copy.deepcopy(model.C))           # 各技术年度成本
            all_totalCost_exo.append(copy.deepcopy(model.totalCost)) # 年度总成本
            all_gridInv_exo.append(copy.deepcopy(model.gridInv))     # 电网成本  
            all_elec_exo.append(copy.deepcopy(model.elec))           # 电力需求



            # tcosts[l][scenario].append( 1e-12 * model.simulate())
            # if scenario == sim_scenario and savegif:
            #     print("saving the figure...")
            #     model.make_gif(f'static_{scenario.replace(" ", "_")}_{n}')
                
            #     model.plotFinalEnergyBySource(label,filename=f'{n}_static_area_{scenario.replace(" ", "_")}_{n}')
            #     model.plotFinalEnergy(label,filename=f'{n}_static_{scenario.replace(" ", "_")}_{n}')
                
            #     model.plotIndividualTechAreas(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
            #     model.plotCapacityExpansion(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
            #     model.plotNewBuildsAndRetirements(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
            
    # # 找最低和最高cost索引

    def print_detailed_cost_breakdown(idx, scenario_name, mode = 'policy'):
        """
            打印详细成本分解
            
            参数:
            - idx: 要分析的索引
            - scenario_name: 场景名称
            - mode: 'policy' 或 'exo'，指定使用哪组数据
        """
        
        print(f"\n=== {scenario_name} Detailed Cost Breakdown ===")
        
        if mode == 'policy':
            all_C = all_C_policy
            all_gridInv = all_gridInv_policy
            all_totalCost = all_totalCost_policy
            all_costs = all_costs_policy
            all_elec = all_elec_policy
            all_q = all_q_policy
        elif mode == 'exo':
            all_C = all_C_exo
            all_gridInv = all_gridInv_exo
            all_totalCost = all_totalCost_exo
            all_costs = all_costs_exo
            all_elec = all_elec_exo
            all_q = all_q_exo
    
        # 各技术的总成本
        print("Technology Costs (trillion USD):")
        tech_costs = {}
        total_tech_cost = 0
        for tech in model.technology[:13]:  # 前13个是主要技术
            tech_total_cost = sum(all_C[idx][tech]) * 1e-12  # 转换为trillion USD
            tech_costs[tech] = tech_total_cost
            total_tech_cost += tech_total_cost
            print(f"  {tech}: {tech_total_cost:.3f}")
        
        # 电网成本
        grid_total_cost = sum(all_gridInv[idx]) * 1e-3  # 从billion转为trillion USD
        print(f"  Grid investment: {grid_total_cost:.3f}")
        
        # 总成本
        system_total_cost = sum(all_totalCost[idx]) * 1e-12
        print(f"\nTotal system cost: {system_total_cost:.3f} trillion USD")
        print(f"NPV (discounted): {all_costs[idx]:.3f} trillion USD")
        
        # 检查是否有电力缺口惩罚
        print(f"\n=== Electricity Balance Check ===")
        penalty_found = False
        overbuild_penalty_found = False

    
        for y in range(model.y0, model.yend+1):
            elec_demand = all_elec[idx][y-model.y0]
            elec_supply = sum([all_q[idx][model.technology[x]][y-model.y0] 
                          for x in model.carrierInputs[model.carrier.index('electricity')]])
            
            deficit = max(0, elec_demand - elec_supply)

            #检查underbuild penalty
            if deficit > 0.1:  # 有明显缺口
                penalty_cost = 10000 * 1/(1000/(60*60)) * 1e9 * deficit * 1e-12
                print(f"Year {y}: Deficit={deficit:.2f} EJ, Penalty={penalty_cost:.3f} trillion USD")
                penalty_found = True
            
                 
            # 检查overbuild penalty
            overbuild_ratio = elec_supply / (elec_demand + 1e-9)
            if overbuild_ratio > 1.3:
                excess_ratio = overbuild_ratio - 1.3
                overbuild_penalty = 500 * 1/(1000/(60*60)) * 1e9 * \
                                (excess_ratio**2) * elec_demand * 1e-12
                print(f"Year {y}: Overbuild ratio={overbuild_ratio:.2f}, Overbuild Penalty={overbuild_penalty:.3f} trillion USD")
                overbuild_penalty_found = True
        
        
        if not penalty_found:
            print("No significant electricity deficit penalties found.")
    
        if not overbuild_penalty_found:
            print("No significant electricity overbuild penalties found.")
            
        return tech_costs

    
    idx_min_exo = np.argmin(all_costs_exo)
    idx_max_exo = np.argmax(all_costs_exo)
    print('exo highest cost:',max(all_costs_exo))

    #print_detailed_cost_breakdown(idx_min_exo, "Exo Lowest Cost")
    print_detailed_cost_breakdown(idx_max_exo, "Exo Highest Cost", mode = 'exo')

   

    # # # 找对应的learning rate参数
    # #omega_min_exo = all_omega_exo[idx_min_exo]
    # #omega_max_exo = all_omega_exo[idx_max_exo]


    
    # # 找85 和15 percentile cost 索引
    # p15 = np.percentile(all_costs_exo, 15)
    # p85 = np.percentile(all_costs_exo, 85)
    # idx_min_exo = np.argmin(np.abs(np.array(all_costs_exo) - p15))
    # idx_max_exo = np.argmin(np.abs(np.array(all_costs_exo) - p85))
    
    
    # # # # 找到最高和最低solar learning rate 的索引
    # omega_solar = [omega['solar pv electricity'] for omega in all_omega_exo]
    # idx_min_exo = np.argmin(omega_solar)
    # idx_max_exo = np.argmax(omega_solar)
    # print('lowest lr exo:', min(omega_solar))
    # print('highest lr exo:', max(omega_solar))
    

    



    # --- Policy mode ---

    # set policy mode
    model.mode = 'policy'
    print('start:',model.mode)

    # load policy file
    #model.policy.load('energySim' + os.path.sep + 'fast_transition_policy_new.pth')
    policy_path = f'results/{label}_{sim_scenario}_policy.pth'
    model.policy.load(policy_path)
    # run multiple iterations to explore cost parameters' uncertainty
    np.random.seed(0)

    all_shares = []

    all_costs_policy = []
    all_q_policy = []
    all_c_policy = []
    all_omega_policy = []

    all_C_policy = []        # 各技术的年度成本
    all_totalCost_policy = [] # 年度总成本
    all_gridInv_policy = []   # 电网投资成本
    all_elec_policy = []      # 电力需求

    for n in range(nsim):
        # for each cost assumption, compute total costs
        # and append it to the dictionary
        # 1e-12 is used to convert from USD to trillion USD
        for l in labels:
            #print("simulating...",n)
            #tcosts[l+' - decision rule'][scenario].append( 1e-12 * model.simulate())
            cost = 1e-12 * model.simulate()
            tcosts[l+' - decision rule'][scenario].append(cost)
            all_costs_policy.append(cost)
            all_q_policy.append(copy.deepcopy(model.q))
            all_c_policy.append(copy.deepcopy(model.c))
            all_omega_policy.append(copy.deepcopy(model.omega))  # 记录本次sample的learning rate参数

            all_C_policy.append(copy.deepcopy(model.C))           # 各技术年度成本
            all_totalCost_policy.append(copy.deepcopy(model.totalCost)) # 年度总成本
            all_gridInv_policy.append(copy.deepcopy(model.gridInv))     # 电网成本  
            all_elec_policy.append(copy.deepcopy(model.elec))           # 电力需求

            
            # if scenario == sim_scenario  and savegif:
            #     print("saving the figure...")
            #     model.make_gif(f'dynamic_{scenario.replace(" ", "_")}_{n}')
            #     model.plotFinalEnergyBySource(label, filename=f'{n}_dynamic_area_{scenario.replace(" ", "_")}_{n}')
            #     model.plotFinalEnergy(label,filename=f'{n}_dynamic_{scenario.replace(" ", "_")}_{n}')
                
            #     model.plotIndividualTechAreas(filename=f'dynamic_area_{scenario.replace(" ", "_")}_{n}')
            #     model.plotCapacityExpansion(filename=f'dynamic_area_{scenario.replace(" ", "_")}_{n}')
            #     model.plotNewBuildsAndRetirements(filename=f'dynamic_area_{scenario.replace(" ", "_")}_{n}')
            #     shares_df = model.get_generation_shares()
            #     #print(shares_df)
            #     all_shares.append(shares_df)
    
    
    
    




    # #找到最高和最低cost 索引
    
    # idx_min_policy = np.argmin(all_costs_policy)
    # idx_max_policy = np.argmax(all_costs_policy)
    # idx_med_policy = np.argmin(np.abs(np.array(all_costs_policy) - np.median(all_costs_policy)))



    # print('policy highest cost:', max(all_costs_policy))
    # # 分析最高和最低成本scenario
    # print_detailed_cost_breakdown(idx_min_policy, "Policy Lowest Cost")
    # print_detailed_cost_breakdown(idx_max_policy, "Policy Highest Cost")

   

  
    # # # # 找到85和15 percentile cost 索引
    # p15 = np.percentile(all_costs_policy, 15)
    # p85 = np.percentile(all_costs_policy, 85)
    # idx_min_policy = np.argmin(np.abs(np.array(all_costs_policy) - p15))
    # idx_max_policy = np.argmin(np.abs(np.array(all_costs_policy) - p85))        
 
    # print('policy 95 highest cost:', p85)
    # # 分析最高和最低成本scenario
    # print_detailed_cost_breakdown(idx_min_policy, "Policy 5 Lowest Cost")
    # print_detailed_cost_breakdown(idx_max_policy, "Policy 95 Highest Cost")



    #找到最高/最低solar learning rate 的索引 （可以替换其他tech）

    # 替换成SMR
    #omega_solar_policy = [omega['solar pv electricity'] for omega in all_omega_policy]
    omega_solar_policy = [omega['SMR electricity'] for omega in all_omega_policy]
    idx_min_policy = np.argmin(omega_solar_policy)
    idx_max_policy = np.argmax(omega_solar_policy)
    print('lowest lr policy:', min(omega_solar_policy))
    print('highest lr policy:', max(omega_solar_policy))





    def plot_tech_comparison(data_list, indices, tech, label, title, name=None, 
                        ylabel='Value', legend_labels=None):
        """
        绘制指定技术在不同索引下的时间序列对比图
        
        参数:
        - data_list: 包含所有仿真结果的列表 (如 all_c_policy 或 all_q_policy)
        - indices: 要对比的索引列表 (如 [idx_min_policy, idx_max_policy])
        - tech: 技术名称 (如 'solar pv electricity')
        - label: 保存文件的标签
        - title: 图表标题
        - name: 保存文件名 (如果为None则显示图表)
        - ylabel: y轴标签 (默认为 'Value')
        - legend_labels: 图例标签列表 (如果为None则使用默认标签)
        """
        years = range(model.y0, model.yend + 1)
        fig = plt.figure(figsize=(10, 6))
        
        # 如果没有提供图例标签，使用默认标签
        if legend_labels is None:
            legend_labels = [f'Index {i}' for i in indices]
        
        # 为每个索引绘制线条
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (idx, legend_label) in enumerate(zip(indices, legend_labels)):
            color = colors[i % len(colors)]
            tech_data = data_list[idx][tech]
            plt.plot(years, tech_data, label=legend_label, color=color, linewidth=2)
        
        plt.xlabel('Year')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if name:
            save_dir = f'results/figures/{label}'
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f'{save_dir}/{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def plot_final_energy_by_source(q_dict, label, title, name=None):
        # colors = ['black','saddlebrown','darkgray',
        #         'saddlebrown','darkgray',
        #         'magenta','royalblue',
        #         'forestgreen','deepskyblue',
        #         'orange','pink','plum','lawngreen', 'burlywood'] 
        colors = ['black','saddlebrown','darkgray',
          'saddlebrown','darkgray',
          'magenta','royalblue',
          'forestgreen','deepskyblue',
          'orange','steelblue','pink','plum','lawngreen','burlywood']

        years = range(model.y0, model.yend + 1)
        df = pd.DataFrame(q_dict, index=years, columns=q_dict.keys())
        cols = df.columns[[not(x) in ['qgrid','qtransport',
                                    'electricity networks',
                                    'electrolyzers'] 
                                    for x in df.columns]]
        df = df[cols]
        fig, ax = plt.subplots(figsize=(8,6))
        df.plot.area(stacked=True, lw=0, ax=ax, color=colors, legend=False)
        ax.set_title(title)
        ax.set_xlim(2018,2095)
        ax.set_ylim(0,1500)
        ax.set_ylabel('EJ')
        ax.set_xlabel('Year')
        handles, labels_ = ax.get_legend_handles_labels()
        fig.legend(handles, labels_, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.15))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        if name:
            save_dir = f'results/figures/{label}'
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f'{save_dir}/{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

    def plot_solar_cost(c_dict, label, title, name=None):
        years = range(model.y0, model.yend + 1)
        fig = plt.figure(figsize=(8,5))
        plt.plot(years, c_dict['solar pv electricity'], label='Solar PV Cost')
        plt.xlabel('Year')
        plt.ylabel('Unit Solar PV Cost ')
        plt.title(title)
        plt.legend()
        if name:
            save_dir = f'results/figures/{label}'
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f'{save_dir}/{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def plot_all_tech_costs(c_dict, label, title, name=None):
        years = range(model.y0, model.yend + 1)
        techs = list(c_dict.keys())
        n_techs = len(techs)
        ncols = 4
        nrows = int(np.ceil(n_techs / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True)
        axes = axes.flatten()
        for i, tech in enumerate(techs):
            axes[i].plot(years, c_dict[tech], label=tech)
            axes[i].set_title(tech)
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel('Unit Cost')
            axes[i].legend()
        # 去掉多余的空白子图
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if name:
            save_dir = f'results/figures/{label}'
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f'{save_dir}/{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def plot_q_vs_time(q_dict, label, title, name=None):
        years = range(model.y0, model.yend + 1) # 根据你的模型年份范围调整
        techs = [k for k in q_dict.keys() if isinstance(q_dict[k], (list, np.ndarray))]
        n_techs = len(techs)
        ncols = 4
        nrows = int(np.ceil(n_techs / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True)
        axes = axes.flatten()
        for i, tech in enumerate(techs):
            axes[i].plot(years, q_dict[tech], label=tech)
            axes[i].set_title(tech)
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel('q')
            axes[i].legend()
        # 去掉多余的空白子图
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(title)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if name:
            save_dir = f'results/figures/{label}'
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f'{save_dir}/{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    

    def plot_tech_comparison(data_list, indices, tech, label, title, name=None, 
                        ylabel='Value', legend_labels=None):
        """
        绘制指定技术在不同索引下的时间序列对比图
        
        参数:
        - data_list: 包含所有仿真结果的列表 (如 all_c_policy 或 all_q_policy)
        - indices: 要对比的索引列表 (如 [idx_min_policy, idx_max_policy])
        - tech: 技术名称 (如 'solar pv electricity')
        - label: 保存文件的标签
        - title: 图表标题
        - name: 保存文件名 (如果为None则显示图表)
        - ylabel: y轴标签 (默认为 'Value')
        - legend_labels: 图例标签列表 (如果为None则使用默认标签)
        """
        years = range(model.y0, model.yend + 1)
        fig = plt.figure(figsize=(10, 6))
        
        # 如果没有提供图例标签，使用默认标签
        if legend_labels is None:
            legend_labels = [f'Index {i}' for i in indices]
        
        # 为每个索引绘制线条
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, (idx, legend_label) in enumerate(zip(indices, legend_labels)):
            color = colors[i % len(colors)]
            tech_data = data_list[idx][tech]
            plt.plot(years, tech_data, label=legend_label, color=color, linewidth=2)
        
        plt.xlabel('Year')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if name:
            save_dir = f'results/figures/{label}'
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f'{save_dir}/{name}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    
    
        
    # # --- 画最低和最高 cost 的轨迹并保存 ---

    # #### ------ Exogenous

  
    # plot_final_energy_by_source(all_q_exo[idx_min_exo], label, "Exogenous: Final Energy (Lowest Cost)", name="exo_lowest_final_energy")
    # plot_all_tech_costs(all_c_exo[idx_min_exo], label, "Exogenous: All Tech Costs (Lowest Cost)", name="exo_lowest_all_tech_costs")
    # plot_q_vs_time(all_q_exo[idx_min_exo], label, "Exogenous: q vs Time (Lowest Cost)", name="exo_lowest_q_vs_time")

    # plot_final_energy_by_source(all_q_exo[idx_max_exo], label, "Exogenous: Final Energy (Highest Cost)", name="exo_highest_final_energy")
    # plot_all_tech_costs(all_c_exo[idx_max_exo], label, "Exogenous: All Tech Costs (Highest Cost)", name="exo_highest_all_tech_costs")
    # plot_q_vs_time(all_q_exo[idx_max_exo], label, "Exogenous: q vs Time (Highest Cost)", name="exo_Highest_q_vs_time")


    # # #### ------ Policy

    
    # plot_final_energy_by_source(all_q_policy[idx_min_policy], label, "Policy: Final Energy (Lowest Cost)", name="policy_lowest_final_energy")
    # plot_all_tech_costs(all_c_policy[idx_min_policy], label, "Policy: All Tech Costs (Lowest Cost)", name="policy_lowest_all_tech_costs")
    # plot_q_vs_time(all_q_policy[idx_min_policy], label, "Policy: q vs Time (Lowest Cost)", name="policy_lowest_q_vs_time")

    # plot_final_energy_by_source(all_q_policy[idx_max_policy], label, "Policy: Final Energy (Highest Cost)", name="policy_highest_final_energy")
    # plot_all_tech_costs(all_c_policy[idx_max_policy], label, "Policy: All Tech Costs (Highest Cost)", name="policy_highest_all_tech_costs")
    # plot_q_vs_time(all_q_policy[idx_max_policy], label, "Policy: q vs Time (Highest Cost)", name="policy_highest_q_vs_time")
    

    # # --- 画指定技术在不同索引下的时间序列对比图 ---

    # ## highest, median, lowest 
    # plot_tech_comparison(
    #     data_list = all_c_policy,
    #     indices=[idx_min_policy, idx_med_policy, idx_max_policy],
    #     tech='multi-day storage',
    #     label=label,
    #     title='Policy: multi-day storage Cost Comparison',
    #     name='policy_multi-day_storage_cost_comparison',
    #     ylabel='Unit Cost (USD/GJ)',
    #     legend_labels=['Lowest Total Cost', 'Median Total Cost', 'Highest Total Cost']
    # )

    # techs_to_plot = ['solar pv electricity', 'electrolyzers', 'wind electricity', 'multi-day storage']
 
    # for tech in techs_to_plot:
    #     # 生成适合的title和name
    #     tech_display = tech.replace('electricity', 'elec').replace(' ', '_')
    #     title = f'Policy: {tech.title()} Capacity Comparison'
    #     name = f'policy_{tech_display}_Capacity_comparison'
        
    #     plot_tech_comparison(
    #         data_list=all_q_policy,
    #         indices=[idx_min_policy, idx_max_policy], #idx_med_policy, 
    #         tech=tech,
    #         label=label,
    #         title=title,
    #         name=name,
    #         ylabel= 'Capacity (EJ)',  #'Unit Cost (USD/GJ)',
    #         legend_labels=['Lowest Solar Learning Rate', 'Highest Solar Learning Rate']
    #     )


    
    
    # --- 画最低和最高 (solar) learning rate 的轨迹并保存 ---

    #### ------ Exogenous

    # plot_final_energy_by_source(all_q_exo[idx_min_exo], label, "Exogenous: Final Energy (Lowest lr)", name="exo_lowest_lr_final_energy")
    # plot_all_tech_costs(all_c_exo[idx_min_exo], label, "Exogenous: All Tech Costs (Lowest lr)", name="exo_lowest_lr_all_tech_costs")
    # plot_q_vs_time(all_q_exo[idx_min_exo], label, "Exogenous: q vs Time (Lowest lr)", name="exo_lowest_lr_q_vs_time")

    # plot_final_energy_by_source(all_q_exo[idx_max_exo], label, "Exogenous: Final Energy (Highest lr)", name="exo_highest_lr_final_energy")
    # plot_all_tech_costs(all_c_exo[idx_max_exo], label, "Exogenous: All Tech Costs (Highest lr)", name="exo_highest_lr_all_tech_costs")
    # plot_q_vs_time(all_q_exo[idx_max_exo], label, "Exogenous: q vs Time (Highest lr)", name="exo_Highest_lr_q_vs_time")


    # #### ------ Policy

    # plot_final_energy_by_source(all_q_policy[idx_min_policy], label, "Policy: Final Energy (Lowest lr)", name="policy_lowest_lr_final_energy")
    # plot_all_tech_costs(all_c_policy[idx_min_policy], label, "Policy: All Tech Costs (Lowest lr)", name="policy_lowest_lr_all_tech_costs")
    # plot_q_vs_time(all_q_policy[idx_min_policy], label, "Policy: q vs Time (Lowest lr)", name="policy_lowest_lr_q_vs_time")

    # plot_final_energy_by_source(all_q_policy[idx_max_policy], label, "Policy: Final Energy (Highest lr)", name="policy_highest_lr_final_energy")
    # plot_all_tech_costs(all_c_policy[idx_max_policy], label, "Policy: All Tech Costs (Highest lr)", name="policy_highest_lr_all_tech_costs")
    # plot_q_vs_time(all_q_policy[idx_max_policy], label, "Policy: q vs Time (Highest lr)", name="policy_highest_lr_q_vs_time")
  






# === New function: plot_policy_tj_cost_top10 ===
# 按两种基准分别取“第10高”学习率场景，并各自输出两类图，共6个独立文件：
# 1) 轨迹（final energy by source）：
#    - solar 第10高 LR 的轨迹 -> policy_top10_solar_final_energy.png
#    - smr 第10高 LR 的轨迹   -> policy_top10_smr_final_energy.png
#    - smr2 第10高 LR 的轨迹   -> policy_top10_smr_final_energy.png
# 2) 成本演化（Solar & SMR 两条线）：
#    - solar 第10高 LR 的成本演化 -> policy_top10_solar_cost_evolution.png
#    - smr 第10高 LR 的成本演化   -> policy_top10_smr_cost_evolution.png
#    - smr2 第10高 LR 的成本演化   -> policy_top10_smr_cost_evolution.png

def plot_policy_tj_cost_top10(tech='SMR electricity', rank=10):
    """
    为 policy 结果基于指定技术的学习率，选取第 rank 高的样本索引，
    针对所选技术生成两张图：
      - Final energy by source（轨迹，2020-2070，无图例，标题：DPS:“所选的tech”dominant）
      - 成本演化（2020-2070，Solar + SMR + SMR2 三条线）
    """
    if len(all_omega_policy) == 0:
        print('No policy results available.')
        return

    years = range(model.y0, model.yend + 1)
    save_dir = f'results/figures/{label}/smr_top10'
    os.makedirs(save_dir, exist_ok=True)

    tech_input = tech.strip().lower()
    tech_solar_key = 'solar pv electricity'
    tech_smr_key = 'SMR electricity'
    tech_smr2_key = 'SMR2 electricity'

    # 允许三种基准：solar / SMR / SMR2
    if tech_input in ['solar', 'solar pv electricity','solar electricity']:
        base_tech_key = tech_solar_key
        base_tag = 'Solar'
    elif tech_input in ['smr', 'smr electricity', 'small modular reactor', 'small modular reactor electricity']:
        base_tech_key = tech_smr_key
        base_tag = 'SMR'
    elif tech_input in ['smr2', 'smr2 electricity', 'small modular reactor2', 'small modular reactor 2']:
        base_tech_key = tech_smr2_key
        base_tag = 'SMR2'
    else:
        print(f'Unknown tech: {tech}. Use "solar", "SMR" or "SMR2".')
        return

    # 按所选技术的 omega 排序，取第 rank 高
    lr_base = np.array([omega.get(base_tech_key, np.nan) for omega in all_omega_policy])
    valid_idx = np.where(~np.isnan(lr_base))[0]
    if len(valid_idx) < rank:
        print(f'Not enough valid policy samples for {base_tag} rank={rank}.')
        return
    sorted_desc = valid_idx[np.argsort(lr_base[valid_idx])[::-1]]
    idx_base_topN = int(sorted_desc[rank-1])

    # === 图 1：Final energy by source（2020-2070，无 legend）===
    q_dict = all_q_policy[idx_base_topN]
    df = pd.DataFrame(q_dict, index=years, columns=q_dict.keys())
    cols = df.columns[[not(x) in ['qgrid','qtransport','electricity networks','electrolyzers'] for x in df.columns]]
    df = df[cols]
    # colors = [
    #     'black','saddlebrown','darkgray',
    #     'saddlebrown','darkgray',
    #     'magenta','royalblue',
    #     'forestgreen','deepskyblue',
    #     'orange','steelblue',   # solar, SMR
    #     'purple',               # SMR2
    #     'pink',
    #     'plum','lawngreen','burlywood'
    # ]
    colors = [
        'black','saddlebrown','darkgray',
        'saddlebrown','darkgray',
        'magenta','royalblue',
        'forestgreen','deepskyblue',
        'orange','steelblue',   # solar, SMR
        'pink',
        'plum','lawngreen','burlywood'
    ]


    fig1, ax1 = plt.subplots(figsize=(12,6))
    df.plot.area(stacked=True, lw=0, ax=ax1, color=colors, legend=False)
    ax1.set_title(f'DPS: {base_tag} dominant', fontsize=28, weight='bold')
    ax1.set_xlim(2020, 2070)
    ax1.set_ylim(0, 1500)
    ax1.set_ylabel('Generation(EJ)', fontsize=24, weight='bold')
    ax1.set_xlabel('Year', fontsize=24, weight='bold')
    ax1.tick_params(axis='x', labelsize=22)
    ax1.tick_params(axis='y', labelsize=22)
    plt.tight_layout()
    fig1.savefig(f'{save_dir}/policy_top{rank}_{base_tag.lower()}_final_energy.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # === 图 2：成本演化（Solar + SMR + SMR2 三条线）===
    c_dict = all_c_policy[idx_base_topN]
    fig2, ax2 = plt.subplots(figsize=(12,6))
    years_list = list(years)

    # Solar 从 2020 开始
    solar_cost = np.array(c_dict.get(tech_solar_key, [np.nan]*len(years_list)))
    # SMR 从 2030 开始
    smr_cost = np.array(c_dict.get(tech_smr_key, [np.nan]*len(years_list)))
    smr_mask = np.array(years_list) < 2030
    smr_cost[smr_mask] = np.nan
    # SMR2 从 2030 开始
    smr2_cost = np.array(c_dict.get(tech_smr2_key, [np.nan]*len(years_list)))
    smr2_mask = np.array(years_list) < 2030
    smr2_cost[smr2_mask] = np.nan

    ax2.plot(years_list, solar_cost, label='Solar PV', color='tab:orange', linewidth=4)
    ax2.plot(years_list, smr_cost,   label='SMR (≥2030)',  color='tab:blue',   linewidth=4)
    ax2.plot(years_list, smr2_cost,  label='SMR2 (≥2030)', color='purple',     linewidth=4)

    ax2.set_title('Cost Evolution Over Time', fontsize=28, weight='bold')
    ax2.set_xlabel('Year', fontsize=24, weight='bold')
    ax2.set_ylabel('Unit Cost (USD/GJ)', fontsize=24, weight='bold')
    ax2.set_xlim(2020, 2070)
    ax2.tick_params(axis='x', labelsize=22)
    ax2.tick_params(axis='y', labelsize=22)
    ax2.legend(loc='best', fontsize=22, frameon=False)

    plt.tight_layout()
    fig2.savefig(f'{save_dir}/policy_top{rank}_{base_tag.lower()}_cost_evolution.png',
                 dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f'{base_tag} {rank}th highest LR index: {idx_base_topN}, LR={lr_base[idx_base_topN]:.6f}')


# # 调用示例：现在函数只接受1个技术参数+rank，并仅输出该技术的两张图
try:
    for rr in range(1,50):
        plot_policy_tj_cost_top10('smr electricity', rank=rr) # solar rank5; smr: rank 5， smr5:rank5
except Exception as e:
    print('plot_policy_tj_cost_top10 failed:', e)




########
# new plot function: make smrs area plot

def plot_smrs_cost_top10(tech='SMR electricity', rank=10):
    """
    基于指定技术的学习率，选取第 rank 高的样本，
    画一张图：
      - 左轴：Solar / SMR / SMR2 的 unit cost 曲线
      - 右轴：SMR vs SMR2 在自身总和中的 generation share（面积图）
    """
    if len(all_omega_policy) == 0:
        print('No policy results available.')
        return

    years = range(model.y0, model.yend + 1)
    years_list = list(years)
    save_dir = f'results/figures/{label}/smrs'
    os.makedirs(save_dir, exist_ok=True)

    tech_input = tech.strip().lower()
    tech_solar_key = 'solar pv electricity'
    tech_smr_key = 'SMR electricity'
    tech_smr2_key = 'SMR2 electricity'

    # 允许三种基准：solar / SMR / SMR2
    if tech_input in ['solar', 'solar pv electricity', 'solar electricity']:
        base_tech_key = tech_solar_key
        base_tag = 'Solar'
    elif tech_input in ['smr', 'smr electricity', 'small modular reactor', 'small modular reactor electricity']:
        base_tech_key = tech_smr_key
        base_tag = 'SMR'
    elif tech_input in ['smr2', 'smr2 electricity', 'small modular reactor2', 'small modular reactor 2']:
        base_tech_key = tech_smr2_key
        base_tag = 'SMR2'
    else:
        print(f'Unknown tech: {tech}. Use "solar", "SMR" or "SMR2".')
        return

    # 按所选技术的 omega 排序，取第 rank 高
    lr_base = np.array([omega.get(base_tech_key, np.nan) for omega in all_omega_policy])
    valid_idx = np.where(~np.isnan(lr_base))[0]
    if len(valid_idx) < rank:
        print(f'Not enough valid policy samples for {base_tag} rank={rank}.')
        return
    sorted_desc = valid_idx[np.argsort(lr_base[valid_idx])[::-1]]
    idx_base_topN = int(sorted_desc[rank-1])

    # 取该样本的成本和发电量
    c_dict = all_c_policy[idx_base_topN]
    q_dict = all_q_policy[idx_base_topN]

    # === 左轴：成本演化（三条线）===
    fig, ax_cost = plt.subplots(figsize=(12, 6))

    # Solar 从 2020 开始
    solar_cost = np.array(c_dict.get(tech_solar_key, [np.nan] * len(years_list)))
    # SMR 从 2030 开始
    smr_cost = np.array(c_dict.get(tech_smr_key, [np.nan] * len(years_list)))
    smr_mask = np.array(years_list) < 2030
    smr_cost[smr_mask] = np.nan
    # SMR2 从 2030 开始
    smr2_cost = np.array(c_dict.get(tech_smr2_key, [np.nan] * len(years_list)))
    smr2_mask = np.array(years_list) < 2030
    smr2_cost[smr2_mask] = np.nan

    ax_cost.plot(years_list, solar_cost, label='Solar PV',  color='tab:orange', linewidth=4)
    ax_cost.plot(years_list, smr_cost,   label='SMR (≥2030)',  color='tab:blue',   linewidth=4)
    ax_cost.plot(years_list, smr2_cost,  label='SMR2 (≥2030)', color='purple',     linewidth=4)

    ax_cost.set_title('Cost Evolution with SMR/SMR2 Share Background',
                      fontsize=28, weight='bold')
    ax_cost.set_xlabel('Year', fontsize=24, weight='bold')
    ax_cost.set_ylabel('Unit Cost (USD/GJ)', fontsize=24, weight='bold')
    ax_cost.set_xlim(2020, 2070)
    ax_cost.tick_params(axis='x', labelsize=22)
    ax_cost.tick_params(axis='y', labelsize=22)

    # === 右轴：SMR vs SMR2 generation share（相对自身之和）===
    # q_dict 中存的是各技术的年度发电量（EJ）
    smr_gen = np.array(q_dict.get(tech_smr_key, [0.0] * len(years_list)))
    smr2_gen = np.array(q_dict.get(tech_smr2_key, [0.0] * len(years_list)))
    total_gen = smr_gen + smr2_gen

    # 避免 0 除：total_gen==0 的年份，不画 share（设为 NaN）
    share_smr = np.zeros_like(total_gen, dtype=float)
    share_smr2 = np.zeros_like(total_gen, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        share_smr = np.where(total_gen > 0, smr_gen / total_gen, np.nan)
        share_smr2 = np.where(total_gen > 0, smr2_gen / total_gen, np.nan)

    # 2030 年前不画（两者都几乎为 0）
    mask_share = np.array(years_list) >= 2030
    years_share = np.array(years_list)[mask_share]
    share_smr_plot = share_smr[mask_share]
    share_smr2_plot = share_smr2[mask_share]

    ax_share = ax_cost.twinx()
    ax_share.set_ylim(0, 1.0)
    ax_share.set_ylabel('SMR vs SMR2 Share', fontsize=20, weight='bold')
    ax_share.tick_params(axis='y', labelsize=20)

    # 背景面积图：SMR（蓝，稍透明） + SMR2（紫，稍透明）
    ax_share.stackplot(
        years_share,
        share_smr_plot,
        share_smr2_plot,
        labels=['SMR share', 'SMR2 share'],
        colors=['tab:blue', 'purple'],
        alpha=0.18,  # 透明度，避免盖住成本线
    )

    # 合并图例：左轴成本线 + 右轴 share
    # 先取两个轴的 handle + label
    handles_cost, labels_cost = ax_cost.get_legend_handles_labels()
    handles_share, labels_share = ax_share.get_legend_handles_labels()
    # 合在一起去重
    handles = handles_cost + handles_share
    labels_all = labels_cost + labels_share
    # 简单去重保持顺序
    seen = set()
    handles_unique = []
    labels_unique = []
    for h, lab in zip(handles, labels_all):
        if lab not in seen:
            seen.add(lab)
            handles_unique.append(h)
            labels_unique.append(lab)
    ax_cost.legend(handles_unique, labels_unique,
                   loc='best', fontsize=18, frameon=False)

    plt.tight_layout()
    fig.savefig(f'{save_dir}/smrs_cost_share_top{rank}_{base_tag.lower()}_idx{idx_base_topN}.png',
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f'[plot_smrs_cost_top10] {base_tag} rank={rank}, idx={idx_base_topN}, LR={lr_base[idx_base_topN]:.6f}')    

# try:
#     # 比如画 SMR 学习率第 5 高的那条路径的成本 + SMR/SMR2 share 背景
#     plot_smrs_cost_top10('SMR electricity', rank=3)
# except Exception as e:
#     print('plot_smrs_cost_top10 failed:', e)  