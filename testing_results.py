# simulate a trained policy
# compare with exogenous 

# 结果：1. boxplot图： 不同 scenario（No Transition, Slow Transition, Fast Transition）和不同策略（exogenous, policy）下的 Net Present Cost 分布（箱线图）
# 2. GIF动画和面积图 ： 在if scneario == 'fast' 更改成slow/no

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib, os
import energySim._energy_sim_model as _energy_sim_model
import energySim._energy_sim_params as _energy_sim_params
import numpy as np

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
label = '011301'
sim_scenario = 'fast transition'

gt_clip = 1
hidden_size = 2
input_norm = False


savegif = True #individual simulation dynamics 
savebox= False # boxplot of costs 
save_sharebox = False #Boxplot of End-of-Century Generation Share
save_pc = False  #Parallel Coordinates Plot
new_sharebox = True # boxplot of selected techs


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
    #np.random.seed(1)

    # create dictionary to store total costs
    tcosts = {}

    # for each label, create an empty dictionary to store costs
    for l in labels:
        tcosts[l] = {}
        tcosts[l+' - decision rule'] = {}

    # create empty list to store technology expansion
    techExp = []

    # iterate over scenarios
    for scenario in _energy_sim_params.scenarios.keys():

        if 'historical' in scenario or 'nuclear' in scenario:
            continue

        # create empty list to store total costs
        for l in labels:
            tcosts[l][scenario] = []
            tcosts[l+' - decision rule'][scenario] = []

        # pass input data to model
        print("building the model...")
        print('scenario:', scenario)
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
        np.random.seed(0)

        all_shares_exo = []
        
        print('start:', model.mode)
        for n in range(nsim):
            # for each cost assumption, compute total costs
            # and append it to the dictionary
            # 1e-12 is used to convert from USD to trillion USD
            for l in labels:
                #if scenario == sim_scenario:
                
                #print("simulating...",n)
                tcosts[l][scenario].append( 1e-12 * model.simulate())
                if scenario == sim_scenario and savegif:
                    #print("saving the figure...")
                    #model.make_gif(f'static_{scenario.replace(" ", "_")}_{n}')
                    
                    #model.plotFinalEnergyBySource(label,filename=f'{n}_static_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotFinalEnergy(label,filename=f'{n}_static_{scenario.replace(" ", "_")}_{n}')
                    
                    #model.plotIndividualTechAreas(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotCapacityExpansion(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotNewBuildsAndRetirements(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
                    shares_df = model.get_generation_shares()
                    #print(shares_df)
                    all_shares_exo.append(shares_df)
        
      
        # set policy mode
        model.mode = 'policy'
        print('start:', model.mode)

        # load policy file
        #model.policy.load('energySim' + os.path.sep + 'fast_transition_policy_new.pth')
        policy_path = f'results/{label}_{sim_scenario}_policy.pth'
        model.policy.load(policy_path)
        # run multiple iterations to explore cost parameters' uncertainty
        np.random.seed(1)
        
        all_shares_pol = []
        for n in range(nsim):
            # for each cost assumption, compute total costs
            # and append it to the dictionary
            # 1e-12 is used to convert from USD to trillion USD
            for l in labels:
                #if scenario == sim_scenario:
                
                #print("simulating...",n)
                tcosts[l+' - decision rule'][scenario].append( 1e-12 * model.simulate())
                if scenario == sim_scenario  and savegif:
                    print("saving the figure...")
                    #model.make_gif(f'dynamic_{scenario.replace(" ", "_")}_{n}')
                    #model.plotFinalEnergyBySource(label, filename=f'{n}_dynamic_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotFinalEnergy(label,filename=f'{n}_dynamic_{scenario.replace(" ", "_")}_{n}')
                    
                    #model.plotIndividualTechAreas(filename=f'dynamic_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotCapacityExpansion(filename=f'dynamic_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotNewBuildsAndRetirements(filename=f'dynamic_area_{scenario.replace(" ", "_")}_{n}')
                    shares_df = model.get_generation_shares()
                    #print('policy share', shares_df)
                    all_shares_pol.append(shares_df)
    
     

                
        
        
        if scenario == sim_scenario :


            if new_sharebox:
                print('shart sharebox')

                # 是否画 exogenous 结果的开关
                plot_exogenous = False

                # 关注四种技术
                techs_focus = [
                    'solar pv electricity',
                    'wind electricity',
                    'SMR electricity',
                    'SMR2 electricity'
                ]
                # 选择年份
                plot_years = [2030, 2070]

                # 若无exogenous数据且需要画exogenous，安全退出避免报错
                if plot_exogenous and len(all_shares_exo) == 0:
                    print('Warning: all_shares_exo is empty. Skipping new_sharebox plotting.')
                else:
                    # 准备policy数据：每个仿真一行
                    policy_shares = {
                        y: pd.DataFrame([df.loc[y] for df in all_shares_pol])
                        for y in plot_years
                    }

                    # 如需画 exogenous，取第一个仿真（exo路径唯一）
                    if plot_exogenous:
                        exo_share_df = all_shares_exo[0]

                    # 单个图上并排展示每个tech在两个年份的box
                    fig, ax = plt.subplots(figsize=(14, 7))
                    colors = {
                        'solar pv electricity': 'orange',
                        'wind electricity': 'deepskyblue',
                        'SMR electricity': 'steelblue',
                        'SMR2 electricity': 'purple',   # 新颜色
                    }

                    base_positions = np.arange(1, len(techs_focus) + 1)
                    offset = 0.18
                    positions_2030 = base_positions - offset
                    positions_2070 = base_positions + offset

                    data_2030 = [policy_shares[2030][t] for t in techs_focus]
                    data_2070 = [policy_shares[2070][t] for t in techs_focus]

                    box2030 = ax.boxplot(
                        data_2030,
                        positions=positions_2030,
                        widths=0.32,
                        patch_artist=True
                    )
                    for patch, t in zip(box2030['boxes'], techs_focus):
                        patch.set_facecolor(colors[t])
                        patch.set_alpha(0.7)

                    box2070 = ax.boxplot(
                        data_2070,
                        positions=positions_2070,
                        widths=0.32,
                        patch_artist=True
                    )
                    for patch, t in zip(box2070['boxes'], techs_focus):
                        patch.set_facecolor(colors[t])
                        patch.set_alpha(0.4)

                    if plot_exogenous:
                        exo_values_2030 = [exo_share_df.loc[2030][t] for t in techs_focus]
                        exo_values_2070 = [exo_share_df.loc[2070][t] for t in techs_focus]
                        # LP markers: both squares, same color
                        ax.scatter(
                            positions_2030,
                            exo_values_2030,
                            color='black',
                            s=120,
                            marker='s',
                            label='LP(Deterministic)',
                            zorder=10
                        )
                        ax.scatter(
                            positions_2070,
                            exo_values_2070,
                            color='black',
                            s=120,
                            marker='s',
                            zorder=10
                        )

                    ax.set_xticks(base_positions)
                    ax.set_xticklabels(techs_focus)
                    ax.set_title(
                        'Generation share by technology at year 2030 VS 2070',
                        fontsize=28,
                        weight='bold'
                    )
                    ax.set_ylabel('Share of Final Energy', fontsize=24, weight='bold')

                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    ax.tick_params(axis='x', labelsize=22, pad=18)
                    ax.tick_params(axis='y', labelsize=22)

                    # 只有在画 exogenous 时才加 legend（否则只有一类 box，无 legend 也可以）
                    if plot_exogenous:
                        ax.legend(loc='upper right', fontsize=20)

                    plt.tight_layout()
                    fig.subplots_adjust(bottom=0.25)
                    save_dir = f"./results/figures/{label}"
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f"{save_dir}/share_boxplot_wind_solar_smr_smr2_grouped_2030_2070.png")
                    plt.show()

            


            if save_sharebox:
                ## make share boxplot (end-of-century share for each tech, all simulations)
                #only for policy model: since exogenous, same energy transition pathway

                # modes = ['exogenous', 'policy']
                # for mode in modes:
                #     if mode == 'exogenous':
                #         all_shares = all_shares_exo
                #     else:
                #         all_shares = all_shares_pol
                 
        
                    
                years = [2030, 2050,2070]
                for plot_year in years:
                    
                    print(plot_year)
                    #plot_year = 2030
                    #last_year = model.yend

                    # 准备policy的boxplot数据
                    policy_box_data = pd.DataFrame([df.loc[plot_year] for df in all_shares_pol])
                    techs = policy_box_data.columns.tolist()
                    techs = [t for t in techs if t in ['solar pv electricity',
                                                       'wind electricity',
                                                       'SMR electricity']]


                    # 准备exogenous的线条数据（取第一个simulation，因为exogenous都一样）
                    exo_line_data = all_shares_exo[0].loc[plot_year]
            
                    # box_data = pd.DataFrame([df.loc[plot_year] for df in all_shares])
                    # techs = box_data.columns.tolist()

                    # colors = ['black','saddlebrown','darkgray',
                    #         'saddlebrown','darkgray',
                    #         'magenta','royalblue',
                    #         'forestgreen','deepskyblue',
                    #         'orange','pink','plum','lawngreen', 'burlywood']
                    # colors = ['black','saddlebrown','darkgray',
                    #         'saddlebrown','darkgray',
                    #         'magenta','royalblue',
                    #         'forestgreen','deepskyblue',
                    #         'orange','steelblue','pink','plum','lawngreen','burlywood']
                    
                    colors = ['orange', 'deepskyblue', 'steelblue']  # solar, wind, SMR
                    #plt.figure(figsize=(14,6))

                    fig, ax = plt.subplots(figsize=(14, 6))
                     # 绘制policy的boxplot
                    box = ax.boxplot([policy_box_data[t] for t in techs], 
                                    patch_artist=True, 
                                    labels=techs,
                                    positions=range(1, len(techs) + 1))
                    
                    # 设置boxplot颜色
                    for patch, color in zip(box['boxes'], colors):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)  # 设置透明度以便看到线条
                    
                    # 绘制exogenous的黑色粗线
                    x_positions = range(1, len(techs) + 1)
                    exo_values = [exo_line_data[tech] for tech in techs]
                    ax.scatter(x_positions, exo_values, 
                            color='black', 
                            s=100,  # 点的大小
                            marker='o',  
                            label='Exogenous',
                            zorder=10)  # 确保点在最上层
                    x_positions = range(1, len(techs) + 1)
                 

                    ax.set_ylabel(f'Share of Final Energy in {plot_year}')
                    ax.set_xlabel('Technology')
                    ax.set_title(f'Comparison: Exogenous vs Policy Generation Share in {plot_year}')
                    #ax.tick_params(axis='x', rotation=45)
                    ax.tick_params(axis='x')
                    
                    # 添加图例
                    ax.legend(loc='upper right')
                    
                    plt.tight_layout()
                    plt.savefig(f'results/figures/{label}/comparison_{plot_year}_share_boxplot.png')
                    plt.show()
                    # box = plt.boxplot([box_data[t] for t in techs], patch_artist=True, labels=techs)
                    # for patch, color in zip(box['boxes'], colors):
                    #     patch.set_facecolor(color)
                    # plt.ylabel(f'Share of Final Energy in {plot_year}')
                    # plt.xlabel('Technology')
                    # plt.title(f'{mode}: Distribution of Year {plot_year} Generation Share by Technology')
                    # plt.xticks(rotation=45)
                    # plt.tight_layout()
                    # #plt.savefig('./figures/end_century_share_boxplot.png')
                    
                    # plt.savefig(f'results/figures/{label}/{mode}_{plot_year}_share_boxplot.png')
            
                    # plt.show()


                # # plot mid century share
         
                # mid_year = 2050
                # box_data = pd.DataFrame([df.loc[mid_year] for df in all_shares])
                # techs = box_data.columns.tolist()

                
                # plt.figure(figsize=(14,6))
                # box = plt.boxplot([box_data[t] for t in techs], patch_artist=True, labels=techs)
                # for patch, color in zip(box['boxes'], colors):
                #     patch.set_facecolor(color)
                # plt.ylabel('Share of Final Energy in 2050')
                # plt.xlabel('Technology')
                # plt.title('Distribution of Mid-of-Century Generation Share by Technology ')
                # plt.xticks(rotation=45)
                # plt.tight_layout()
                # plt.savefig(f'results/figures/{label}/mid_century_share_boxplot.png')
                # plt.show()
          
          
            if save_pc:

                # parallel coordinates, only for dps
                # Each line is a simulation, each axis is a technology, value is the share in 2100.

                from pandas.plotting import parallel_coordinates
                pc_data = box_data.copy()
                pc_data['sim'] = pc_data.index.astype(str)
                plt.figure(figsize=(14,6))
                parallel_coordinates(pc_data, 'sim', color=plt.cm.tab20.colors, alpha=0.3)
                plt.ylabel('Share of Final Energy in 2100')
                plt.xlabel('Technology')
                plt.title('Parallel Coordinates Plot: End-of-Century Generation Share')
                plt.xticks(rotation=45)
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                plt.savefig('./figures/end_century_share_parallelcoords.png')
                plt.show()


        
        

    # # create cost dataframe from dictionary, update columns,
    # #  and focus on relevant scenarios
    df = pd.DataFrame(tcosts).stack().explode().reset_index()
    df.columns = ['Scenario',
                'Learning rate assumptions', 
                'Net Present Cost [trillion USD]']
    df = df.loc[~df['Scenario'].str.contains('nuclear|historical') ]


if savegif:
    plt.close("all")





# make box plot of costs under fast, slow, no transition scenarios

if savebox:

    # convert scenario name to Sentence case formatting
    df['Scenario'] = df['Scenario'].str.title()
    # 只保留 No Transition 和 Fast Transition
    #df = df[~df['Scenario'].str.contains('Slow Transition')]

    # create figure
    fig = plt.figure(figsize=(15,6))

    # add boxplots
    ax = sns.boxplot(data=df, 
                        hue='Scenario', 
                        y='Net Present Cost [trillion USD]', 
                        x='Learning rate assumptions', 
                        hue_order=['No Transition',
                                   'Slow Transition',
                          
                                    'Fast Transition'],
                        width=0.5, 
                        whis=(5,95),  # whis显示5% - 95% whis = (5,95)， all range (0,100)
                        linewidth=1.75,
                        palette='colorblind', 
                        gap = 0.2,
                        **{'showfliers':False}) # false: 不显示极端值

    #ax.set_ylim(bottom = 0)
    ax.set_xlabel('')

    # set x-axis labels
    ax.set_xticks(ax.get_xticks(),
                [label.get_text().replace(' - ', '\n') 
                for label in ax.get_xticklabels()])

    # move legend on the bottom
    sns.move_legend(ax, "lower center", 
                    ncol=3, bbox_to_anchor=(0.5, -0.6))

    # adjust figure
    fig.subplots_adjust(bottom=0.375, top=0.95, 
                        left=0.075, right=0.95)

    # add axes explaining boxplot
    axes = fig.add_axes([0.8, -0.05, 0.2, 0.35])
    axes.grid(False)
    axes.set_axis_off()
    axes.plot([0,.5], [1,1], color='black')
    axes.plot([0,.5,.5,0,0], [0.5,0.5,1.5,1.5,0.5], color='black')
    axes.fill_between([0,.5], [.5,.5], [1.5,1.5], color='black', alpha=.2)
    axes.plot([0,.5], [0,0], color='black')
    axes.plot([0,.5], [2,2], color='black')
    axes.plot([0.25,0.25], [0,.5], color='black')
    axes.plot([0.25,0.25], [1.5,2], color='black')
    axes.set_ylim(-1,3)
    axes.set_xlim(-1.8,3)
    fontsize = 14
    axes.annotate('50%', xy=(-.5,1),
                        ha='center', va='center',
                        xycoords='data', 
                        fontsize=fontsize)
    axes.annotate('90%', xy=(-1.5,1),
                        ha='center', va='center',
                        xycoords='data',
                        fontsize=fontsize)
    axes.annotate('Median', xy=(.6,1),
                    ha='left', va='center',
                        xycoords='data',
                        fontsize=fontsize)
    axes.plot([-.1,-.5,-.5], [1.5,1.5,1.25], color='black')
    axes.plot([-.1,-.5,-.5], [.5,.5,.75], color='black')
    axes.plot([-.1,-1.5,-1.5], [2,2,1.25], color='silver')
    axes.plot([-.1,-1.5,-1.5], [0,0,.75], color='silver')

    save_dir = f"./results/figures/{label}"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/validation_W1_no_outlier_total_discounted_costs.pdf")   

    #fig.savefig(f"./results/figures/total_discounted_costs.pdf")

    plt.show()