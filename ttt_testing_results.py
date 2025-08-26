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
label = '082201'
sim_scenario = 'fast transition'

gt_clip = 1
hidden_size = 2
input_norm = False


savegif = True #individual simulation dynamics 
savebox= False# boxplot of costs 
save_sharebox = False  #Boxplot of End-of-Century Generation Share
save_pc = False  #Parallel Coordinates Plot



# create labels for different cost assumptions
labels = ['Way et al. (2022)']

# define colors for technologies
techcolors = ['black', 'saddlebrown', 'darkgray', 'saddlebrown', 'darkgray',
              'magenta', 'royalblue', 'forestgreen', 'deepskyblue',
              'orange', 'pink', 'plum', 'lawngreen', 'burlywood'] 

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

        all_costs_exo = []
        all_q_exo = []
        all_c_exo = []
        
        
        for n in range(nsim):
            # for each cost assumption, compute total costs
            # and append it to the dictionary
            # 1e-12 is used to convert from USD to trillion USD
            for l in labels:
                print("simulating...")
                cost = 1e-12 * model.simulate()
                tcosts[l][scenario].append(cost)
                all_costs_exo.append(cost)
                all_q_exo.append(copy.deepcopy(model.q))
                all_c_exo.append(copy.deepcopy(model.c))

                #tcosts[l][scenario].append( 1e-12 * model.simulate())
                if scenario == sim_scenario and savegif:
                    print("saving the figure...")
                    #model.make_gif(f'static_{scenario.replace(" ", "_")}_{n}')
                    
                    #model.plotFinalEnergyBySource(label,filename=f'{n}_static_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotFinalEnergy(label,filename=f'{n}_static_{scenario.replace(" ", "_")}_{n}')
                    
                    #model.plotIndividualTechAreas(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotCapacityExpansion(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotNewBuildsAndRetirements(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
               
        # 找最低和最高cost索引
        idx_min_exo = np.argmin(all_costs_exo)
        idx_max_exo = np.argmax(all_costs_exo)

        """
        p15 = np.percentile(all_costs_exo, 15)
        p85 = np.percentile(all_costs_exo, 85)
        idx_min_exo = np.argmin(np.abs(np.array(all_costs_exo) - p15))
        idx_max_exo = np.argmin(np.abs(np.array(all_costs_exo) - p85))
        """



        # --- Policy mode ---

        # set policy mode
        model.mode = 'policy'

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

        for n in range(nsim):
            # for each cost assumption, compute total costs
            # and append it to the dictionary
            # 1e-12 is used to convert from USD to trillion USD
            for l in labels:
                print("simulating...")
                #tcosts[l+' - decision rule'][scenario].append( 1e-12 * model.simulate())
                cost = 1e-12 * model.simulate()
                tcosts[l+' - decision rule'][scenario].append(cost)
                all_costs_policy.append(cost)
                all_q_policy.append(copy.deepcopy(model.q))
                all_c_policy.append(copy.deepcopy(model.c))
                
                if scenario == sim_scenario  and savegif:
                    print("saving the figure...")
                    #model.make_gif(f'dynamic_{scenario.replace(" ", "_")}_{n}')
                    #model.plotFinalEnergyBySource(label, filename=f'{n}_dynamic_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotFinalEnergy(label,filename=f'{n}_dynamic_{scenario.replace(" ", "_")}_{n}')
                    
                    #model.plotIndividualTechAreas(filename=f'dynamic_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotCapacityExpansion(filename=f'dynamic_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotNewBuildsAndRetirements(filename=f'dynamic_area_{scenario.replace(" ", "_")}_{n}')
                    shares_df = model.get_generation_shares()
                    #print(shares_df)
                    all_shares.append(shares_df)
        
        #idx_min_policy = np.argmin(all_costs_policy)
        #idx_max_policy = np.argmax(all_costs_policy)
        p15 = np.percentile(all_costs_policy, 15)
        p85 = np.percentile(all_costs_policy, 85)

        idx_min_policy = np.argmin(np.abs(np.array(all_costs_policy) - p15))
        idx_max_policy = np.argmin(np.abs(np.array(all_costs_policy) - p85))        


        def plot_final_energy_by_source(q_dict, label, title, name=None):
            colors = ['black','saddlebrown','darkgray',
                    'saddlebrown','darkgray',
                    'magenta','royalblue',
                    'forestgreen','deepskyblue',
                    'orange','pink','plum','lawngreen', 'burlywood'] 
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
        
            
        # --- 画最低和最高 cost 的轨迹并保存 ---
        plot_final_energy_by_source(all_q_exo[idx_min_exo], label, "Exogenous: Final Energy (Lowest Cost)", name="exo_lowest_final_energy")
        #plot_solar_cost(all_c_exo[idx_min_exo], label, "Exogenous: Solar PV Cost (Lowest Cost)", name="exo_lowest_solar_cost")
        plot_all_tech_costs(all_c_exo[idx_min_exo], label, "Exogenous: All Tech Costs (Lowest Cost)", name="exo_lowest_all_tech_costs")
        plot_q_vs_time(all_q_exo[idx_min_exo], label, "Exogenous: q vs Time (Lowest Cost)", name="exo_lowest_q_vs_time")

        plot_final_energy_by_source(all_q_exo[idx_max_exo], label, "Exogenous: Final Energy (Highest Cost)", name="exo_highest_final_energy")
        #plot_solar_cost(all_c_exo[idx_max_exo], label, "Exogenous: Solar PV Cost (Highest Cost)", name="exo_highest_solar_cost")
        plot_all_tech_costs(all_c_exo[idx_max_exo], label, "Exogenous: All Tech Costs (Highest Cost)", name="exo_highest_all_tech_costs")
        plot_q_vs_time(all_q_exo[idx_max_exo], label, "Exogenous: q vs Time (Highest Cost)", name="exo_Highest_q_vs_time")

        #### ------ Policy

        """
        plot_final_energy_by_source(all_q_policy[idx_min_policy], label, "Policy: Final Energy (Lowest Cost)", name="policy_lowest_final_energy")
        #plot_solar_cost(all_c_policy[idx_min_policy], label, "Policy: Solar PV Cost (Lowest Cost)", name="policy_lowest_solar_cost")
        plot_all_tech_costs(all_c_policy[idx_min_policy], label, "Policy: All Tech Costs (Lowest Cost)", name="policy_lowest_all_tech_costs")
        plot_q_vs_time(all_q_policy[idx_min_policy], label, "Policy: q vs Time (Lowest Cost)", name="policy_lowest_q_vs_time")

        plot_final_energy_by_source(all_q_policy[idx_max_policy], label, "Policy: Final Energy (Highest Cost)", name="policy_highest_final_energy")
        #plot_solar_cost(all_c_policy[idx_max_policy], label, "Policy: Solar PV Cost (Highest Cost)", name="policy_highest_solar_cost")
        plot_all_tech_costs(all_c_policy[idx_max_policy], label, "Policy: All Tech Costs (Highest Cost)", name="policy_highest_all_tech_costs")
        plot_q_vs_time(all_q_policy[idx_max_policy], label, "Policy: q vs Time (Highest Cost)", name="policy_highest_q_vs_time")
        """         
        
        
        if scenario == sim_scenario :

            if save_sharebox:
                ## make share boxplot (end-of-century share for each tech, all simulations)
                #only for policy model: since exogenous, same energy transition pathway

                # Get the last year from each simulation
                last_year = model.yend
                
                box_data = pd.DataFrame([df.loc[last_year] for df in all_shares])
                techs = box_data.columns.tolist()

                colors = ['black','saddlebrown','darkgray',
                        'saddlebrown','darkgray',
                        'magenta','royalblue',
                        'forestgreen','deepskyblue',
                        'orange','pink','plum','lawngreen', 'burlywood']
                
                plt.figure(figsize=(14,6))
                box = plt.boxplot([box_data[t] for t in techs], patch_artist=True, labels=techs)
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                plt.ylabel('Share of Final Energy in 2100')
                plt.xlabel('Technology')
                plt.title('Distribution of End-of-Century Generation Share by Technology')
                plt.xticks(rotation=45)
                plt.tight_layout()
                #plt.savefig('./figures/end_century_share_boxplot.png')
                
                plt.savefig(f'results/figures/{label}/end_century_share_boxplot.png')
           
                plt.show()

                # plot mid century share
         
                mid_year = 2050
                box_data = pd.DataFrame([df.loc[mid_year] for df in all_shares])
                techs = box_data.columns.tolist()

                
                plt.figure(figsize=(14,6))
                box = plt.boxplot([box_data[t] for t in techs], patch_artist=True, labels=techs)
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                plt.ylabel('Share of Final Energy in 2050')
                plt.xlabel('Technology')
                plt.title('Distribution of Mid-of-Century Generation Share by Technology ')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'results/figures/{label}/mid_century_share_boxplot.png')
                plt.show()
          
          
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
                        whis=(5,95),
                        linewidth=1.75,
                        palette='colorblind', 
                        gap = 0.2,
                        **{'showfliers':False})

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
    fig.savefig(f"{save_dir}/total_discounted_costs.pdf")   

    #fig.savefig(f"./results/figures/total_discounted_costs.pdf")

    plt.show()