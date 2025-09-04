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

nsim =300
label = '090203'
sim_scenario = 'fast transition'

gt_clip = 1
hidden_size = 32
input_norm = False


savegif = False #individual simulation dynamics 
savebox= True # boxplot of costs 
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
                    print("saving the figure...")
                    #model.make_gif(f'static_{scenario.replace(" ", "_")}_{n}')
                    
                    #model.plotFinalEnergyBySource(label,filename=f'{n}_static_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotFinalEnergy(label,filename=f'{n}_static_{scenario.replace(" ", "_")}_{n}')
                    
                    #model.plotIndividualTechAreas(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotCapacityExpansion(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
                    #model.plotNewBuildsAndRetirements(filename=f'static_area_{scenario.replace(" ", "_")}_{n}')
    
        
      
        # set policy mode
        model.mode = 'policy'

        # load policy file
        #model.policy.load('energySim' + os.path.sep + 'fast_transition_policy_new.pth')
        policy_path = f'results/{label}_{sim_scenario}_policy.pth'
        model.policy.load(policy_path)
        # run multiple iterations to explore cost parameters' uncertainty
        np.random.seed(0)
        all_shares = []
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
                    #print(shares_df)
                    all_shares.append(shares_df)
    
     

                
        
        
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
                        whis=(5,95),  # whis显示5% - 95% whis = (5,95)， all range (0,100)
                        linewidth=1.75,
                        palette='colorblind', 
                        gap = 0.2,
                        **{'showfliers':False}) # false: 不显示极端值

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
    fig.savefig(f"{save_dir}/no_outlier_total_discounted_costs.pdf")   

    #fig.savefig(f"./results/figures/total_discounted_costs.pdf")

    plt.show()