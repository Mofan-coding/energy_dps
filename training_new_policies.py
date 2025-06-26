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

label = 062501
scenario = 'fast'

# select the number of cost projection simulations
# needed to explore parameters' uncertainty
# used only if new simulations are run
nsim = 1   #100


# create labels for different cost assumptions
labels = ['Way et al. (2022)']

# define colors for technologies
techcolors = ['black','saddlebrown','darkgray',
                  'saddlebrown','darkgray',
                  'magenta','royalblue',
                  'forestgreen','deepskyblue',
                  'orange','pink','plum','lawngreen', 'burlywood'] 

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

        if 'fast' not in scenario:
            continue

        # create empty list to store total costs
        for l in labels:
            tcosts[l][scenario] = []
            tcosts[l+' - decision rule'][scenario] = []

        # pass input data to model
        model = _energy_sim_model.EnergyModel(\
                    EFgp = _energy_sim_params.scenarios[scenario][0],
                    slack = _energy_sim_params.scenarios[scenario][1],
                    costparams=_energy_sim_params.costsAssumptions['Way et al. (2022)'],)

        ## simulate model

        # set simulation mode
        model.mode = 'exogenous'
        for n in range(nsim):
            # for each cost assumption, compute total costs
            # and append it to the dictionary
            # 1e-12 is used to convert from USD to trillion USD
            for l in labels:
                tcosts[l][scenario].append( 1e-12 * model.simulate())

        # set policy mode
        model.mode = 'policy'

        # train model
        #
        # model.policy.train(iter=100, batch_size=30, popsize=5, dist=False)
        model.policy.train(iter=100, batch_size=30, popsize=5, dist=False)
        os.makedirs('results', exist_ok=True)
        policy_path = f'results/{label}_{scenario}_policy.pth'
        #model.policy.save('energySim' + os.path.sep + '_'.join(scenario.split(' ')) + '_policy_new_training.pth')
        model.policy.save(policy_path)

        # run multiple iterations to explore cost parameters' uncertainty
        for n in range(nsim):
            # for each cost assumption, compute total costs
            # and append it to the dictionary
            # 1e-12 is used to convert from USD to trillion USD
            for l in labels:
                tcosts[l+' - decision rule'][scenario].append( 1e-12 * model.simulate())

        # append technology expansion to list
        for t in model.technology[5:13]:
            techExp.append([t, scenario,
                             model.z[t][0], model.z[t][-1],
                            model.c[t][0], 
                            _energy_sim_params.costparams['omega'][t]])

    # # create dataframe from dictionary, update columns,
    # #  and focus on relevant scenarios
    df = pd.DataFrame(tcosts).stack().explode().reset_index()
    df.columns = ['Scenario',
                'Learning rate assumptions', 
                'Net Present Cost [trillion USD]']
    df = df.loc[~df['Scenario'].str.contains('nuclear|historical') ]

    # # save dataframe to csv
    # df.to_csv('energySim' + os.path.sep + 'Costs_all.csv')

    # # convert tech expansion list to dataframe
    # df = pd.DataFrame(techExp, 
    #                 columns=['Technology', 
    #                         'Scenario', 
    #                         'Reference production [EJ]',
    #                         'Final production [EJ]',
    #                         'Reference cost [USD/GJ]',
    #                         'Learning exponent'])
    # df.to_csv('energySim' + os.path.sep + 'TechnologyExpansion.csv')

# read data
# df = pd.read_csv('energySim' + os.path.sep + 'Costs_all.csv')

# convert scenario name to Sentence case formatting
df['Scenario'] = df['Scenario'].str.title()
print(df)
# # create figure
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

plt.gca().set_xlabel('Experience curve assumptions')

# set x-axis labels
plt.gca()\
    .set_xticks(plt.gca().get_xticks(),
        [label.get_text().replace(' - ', '\n') \
            for label in ax.get_xticklabels()])

# move legend on the bottom
sns.move_legend(ax, "lower center", 
                ncol=3, bbox_to_anchor=(0.5, -0.6))

# adjust figure
plt.subplots_adjust(bottom=0.375, top=0.95, 
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

plt.show()
