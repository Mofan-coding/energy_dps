import energySim._energy_sim_model as _energy_sim_model
import energySim._energy_sim_params as _energy_sim_params
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rc('savefig', dpi=300)
sns.set_style('whitegrid')
sns.set_context('talk')
matplotlib.rc('font',
                **{'family':'sans-serif',
                   'sans-serif':'Helvetica'})

# iterate over scenarios
for scenario in _energy_sim_params.scenarios.keys():

    # pass input data to model
    model = _energy_sim_model.EnergyModel(\
                EFgp = _energy_sim_params.scenarios[scenario][0],
                slack = _energy_sim_params.scenarios[scenario][1])

    # simulate model
    model.simulate()

    # plot some figures for quick check
    model.plotDemand()
    model.plotFinalEnergyBySource()
    model.plotS7()

    # compute costs with one random set of parameters
    model.computeCost(\
                _energy_sim_params.costparams, 
                learningRateTechs = \
                    _energy_sim_params.learningRateTechs)
    
    # plot costs
    model.plotCostBySource()

    plt.show()

plt.show()
