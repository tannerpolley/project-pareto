#%%
import numpy as np
from scipy.stats import qmc
import pandas as pd
from pareto.models_extra.desalination_models import nf_Li
from pyomo.environ import value


def run_simulation(flowrate, salinity):
    model = nf_Li.build()
    nf_Li.set_operating_conditions(
        model,
        feed_flow_mass=flowrate,
        Li_conc=salinity,
    )
    nf_Li.initialize_system(model)
    nf_Li.optimize_set_up(model)
    nf_Li.solve(model, show=False)
    # percent_Li_efficiency = model.fs.nf.rejection_intrinsic_phase_comp[0, "Liq", "Li_+"].value * 100
    # CAPEX = value(model.fs.costing.total_capital_cost * model.fs.costing.capital_recovery_factor)
    # OPEX = value(model.fs.costing.total_operating_cost)
    C_annual = value(model.fs.costing.total_annualized_cost)


    mole_flow_rate_Li = (value(model.fs.retentate.flow_mol_phase_comp[0, 'Liq', 'Li_+']))  # mol/s
    mass_flow_rate_Li = mole_flow_rate_Li*.006941 # kg/s
    price_lithium_per_kg = 10.739 # $/kg
    price_year_lithium = mass_flow_rate_Li*price_lithium_per_kg*3.154e7 # $/year
    profit = price_year_lithium - C_annual
    # print(CAPEX, OPEX, price_year_lithium)
    # return CAPEX, OPEX, percent_Li_efficiency, price_year_lithium
    print(profit)
    return profit


# Number of runs and inputs/outputs
n_runs = 1000
n_inputs = 2
n_outputs = 5

# Define bounds for the inputs
bounds = [[1, 20], [0.001, 0.01]]

# Generate Latin Hypercube Sampling for inputs
sampler = qmc.LatinHypercube(d=n_inputs)
lhs_sample = sampler.random(n=n_runs)

# Scale inputs to the specified bounds
# scaled_inputs = qmc.scale(lhs_sample, [bound[0] for bound in bounds], [bound[1] for bound in bounds])
flowrate = np.linspace(3, 4, 20)
salinity = np.linspace(.004, .0045, 20)
scaled_inputs = np.concatenate((flowrate, salinity))

# Generate outputs using a for loop
outputs = np.zeros((n_runs, n_outputs))
profit_array = np.zeros((len(flowrate), len(salinity)))
for i, flow in enumerate(flowrate):
    for j, sal in enumerate(salinity):
        try:
            # output1, output2, output3, output4 = run_simulation(flow, sal)
            profit = run_simulation(flow, sal)
            status = 1
            profit_array[i, j] = profit
        except:
            output1, output2, output3, output4 = 0, 0, 0, 0
            profit = np.nan
            status = 0
            profit_array[i, j] = profit



#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X, Y = np.meshgrid(flowrate, salinity)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, profit_array, cmap='viridis')
print(min(profit_array[:,-1]))
# ax.set_zlim(-2000000, 4000000)
plt.show()

#%%

# Combine inputs and outputs for clarity
combined_data = np.hstack([scaled_inputs, outputs])

# Create a DataFrame for better readability
columns = ['Flow (L/s)', 'Inlet TDS (kg/L)', 'CAPEX (kUSD/year)', 'OPEX (kUSD/year)', 'Li Efficiency (%)', 'Li Profit (kUSD/year)', 'Status']
df = pd.DataFrame(combined_data, columns=columns)
df.to_csv(r'C:\Users\Tanner\Documents\git\project-pareto\pareto\examples\desalination_jupyter_notebooks\nf_data_profit.csv', index=False)