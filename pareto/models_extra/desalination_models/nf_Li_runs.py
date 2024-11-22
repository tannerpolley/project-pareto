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
    percent_Li_efficiency =  model.fs.nf.rejection_intrinsic_phase_comp[0, "Liq", "Li_+"].value * 100
    CAPEX = value(model.fs.costing.total_capital_cost * model.fs.costing.capital_recovery_factor)/1000
    OPEX = value(model.fs.costing.total_operating_cost)/1000
    return CAPEX, OPEX, percent_Li_efficiency


# Number of runs and inputs/outputs
n_runs = 1000
n_inputs = 2
n_outputs = 4

# Define bounds for the inputs
bounds = [[1, 20], [0.001, 0.01]]

# Generate Latin Hypercube Sampling for inputs
sampler = qmc.LatinHypercube(d=n_inputs)
lhs_sample = sampler.random(n=n_runs)

# Scale inputs to the specified bounds
scaled_inputs = qmc.scale(lhs_sample, [bound[0] for bound in bounds], [bound[1] for bound in bounds])

# Generate outputs using a for loop
outputs = np.zeros((n_runs, n_outputs))  # Placeholder for outputs
for i in range(n_runs):
    input1, input2 = scaled_inputs[i]
    try:
        output1, output2, output3 = run_simulation(input1, input2)
        status = 1
    except:
        output1, output2, output3 = 0, 0, 0
        status = 0
    outputs[i] = [output1, output2, output3, status]

#%%

# Combine inputs and outputs for clarity
combined_data = np.hstack([scaled_inputs, outputs])

# Create a DataFrame for better readability
columns = ['Inlet TDS (kg/L)', 'Flow (L/s)', 'CAPEX (kUSD/year)', 'OPEX (kUSD/year)', 'Li Efficiency (%)', 'Status']
df = pd.DataFrame(combined_data, columns=columns)
df.to_csv(r'C:\Users\Tanner\Documents\git\project-pareto\pareto\examples\desalination_jupyter_notebooks\nf_data.csv', index=False)