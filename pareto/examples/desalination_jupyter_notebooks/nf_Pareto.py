#%%
import numpy as np
from scipy.stats import qmc
import pandas as pd
from pareto.models_extra.desalination_models import nf_Li
from pyomo.environ import value
from idaes.core.util.exceptions import InitializationError


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
    CAPEX = value(model.fs.costing.total_capital_cost * model.fs.costing.capital_recovery_factor)
    OPEX = value(model.fs.costing.total_operating_cost)
    return CAPEX, OPEX, percent_Li_efficiency


# Number of runs and inputs/outputs
n_runs = 5000
n_inputs = 2
n_outputs = 3

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
    except:
        output1, output2, output3 = 0, 0, 0
    print(output1, output2, output3)
    outputs[i] = [output1, output2, output3]

#%%

# Combine inputs and outputs for clarity
combined_data = np.hstack([scaled_inputs, outputs])

# Create a DataFrame for better readability
# columns = [f"Input_{i+1}" for i in range(n_inputs)] + [f"Output_{i+1}" for i in range(n_outputs)]
columns = ['Inlet TDS (kg/L)', 'Flow (L/s)', 'CAPEX (kUSD/year)', 'CAPEX (kUSD/year)', 'Li Efficiency (%)']
df = pd.DataFrame(combined_data, columns=columns)

df = df[~(df == 0).any(axis=1)]
df['CAPEX (kUSD/year)'] = df['CAPEX (kUSD/year)']/1000
df['CAPEX (kUSD/year)'] = df['CAPEX (kUSD/year)']/1000
df.columns = ['Flow (L/s)', 'Inlet TDS (kg/L)', 'CAPEX (kUSD/year)', 'OPEX (kUSD/year)', 'Li Efficiency']
df.to_csv(r'pareto\examples\desalination_jupyter_notebooks\nf_data_2.csv', index=False)