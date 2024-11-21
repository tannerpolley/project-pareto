#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2024 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
####
# This code has been adapted from a test in the WaterTAP repo to
# simulate a simple separation of lithium and magnesium
#
# watertap > unit_models > tests > test_nanofiltration_DSPMDE_0D.py
# test defined as test_pressure_recovery_step_2_ions()
#
# also used the following flowsheet as a reference
# watertap > examples > flowsheets > nf_dspmde > nf.py
#
# https://github.com/watertap-org/watertap/blob/main/tutorials/nawi_spring_meeting2023.ipynb
####
"""
Nanofiltration flowsheet for Donnan steric pore model with dielectric exclusion
"""

# import statements
from pyomo.environ import (
    value,
    ConcreteModel,
    Constraint,
    Objective,
    TransformationFactory,
    floor,
    log10,
)
from pyomo.network import Arc

import idaes.core.util.scaling as iscale
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Feed, Product

from idaes.core.util import DiagnosticsToolbox
import idaes.logger as idaeslog

from watertap.core.solvers import get_solver
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    ActivityCoefficientModel,
    DensityCalculation,
    MCASParameterBlock,
)
from watertap.unit_models.nanofiltration_DSPMDE_0D import NanofiltrationDSPMDE0D
from watertap.unit_models.pressure_changer import Pump
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.pump import cost_pump

_log = idaeslog.getLogger(__name__)


def main():
    """
    Builds and solves the NF flowsheet

    Returns:
        m: pyomo model
    """
    solver = get_solver()
    m = build()
    set_operating_conditions(m)
    initialize_system(m, solver)
    _log.info("Initialization Okay")
    if degrees_of_freedom(m) != 0:
        raise ValueError("Degrees of freedom were not equal to zero")
    optimize_set_up(m)
    solve(m, solver)
    _log.info("Solved Box Problem")
    m.fs.nf.report()
    solve(m, solver)
    # m.fs.nf.report()
    print("Optimal NF feed pressure (Bar)", m.fs.pump.outlet.pressure[0].value / 1e5)
    print("Optimal area (m2)", m.fs.nf.area.value)
    print(
        "Optimal NF vol recovery (%)",
        m.fs.nf.recovery_vol_phase[0.0, "Liq"].value * 100,
    )
    print(
        "Optimal Li rejection (%)",
        m.fs.nf.rejection_intrinsic_phase_comp[0, "Liq", "Li_+"].value * 100,
    )
    print(
        "total annualized capital cost: %.2f $/yr"
        % value(m.fs.costing.total_capital_cost * m.fs.costing.capital_recovery_factor)
    )
    print(
        "total annual operating cost: %.2f $/yr"
        % value(m.fs.costing.total_operating_cost)
    )

    return m

def build():
    """
    Builds the NF flowsheet

    Returns:
        m: pyomo model
    """
    # create the model
    m = ConcreteModel()

    # create the flowsheet
    m.fs = FlowsheetBlock(dynamic=False)

    # define the property model
    default = {
        "solute_list": ["Li_+", "Cl_-"],
        "diffusivity_data": {
            ("Liq", "Li_+"): 1.03e-09,
            ("Liq", "Cl_-"): 2.03e-09,
        },
        "mw_data": {"H2O": 0.018, "Li_+": 0.0069, "Cl_-": 0.035},
        "stokes_radius_data": {
            "Li_+": 3.61e-10,
            # "Mg_2+": 4.07e-10,
            # "Cl_-": 3.28e-10
            # adjusted Cl and Mg to values from nf.py'
            "Cl_-": 0.121e-9,
        },
        "charge": {"Li_+": 1, "Cl_-": -1},
        "activity_coefficient_model": ActivityCoefficientModel.ideal,
        "density_calculation": DensityCalculation.constant,
    }
    
    m.fs.properties = MCASParameterBlock(**default)

    # add the feed and product streams
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.permeate = Product(property_package=m.fs.properties)
    m.fs.retentate = Product(property_package=m.fs.properties)

    # define unit models
    m.fs.pump = Pump(property_package=m.fs.properties)
    m.fs.nf = NanofiltrationDSPMDE0D(property_package=m.fs.properties)

    # connect the streams and blocks
    m.fs.feed_to_pump = Arc(source=m.fs.feed.outlet, destination=m.fs.pump.inlet)
    m.fs.pump_to_nf = Arc(source=m.fs.pump.outlet, destination=m.fs.nf.inlet)
    m.fs.nf_to_permeate = Arc(
        source=m.fs.nf.permeate, destination=m.fs.permeate.inlet
    )
    m.fs.nf_to_retentate = Arc(
        source=m.fs.nf.retentate, destination=m.fs.retentate.inlet
    )
    TransformationFactory("network.expand_arcs").apply_to(m)
    add_costs(m)

    # Setting default scaling values would go right here
    # Also getting scaling factor values from iscale would go right here

    iscale.calculate_scaling_factors(m)
    return m

def add_costs(m):
    m.fs.costing = WaterTAPCosting()
    m.fs.nf.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.pump.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_pump,
        costing_method_arguments={"pump_type": "low_pressure"},
    )
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.permeate.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.permeate.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.permeate.properties[0].flow_vol)

def set_operating_conditions(m, feed_flow_mass=1, Li_conc = .00119, solver=None):
    """
    Fixes the initial variables needed to create 0 DOF

    Args:
        m: pyomo model
    """

    conc_mass_phase_comp = {"Li_+": Li_conc*1000, "Cl_-": Li_conc*1000/.1957}

    if solver is None:
        solver = get_solver()

    # fix the inlet flow to the block as water flowrate
    m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].fix(feed_flow_mass)

    # fix the ion concentrations and unfix ion flows
    for ion, x in conc_mass_phase_comp.items():
        m.fs.feed.properties[0].conc_mass_phase_comp["Liq", ion].fix(x)
        m.fs.feed.properties[0].flow_mol_phase_comp["Liq", ion].unfix()
    # solve for the new flow rates
    solver.solve(m.fs.feed)
    # fix new water concentration
    m.fs.feed.properties[0].conc_mass_phase_comp["Liq", "H2O"].fix()
    # unfix ion concentrations and fix flows
    for ion, x in conc_mass_phase_comp.items():
        m.fs.feed.properties[0].conc_mass_phase_comp["Liq", ion].unfix()
        m.fs.feed.properties[0].flow_mol_phase_comp["Liq", ion].fix()
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", ion].unfix()
    m.fs.feed.properties[0].conc_mass_phase_comp["Liq", "H2O"].unfix()
    m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].unfix()
    m.fs.feed.properties[0].flow_mol_phase_comp["Liq", "H2O"].fix()

    _add = 0
    for i in m.fs.feed.properties[0].flow_mol_phase_comp:
        scale = -1 * floor(log10((m.fs.feed.properties[0].flow_mol_phase_comp[i].value)))
        print(f"{i} flow_mol_phase_comp scaling factor = {10 ** (scale + _add)}")
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 10 ** (scale + _add), index=i
        )

    # assert electroneutrality
    m.fs.feed.properties[0].assert_electroneutrality(
        defined_state=True, adjust_by_ion="Cl_-", get_property="flow_mol_phase_comp"
    )

    # switching to concentration for ease of adjusting in UI
    # addresses error in fixing flow_mol_phase_comp
    for ion, x in conc_mass_phase_comp.items():
        m.fs.feed.properties[0].conc_mass_phase_comp["Liq", ion].unfix()
        m.fs.feed.properties[0].flow_mol_phase_comp["Liq", ion].fix()

    # pump variables
    m.fs.pump.efficiency_pump[0].fix(0.75)
    m.fs.pump.control_volume.properties_in[0].temperature.fix(298.15)
    m.fs.pump.control_volume.properties_in[0].pressure.fix(101325)
    m.fs.pump.outlet.pressure[0].fix(2e5)
    iscale.set_scaling_factor(m.fs.pump.control_volume.work, 1e-4)

    # membrane operation
    m.fs.nf.recovery_vol_phase[0, "Liq"].setub(0.95)
    # m.fs.nf.recovery_vol_phase[0, "Liq"].fix(recovery)
    m.fs.nf.spacer_porosity.fix(0.85)
    m.fs.nf.channel_height.fix(5e-4)
    m.fs.nf.velocity[0, 0].fix(0.1)
    m.fs.nf.area.fix(200)
    m.fs.nf.mixed_permeate[0].pressure.fix(101325)

    # variables for calculating mass transfer coefficient with spiral wound correlation
    m.fs.nf.spacer_mixing_efficiency.fix()
    m.fs.nf.spacer_mixing_length.fix()

    # membrane properties
    m.fs.nf.radius_pore.fix(0.5e-9)
    m.fs.nf.membrane_thickness_effective.fix(1.33e-6)
    m.fs.nf.membrane_charge_density.fix(-60)
    m.fs.nf.dielectric_constant_pore.fix(41.3)

    iscale.calculate_scaling_factors(m)


def solve(blk, solver=None, show=True):
    """
    Optimizes the flowsheet

    Args:
        m: pyomo model
        solver: optimization solver
    """
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=show)
    if results.solver.termination_condition != "optimal":
        raise ValueError("The solver did not return optimal termination")
    return results


def initialize_system(m, solver=None):
    """
    Initializes the flowsheet units

    Args:
        m: pyomo model
        solver: optimization solver
    """

    if solver is None:
        solver = get_solver()

    # dt = DiagnosticsToolbox(m)
    # dt.report_structural_issues()
    # dt.display_potential_evaluation_errors()

    m.fs.feed.initialize(optarg=solver.options)
    propagate_state(m.fs.feed_to_pump)

    m.fs.pump.initialize(optarg=solver.options)
    propagate_state(m.fs.pump_to_nf)

    m.fs.nf.initialize(optarg=solver.options)
    propagate_state(m.fs.nf_to_permeate)
    propagate_state(m.fs.nf_to_retentate)

    m.fs.permeate.initialize(optarg=solver.options)
    m.fs.retentate.initialize(optarg=solver.options)

    print(f"DOF: {degrees_of_freedom(m)}")

    m.fs.costing.initialize()

def optimize_set_up(m, pressure_limit=7e6):
    """
    Unfixes select variables to enable optimization with DOF>0

    Args:
        m: pyomo model
    """
    # limit Li loss
    # m.fs.objective = Objective(
    #     expr=m.fs.retentate.flow_mol_phase_comp[0, "Liq", "Li_+"]
    # )
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)

    m.fs.pressure_constraint = Constraint(
        expr=m.fs.pump.outlet.pressure[0] <= pressure_limit
    )

    m.fs.pump.outlet.pressure[0].unfix()
    m.fs.nf.area.unfix()



if __name__ == "__main__":
    main()
