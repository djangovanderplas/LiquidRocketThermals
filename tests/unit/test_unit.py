import numpy as np
from matplotlib import pyplot as plt

from PyRocket.IsentropicRelations import Isentropic
from PyRocket.Output import Settings2D
from PyRocket.PlottingFunctions import multi_plot
from PyRocket.SectionThermalSim import HeatEquationSolver
from PyRocket.runtime import build_runtime_context
from PyRocket.coolant import CoolPropMixture


def _context():
    return build_runtime_context()


def cooling_fluid_test():
    context = _context()
    N = 100                                 # number of steps
    Cp = np.ndarray(N)                      # specific heat at constant pressure [J/kg/K]
    k = np.ndarray(N)                       # thermal conductivity [W/m/K]
    rho = np.ndarray(N)                     # density [kg/m^3]
    Pr = np.ndarray(N)                      # Prandtl number

    if context.coolant.phase == 'l':
        T_arr = np.linspace(288, 600, N)      # Temperature array [K]
    elif context.coolant.phase == 'g':
        T_arr = np.linspace(288, 1200, N)
    else:
        raise ValueError("Coolant phase must be 'l' or 'g'")

    for i in range(len(T_arr)):
        # calculate coolant properties at every temperature
        coolant = CoolPropMixture(IDs=context.coolant.IDs, ws=context.coolant.ws, T=T_arr[i], P=context.coolant.P)
        Cp[i] = coolant.Cp
        rho[i] = coolant.rho
        Pr[i] = coolant.Pr
        k[i] = coolant.Cp * coolant.mu / coolant.Pr

    f, axes = plt.subplots(4, 1)

    axes[0].plot(T_arr, Cp)
    axes[0].set_ylabel('Cp [J/kg/K]')

    axes[1].plot(T_arr, k)
    axes[1].set_ylabel('k [W/m/K]')

    axes[2].plot(T_arr, rho)
    axes[2].set_ylabel('rho [kg/m^3]')

    axes[3].plot(T_arr, Pr)
    axes[3].set_ylabel('Pr')

    plt.xlabel('Temperature [K]')
    plt.show()


def material_property_test():
    context = _context()
    try:
        context.material.plot(context.material.sig_u, 'ultimate strength [Pa]')
    except Exception:
        print('Warning: no variable ultimate strength of material set')
    try:
        context.material.plot(context.material.E, 'youngs modulus [Pa]')
    except Exception:
        print('Warning: no variable youngs modulus of material set')
    try:
        context.material.plot(context.material.k, 'thermal conductivity [W/m/K]')
    except Exception:
        print('Warning: no variable thremal conductivity of material set')


def isentropic_relations_test():
    context = _context()
    gas = Isentropic(context.cea.Pc, context.cea.Tc, context.cea.gamma, context.cea.Pr, context.geometry[:, 0], context.geometry[:, 1])
    gas.calculate()

    multi_plot(context.geometry[:, 0], gas.M, gas.T_s, gas.p_s/1e5, gas.T_aw, 'M', 'T_s [K]', 'p_s [bar]', 'T_aw [K]')


def section_thermal_sim_test():
    context = _context()

    def halpha_func(T, idx):
        return 1600, 2400

    def halpha_c_func(T, idx):
        return 1113

    q_rad = 0
    T_c = 288
    T_amb = 288
    idx = 1
    settings = Settings2D(cell_size=2e-4, time_step=4, tolerance=1e-3, max_iter=100, save_fig=False, print_result=False, run_time='steady_state')
    settings.output_msg()
    test_output_path = context.repo_root / "outputs" / "TestSectionThermalSim"
    test_output_path.mkdir(parents=True, exist_ok=True)

    solver = HeatEquationSolver(
        idx,
        context.gas,
        context.material,
        context.cooling_geom,
        halpha_func,
        halpha_c_func,
        q_rad,
        T_c,
        T_amb,
        path=test_output_path,
        settings=settings
    )

    solver.run_sim()

    if solver.diff < 1e-3:
        print('Solver converged sucessfully')


if __name__ == "__main__":
    cooling_fluid_test()
    material_property_test()
    isentropic_relations_test()
    section_thermal_sim_test()
