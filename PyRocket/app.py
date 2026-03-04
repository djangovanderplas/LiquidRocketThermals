from pathlib import Path

from .PlottingFunctions import Plotting1D
from .RegenCooling import HeatTransfer
from .runtime import build_runtime_context


def run_simulation(config_path: str | Path | None = None, save_plots: bool = True, show_plots: bool = False):
    context = build_runtime_context(config_path)

    # give terminal message about simulation settings
    context.settings.output_msg()
    context.output.output_msg()

    # heat transfer sim
    sim = HeatTransfer(
        cea=context.cea,
        gas=context.gas,
        geometry=context.geometry,
        material=context.material,
        coolant=context.coolant,
        cooling_geometry=context.cooling_geom,
        m_dot=context.m_dot,
        m_dot_coolant=context.m_dot_coolant,
        T_amb=context.ambient_temp,
        output=context.output,
        settings2D=context.settings,
        model=context.hot_gas_method,
        cool_model=context.cooling_method,
        eta_c_star=context.eta_c_star,
        film=context.film,
    )

    sim.run()

    # plotting
    plot = Plotting1D(save_path=context.output_dir, geometry=context.geometry, save=save_plots, show=show_plots)
    plot.temperature_plot()
    plot.pressure_plot()
    plot.heat_transfer_coeff_plot()
    plot.heat_flux_plot()
    plot.reynolds_plot()

    return context
