from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

import numpy as np
import thermo

from .CEAClass import BipropCEA
from .FilmCooling import FilmCooling
from .GeomClass import ChamberGeometry, CoolingGeometry
from .IsentropicRelations import Isentropic
from .Output import Output1D, Settings2D
from . import MaterialLib as matlib
from . import PropLibrary as proplib


@dataclass
class SimulationContext:
    repo_root: Path
    config_path: Path
    output_dir: Path
    contour_csv: Path
    chamber: ChamberGeometry
    geometry: np.ndarray
    cooling_geom: CoolingGeometry
    settings: Settings2D
    output: Output1D
    cea: BipropCEA
    gas: Isentropic
    coolant: Any
    film: FilmCooling
    material: Any
    hot_gas_method: str
    cooling_method: str
    ambient_temp: float
    eta_c_star: float
    m_dot: float
    m_dot_coolant: float


def _resolve_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _resolve_material(material_name: str):
    if not hasattr(matlib, material_name):
        raise ValueError(f"Unknown material '{material_name}' in config.toml")
    return getattr(matlib, material_name)


def _resolve_propellant_name(name: str):
    # Use PropLibrary symbols if present, else pass through RocketCEA-native names.
    return getattr(proplib, name, name)


def _load_toml(config_path: Path) -> dict[str, Any]:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def build_runtime_context(config_path: str | Path | None = None) -> SimulationContext:
    module_dir = Path(__file__).resolve().parent
    repo_root = module_dir.parent
    resolved_config_path = _resolve_path(str(config_path), repo_root) if config_path else repo_root / "config.toml"
    cfg = _load_toml(resolved_config_path)

    paths_cfg = cfg["paths"]
    operating_cfg = cfg["operating"]
    coolant_cfg = cfg["coolant"]
    film_cfg = cfg["film_cooling"]
    chamber_cfg = cfg["chamber_geometry"]
    cooling_cfg = cfg["cooling_geometry"]
    solver_cfg = cfg["solver"]
    thermocouple_cfg = cfg["thermocouples"]

    contour_csv = _resolve_path(paths_cfg["contour_csv"], repo_root)
    output_dir = _resolve_path(paths_cfg["output_dir"], repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    material = _resolve_material(chamber_cfg["material"])
    fuel = _resolve_propellant_name(operating_cfg["fuel"])
    oxidizer = _resolve_propellant_name(operating_cfg["oxidizer"])

    m_dot = float(operating_cfg["m_dot"])
    mixture_ratio = float(operating_cfg["mixture_ratio"])
    m_dot_f = m_dot / (mixture_ratio + 1.0)
    m_dot_coolant = m_dot_f * float(coolant_cfg["m_dot_coolant_ratio_to_fuel"])
    m_dot_film = m_dot_coolant * float(film_cfg["m_dot_ratio"])

    chamber = ChamberGeometry(
        chamber_cfg["D_c"],
        chamber_cfg["D_t"],
        chamber_cfg["D_e"],
        chamber_cfg["L_cyl"],
        chamber_cfg["r_1"],
        chamber_cfg["r_2"],
        chamber_cfg["r_n"],
        chamber_cfg["phi_conv"],
        chamber_cfg["phi_div"],
        chamber_cfg["phi_e"],
        chamber_cfg["step_size"],
        contour_path=contour_csv,
    )
    chamber.contour()
    chamber.plot_contour(output_dir)
    geometry = chamber.geometry

    cooling_geom = CoolingGeometry(
        geometry,
        cooling_cfg["h_c"],
        cooling_cfg["psi"],
        cooling_cfg["t_w_i"],
        cooling_cfg["t_w_o"],
        cooling_cfg["n_channels"],
    )
    cooling_geom.channel_geometry()
    cooling_geom.set_thermocouples(thermocouple_cfg["x"], thermocouple_cfg["r"])

    cell_size = solver_cfg.get("cell_size", solver_cfg["cell_size_factor"] * cooling_cfg["t_w_i"])
    time_step = solver_cfg.get("time_step", (cell_size ** 2) / material.alpha)
    settings = Settings2D(
        cell_size,
        time_step,
        solver_cfg["tolerance"],
        solver_cfg["max_iter"],
        solver_cfg["save_fig"],
        solver_cfg["print_result"],
        solver_cfg["run_time"],
        cooling_cfg["start_idx"],
        solver_cfg["adaptive_up"],
        thermocouple_cfg["log"],
        cooling_geom.thermocouples,
    )

    output = Output1D(geometry, output_dir)

    cea = BipropCEA(fuel, oxidizer, operating_cfg["Pc"])
    cea.metric_cea_output("throat", mixture_ratio, chamber.expansion_ratio)

    gas = Isentropic(operating_cfg["Pc"], cea.Tc, cea.gamma, cea.Pr, geometry[:, 0], geometry[:, 1])
    gas.calculate()

    coolant = thermo.Mixture(
        coolant_cfg["fluid_ids"],
        ws=coolant_cfg["mass_fractions"],
        P=coolant_cfg["inlet_pressure"],
        T=coolant_cfg["inlet_temp"],
    )

    film = FilmCooling(
        cea,
        coolant,
        film_cfg["injection_velocity"],
        film_cfg["injector_diameter"],
        chamber_cfg["D_c"],
        m_dot - m_dot_film,
        m_dot_film,
    )
    film.film_length()
    film.cooled_idx = np.where(geometry[:, 0] <= film.liquid_film_length)[0]

    return SimulationContext(
        repo_root=repo_root,
        config_path=resolved_config_path,
        output_dir=output_dir,
        contour_csv=contour_csv,
        chamber=chamber,
        geometry=geometry,
        cooling_geom=cooling_geom,
        settings=settings,
        output=output,
        cea=cea,
        gas=gas,
        coolant=coolant,
        film=film,
        material=material,
        hot_gas_method=operating_cfg["hot_gas_method"],
        cooling_method=operating_cfg["cooling_method"],
        ambient_temp=operating_cfg["ambient_temp"],
        eta_c_star=operating_cfg["eta_c_star"],
        m_dot=m_dot,
        m_dot_coolant=m_dot_coolant,
    )
