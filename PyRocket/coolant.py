from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from CoolProp.CoolProp import PhaseSI, PropsSI


_FLUID_ALIASES = {
    "C2H5OH": "Ethanol",
    "H2O": "Water",
    "N2": "Nitrogen",
    "O2": "Oxygen",
    "CH4": "Methane",
    "C3H8": "Propane",
}


@dataclass
class CoolPropMixture:
    IDs: list[str]
    ws: list[float]
    T: float
    P: float

    def __post_init__(self):
        self.IDs = list(self.IDs)
        self.ws = self._normalize(self.ws)
        self._cp_ids = [self._map_id(fluid) for fluid in self.IDs]
        self._molar_masses = [PropsSI("M", fluid) for fluid in self._cp_ids]
        self.zs = self._mass_to_mole_fractions(self.ws, self._molar_masses)
        self._backend = self._build_backend(self._cp_ids, self.zs)

        self.rho = PropsSI("Dmass", "T", self.T, "P", self.P, self._backend)
        self.Cp = PropsSI("Cpmass", "T", self.T, "P", self.P, self._backend)
        self.mu = PropsSI("VISCOSITY", "T", self.T, "P", self.P, self._backend)
        self.k = PropsSI("CONDUCTIVITY", "T", self.T, "P", self.P, self._backend)
        self.Pr = self.Cp * self.mu / self.k

        self.phase = self._phase_symbol()
        self.Prl = self.Pr
        self.Prg = self.Pr
        self.Cpl = self.Cp
        self.Cpg = self.Cp
        self.mul = self.mu
        self.mug = self.mu

        self.Tbs = [self._safe_boiling_point(fluid) for fluid in self._cp_ids]
        self.Hvap_Tbs = [self._safe_hvap_at_tb(fluid, tb) for fluid, tb in zip(self._cp_ids, self.Tbs)]
        self.sigma = self._surface_tension(self._cp_ids, self.ws)
        self.MW = 1.0 / np.sum(np.asarray(self.ws) / np.asarray(self._molar_masses))

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        arr = np.asarray(values, dtype=float)
        total = arr.sum()
        if total <= 0:
            raise ValueError("Mass fractions must sum to a positive number")
        return (arr / total).tolist()

    @staticmethod
    def _map_id(fluid_id: str) -> str:
        return _FLUID_ALIASES.get(fluid_id, fluid_id)

    @staticmethod
    def _mass_to_mole_fractions(ws: list[float], molar_masses: list[float]) -> list[float]:
        w = np.asarray(ws)
        m = np.asarray(molar_masses)
        mol = w / m
        return (mol / mol.sum()).tolist()

    @staticmethod
    def _build_backend(cp_ids: list[str], zs: list[float]) -> str:
        if len(cp_ids) == 1:
            return cp_ids[0]
        terms = [f"{name}[{z}]" for name, z in zip(cp_ids, zs)]
        return "HEOS::" + "&".join(terms)

    def _phase_symbol(self) -> str:
        phase_name = PhaseSI("T", self.T, "P", self.P, self._backend)
        if "liquid" in phase_name:
            return "l"
        if "gas" in phase_name:
            return "g"
        if "supercritical" in phase_name:
            return "g"
        raise ValueError(f"Unsupported coolant phase '{phase_name}'")

    @staticmethod
    def _safe_boiling_point(fluid: str) -> float:
        try:
            return PropsSI("T", "P", 101325.0, "Q", 0, fluid)
        except Exception:
            return np.nan

    @staticmethod
    def _safe_hvap_at_tb(fluid: str, tb: float) -> float:
        if np.isnan(tb):
            return 0.0
        h_v = PropsSI("Hmass", "P", 101325.0, "Q", 1, fluid)
        h_l = PropsSI("Hmass", "P", 101325.0, "Q", 0, fluid)
        return max(h_v - h_l, 0.0)

    def _surface_tension(self, cp_ids: list[str], ws: list[float]) -> float:
        sigmas = []
        for fluid in cp_ids:
            try:
                tb = self._safe_boiling_point(fluid)
                t_eval = min(self.T, tb * 0.99) if not np.isnan(tb) else self.T
                sigma = PropsSI("SURFACE_TENSION", "T", t_eval, "Q", 0, fluid)
            except Exception:
                sigma = np.nan
            sigmas.append(sigma)

        sigmas = np.asarray(sigmas, dtype=float)
        valid = np.isfinite(sigmas)
        if not np.any(valid):
            return 0.02
        return float(np.sum(sigmas[valid] * np.asarray(ws)[valid]))
