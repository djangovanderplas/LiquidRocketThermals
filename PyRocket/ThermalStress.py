import numpy as np


class ThermalStress:
    """Phase A stress-index post processor.

    Computes a thermal + pressure stress estimate at sampled boundary temperatures.
    This is an engineering screening tool, not a full elasticity solution.
    """

    VALID_MODELS = {
        "hoop_restrained_axial_free",
        "biaxial_restrained",
    }

    def __init__(self, material, constraint_model="hoop_restrained_axial_free", include_global_restraint=True, T_ref_global=288.15):
        if constraint_model not in self.VALID_MODELS:
            raise ValueError(
                f"Invalid stress constraint model '{constraint_model}'. "
                f"Use one of: {sorted(self.VALID_MODELS)}"
            )

        self.material = material
        self.constraint_model = constraint_model
        self.include_global_restraint = include_global_restraint
        self.T_ref_global = T_ref_global

    @staticmethod
    def _material_value(prop, T):
        return prop(T) if callable(prop) else float(prop)

    def _thermal_components(self, T, T_section_mean):
        E = self._material_value(self.material.E, T)
        alpha = self._material_value(self.material.a, T)
        nu = self._material_value(self.material.v, T)

        denom = max(1.0e-6, 1.0 - nu)

        dT_self = T - T_section_mean
        dT_global = T_section_mean - self.T_ref_global if self.include_global_restraint else 0.0
        dT = dT_self + dT_global

        sigma_theta_th = -E * alpha * dT / denom

        if self.constraint_model == "hoop_restrained_axial_free":
            sigma_z_th = 0.0
        elif self.constraint_model == "biaxial_restrained":
            sigma_z_th = sigma_theta_th
        else:
            raise ValueError("Unsupported stress model")

        return sigma_theta_th, sigma_z_th

    @staticmethod
    def _pressure_components(p_i, p_o, r_i, t):
        delta_p = p_i - p_o
        t_eff = max(1.0e-9, t)
        sigma_theta_p = delta_p * r_i / t_eff
        sigma_z_p = delta_p * r_i / (2.0 * t_eff)
        return sigma_theta_p, sigma_z_p

    def evaluate_section(self, boundary_temperatures, T_section_mean, p_i, p_o, r_i, t):
        sigma_theta_p, sigma_z_p = self._pressure_components(p_i, p_o, r_i, t)

        sigma_theta = np.zeros_like(boundary_temperatures, dtype=float)
        sigma_z = np.zeros_like(boundary_temperatures, dtype=float)
        sigma_vm = np.zeros_like(boundary_temperatures, dtype=float)

        for i, T in enumerate(boundary_temperatures):
            sigma_theta_th, sigma_z_th = self._thermal_components(float(T), float(T_section_mean))
            sigma_theta[i] = sigma_theta_th + sigma_theta_p
            sigma_z[i] = sigma_z_th + sigma_z_p
            sigma_vm[i] = np.sqrt(sigma_theta[i] ** 2 + sigma_z[i] ** 2 - sigma_theta[i] * sigma_z[i])

        idx_max = int(np.argmax(sigma_vm))
        sigma_vm_max = float(sigma_vm[idx_max])

        # single-point estimate at hottest chamber wall face for reporting
        T_hot = float(np.max(boundary_temperatures))
        sigma_theta_th_hot, sigma_z_th_hot = self._thermal_components(T_hot, float(T_section_mean))
        sigma_theta_hot = float(sigma_theta_th_hot + sigma_theta_p)
        sigma_z_hot = float(sigma_z_th_hot + sigma_z_p)

        sig_u = float(self.material.sig_u)
        fos_u = np.inf if sigma_vm_max <= 0.0 else sig_u / sigma_vm_max

        return {
            "sigma_theta_hot": sigma_theta_hot,
            "sigma_z_hot": sigma_z_hot,
            "sigma_vm_max": sigma_vm_max,
            "fos_u": float(fos_u),
            "sigma_theta_samples": sigma_theta,
            "sigma_z_samples": sigma_z,
            "sigma_vm_samples": sigma_vm,
        }
