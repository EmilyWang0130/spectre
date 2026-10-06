# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def velocity(x, adiabatic_index, density, pressure, electron_fraction, shear_velocity, shear_thickness, perturbation_amplitude, perturbation_wavelength, perturbation_width, magnetic_field):
    result = np.zeros(x.size)
    result[0] = -shear_velocity * np.tanh(x[1] / shear_thickness)
    result[1] = (
        perturbation_amplitude
        * np.sin(2.0 * np.pi * x[0] / perturbation_wavelength)
        * np.exp(-((x[1] / perturbation_width) ** 2))
    )
    return result


def rest_mass_density(x, adiabatic_index, density, pressure, electron_fraction, shear_velocity, shear_thickness, perturbation_amplitude, perturbation_wavelength, perturbation_width, magnetic_field):
    return density


def electron_fraction(x, adiabatic_index, density, pressure, electron_fraction, shear_velocity, shear_thickness, perturbation_amplitude, perturbation_wavelength, perturbation_width, magnetic_field):
    return electron_fraction


def specific_internal_energy(x, adiabatic_index, density, pressure, electron_fraction, shear_velocity, shear_thickness, perturbation_amplitude, perturbation_wavelength, perturbation_width, magnetic_field):
    return pressure / (adiabatic_index - 1.0) / density


def pressure(x, adiabatic_index, density, pressure, electron_fraction, shear_velocity, shear_thickness, perturbation_amplitude, perturbation_wavelength, perturbation_width, magnetic_field):
    return pressure


def specific_enthalpy(x, adiabatic_index, density, pressure, electron_fraction, shear_velocity, shear_thickness, perturbation_amplitude, perturbation_wavelength, perturbation_width, magnetic_field):
    return 1.0 + adiabatic_index * pressure / (adiabatic_index - 1.0) / density


def lorentz_factor(x, adiabatic_index, density, pressure, electron_fraction, shear_velocity, shear_thickness, perturbation_amplitude, perturbation_wavelength, perturbation_width, magnetic_field):
    v = velocity(x, adiabatic_index, density, pressure, electron_fraction, shear_velocity, shear_thickness, perturbation_amplitude, perturbation_wavelength, perturbation_width, magnetic_field)
    return 1.0 / np.sqrt(1.0 - np.dot(v, v))


def magnetic_field(x, adiabatic_index, density, pressure, electron_fraction, shear_velocity, shear_thickness, perturbation_amplitude, perturbation_wavelength, perturbation_width, magnetic_field):
    return np.array(magnetic_field)


def divergence_cleaning_field(x, adiabatic_index, density, pressure, electron_fraction, shear_velocity, shear_thickness, perturbation_amplitude, perturbation_wavelength, perturbation_width, magnetic_field):
    return 0.0
