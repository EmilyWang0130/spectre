# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def _sign(y):
    return -1.0 if y < 0.0 else 1.0


def velocity(
    x,
    adiabatic_index,
    shear_velocity,
    strip_half_width,
    transition_thickness,
    upper_density,
    lower_density,
    upper_ye,
    lower_ye,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    dim = x.size
    result = np.zeros(dim)
    sgn = _sign(x[1])
    layer = x[1] - sgn * strip_half_width
    result[0] = sgn * shear_velocity * np.tanh(layer / transition_thickness)
    result[1] = (
        sgn
        * perturbation_amplitude
        * shear_velocity
        * np.sin(2.0 * np.pi * x[0])
        * np.exp(-((layer / perturbation_width) ** 2))
    )
    return result


def rest_mass_density(
    x,
    adiabatic_index,
    shear_velocity,
    strip_half_width,
    transition_thickness,
    upper_density,
    lower_density,
    upper_ye,
    lower_ye,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    sgn = _sign(x[1])
    layer = x[1] - sgn * strip_half_width
    v_x = sgn * shear_velocity * np.tanh(layer / transition_thickness)
    return (
        0.5 * (upper_density + lower_density)
        + 0.5 * (upper_density - lower_density) * v_x / shear_velocity
    )


def electron_fraction(
    x,
    adiabatic_index,
    shear_velocity,
    strip_half_width,
    transition_thickness,
    upper_density,
    lower_density,
    upper_ye,
    lower_ye,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    sgn = _sign(x[1])
    layer = x[1] - sgn * strip_half_width
    v_x = sgn * shear_velocity * np.tanh(layer / transition_thickness)
    return (
        0.5 * (upper_ye + lower_ye)
        + 0.5 * (upper_ye - lower_ye) * v_x / shear_velocity
    )


def specific_internal_energy(
    x,
    adiabatic_index,
    shear_velocity,
    strip_half_width,
    transition_thickness,
    upper_density,
    lower_density,
    upper_ye,
    lower_ye,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    rho = rest_mass_density(
        x,
        adiabatic_index,
        shear_velocity,
        strip_half_width,
        transition_thickness,
        upper_density,
        lower_density,
        upper_ye,
        lower_ye,
        pressure,
        perturbation_amplitude,
        perturbation_width,
        magnetic_field,
    )
    return pressure / (adiabatic_index - 1.0) / rho


def pressure(
    x,
    adiabatic_index,
    shear_velocity,
    strip_half_width,
    transition_thickness,
    upper_density,
    lower_density,
    upper_ye,
    lower_ye,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return pressure


def specific_enthalpy(
    x,
    adiabatic_index,
    shear_velocity,
    strip_half_width,
    transition_thickness,
    upper_density,
    lower_density,
    upper_ye,
    lower_ye,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return 1.0 + adiabatic_index * specific_internal_energy(
        x,
        adiabatic_index,
        shear_velocity,
        strip_half_width,
        transition_thickness,
        upper_density,
        lower_density,
        upper_ye,
        lower_ye,
        pressure,
        perturbation_amplitude,
        perturbation_width,
        magnetic_field,
    )


def lorentz_factor(
    x,
    adiabatic_index,
    shear_velocity,
    strip_half_width,
    transition_thickness,
    upper_density,
    lower_density,
    upper_ye,
    lower_ye,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    v = velocity(
        x,
        adiabatic_index,
        shear_velocity,
        strip_half_width,
        transition_thickness,
        upper_density,
        lower_density,
        upper_ye,
        lower_ye,
        pressure,
        perturbation_amplitude,
        perturbation_width,
        magnetic_field,
    )
    return 1.0 / np.sqrt(1.0 - np.dot(v, v))


def magnetic_field(
    x,
    adiabatic_index,
    shear_velocity,
    strip_half_width,
    transition_thickness,
    upper_density,
    lower_density,
    upper_ye,
    lower_ye,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return np.array(magnetic_field)


def divergence_cleaning_field(
    x,
    adiabatic_index,
    shear_velocity,
    strip_half_width,
    transition_thickness,
    upper_density,
    lower_density,
    upper_ye,
    lower_ye,
    pressure,
    perturbation_amplitude,
    perturbation_width,
    magnetic_field,
):
    return 0.0
