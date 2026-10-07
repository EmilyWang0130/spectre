# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def _is_left(x, interface_angle):
    assert len(x) == 3
    return x[0] * np.cos(interface_angle) + x[1] * np.sin(interface_angle) <= 0.0


def _to_grid_frame(v, interface_angle):
    c, s = np.cos(interface_angle), np.sin(interface_angle)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2]])


def rest_mass_density(
    x,
    adiabatic_index,
    left_density,
    right_density,
    left_pressure,
    right_pressure,
    left_velocity,
    right_velocity,
    left_magnetic_field,
    right_magnetic_field,
    lapse,
    shift,
    interface_angle,
):
    return left_density if _is_left(x, interface_angle) else right_density


def spatial_velocity(
    x,
    adiabatic_index,
    left_density,
    right_density,
    left_pressure,
    right_pressure,
    left_velocity,
    right_velocity,
    left_magnetic_field,
    right_magnetic_field,
    lapse,
    shift,
    interface_angle,
):
    return _to_grid_frame(
        left_velocity if _is_left(x, interface_angle) else right_velocity,
        interface_angle,
    )


def compute_pressure(
    x,
    adiabatic_index,
    left_density,
    right_density,
    left_pressure,
    right_pressure,
    left_velocity,
    right_velocity,
    left_magnetic_field,
    right_magnetic_field,
    lapse,
    shift,
    interface_angle,
):
    return left_pressure if _is_left(x, interface_angle) else right_pressure


def specific_internal_energy(x, *args):
    adiabatic_index = args[0]
    return (
        1.0
        / (adiabatic_index - 1.0)
        * compute_pressure(x, *args)
        / rest_mass_density(x, *args)
    )


def lorentz_factor(x, *args):
    v = spatial_velocity(x, *args)
    return 1.0 / np.sqrt(1.0 - np.dot(v, v))


def specific_enthalpy(x, *args):
    adiabatic_index = args[0]
    return 1.0 + adiabatic_index * specific_internal_energy(x, *args)


def magnetic_field(
    x,
    adiabatic_index,
    left_density,
    right_density,
    left_pressure,
    right_pressure,
    left_velocity,
    right_velocity,
    left_magnetic_field,
    right_magnetic_field,
    lapse,
    shift,
    interface_angle,
):
    return _to_grid_frame(
        left_magnetic_field
        if _is_left(x, interface_angle)
        else right_magnetic_field,
        interface_angle,
    )


def divergence_cleaning_field(x, *args):
    return 0.0
