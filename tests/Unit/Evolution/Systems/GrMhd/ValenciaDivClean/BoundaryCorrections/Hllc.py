# Distributed under the MIT License.
# See LICENSE.txt for details.

# Python reference implementation of the HLLC boundary correction. Mirrors
# `Hll.py`'s argument-order convention (state, flux, temporary, primitive
# with SpatialVelocity at the end); the test framework passes values
# positionally.

import numpy as np

_atmosphere_density_cutoff = 1.0e-8
_magnetic_field_magnitude_for_hydro = 1.0e-30
_polytropic_constant = 100.0
_polytropic_gamma = 2.0


def _hll_char_speed(
    lapse,
    shift,
    normal_covector,
    normal_dot_mesh_velocity,
    lapse_sign,
    tilde_b,
    rest_mass_density,
    spatial_velocity,
    spatial_velocity_one_form,
):
    """Reproduce Hll.py's char-speed logic."""
    magnetic_field_magnitude = np.sqrt(np.einsum("i,i->", tilde_b, tilde_b))
    if (
        magnetic_field_magnitude < _magnetic_field_magnitude_for_hydro
        and rest_mass_density > _atmosphere_density_cutoff
    ):
        pressure = _polytropic_constant * rest_mass_density**_polytropic_gamma
        sound_speed_squared = min(
            max(
                (
                    _polytropic_gamma
                    * (_polytropic_gamma - 1.0)
                    * pressure
                    / (
                        rest_mass_density * (_polytropic_gamma - 1.0)
                        + _polytropic_gamma * pressure
                    )
                ),
                0.0,
            ),
            1.0,
        )
        # No clamp on `velocity_dot_normal`: the C++ implementation only
        # clamps `velocity_squared`, so we mirror that here.
        velocity_dot_normal = np.einsum(
            "i,i->", spatial_velocity, normal_covector
        )
        velocity_squared = min(
            max(
                np.einsum("i,i->", spatial_velocity, spatial_velocity_one_form),
                0.0,
            ),
            1.0 - 1.0e-8,
        )
        one_over_lorentz_factor_squared = 1.0 - velocity_squared
        d = np.sqrt(
            max(
                0.0,
                sound_speed_squared
                * one_over_lorentz_factor_squared
                * (
                    1.0
                    - velocity_squared * sound_speed_squared
                    - velocity_dot_normal**2 * (1.0 - sound_speed_squared)
                ),
            )
        )
        Lambda = (
            lapse
            / (1.0 - velocity_squared * sound_speed_squared)
            * (
                velocity_dot_normal * (1.0 - sound_speed_squared)
                + lapse_sign * d
            )
        )
        return np.asarray(
            (Lambda - np.dot(shift, normal_covector))
            if normal_dot_mesh_velocity is None
            else (
                Lambda
                - np.dot(shift, normal_covector)
                - normal_dot_mesh_velocity
            )
        )
    return np.asarray(
        (lapse_sign * lapse - np.dot(shift, normal_covector))
        if normal_dot_mesh_velocity is None
        else (
            lapse_sign * lapse
            - np.dot(shift, normal_covector)
            - normal_dot_mesh_velocity
        )
    )


def dg_package_data(
    tilde_d,
    tilde_ye,
    tilde_tau,
    tilde_s,
    tilde_b,
    tilde_phi,
    flux_tilde_d,
    flux_tilde_ye,
    flux_tilde_tau,
    flux_tilde_s,
    flux_tilde_b,
    flux_tilde_phi,
    lapse,
    shift,
    spatial_velocity_one_form,
    spatial_metric,
    rest_mass_density,
    electron_fraction,
    temperature,
    spatial_velocity,
    specific_internal_energy,
    pressure,
    lorentz_factor,
    normal_covector,
    normal_vector,
    mesh_velocity,
    normal_dot_mesh_velocity,
    equation_of_state,
):
    # Char speeds: reuse the HLL logic verbatim.
    largest_outgoing_char_speed = _hll_char_speed(
        lapse,
        shift,
        normal_covector,
        normal_dot_mesh_velocity,
        1.0,
        tilde_b,
        rest_mass_density,
        spatial_velocity,
        spatial_velocity_one_form,
    )
    largest_ingoing_char_speed = _hll_char_speed(
        lapse,
        shift,
        normal_covector,
        normal_dot_mesh_velocity,
        -1.0,
        tilde_b,
        rest_mass_density,
        spatial_velocity,
        spatial_velocity_one_form,
    )

    # HLLC extras.
    # Advection speed u = alpha v^n - beta^n (with mesh-velocity correction).
    v_dot_n = np.einsum("i,i->", spatial_velocity, normal_covector)
    beta_dot_n = np.dot(shift, normal_covector)
    advection_speed = np.asarray(
        (lapse * v_dot_n - beta_dot_n)
        if normal_dot_mesh_velocity is None
        else (lapse * v_dot_n - beta_dot_n - normal_dot_mesh_velocity)
    )
    # Pressure-flux coefficient P_tilde = alpha sqrt(gamma) p; compute
    # sqrt(det(gamma)) from the metric.
    sqrt_det_gamma = np.sqrt(np.linalg.det(spatial_metric))
    pressure_flux_coefficient = np.asarray(lapse * sqrt_det_gamma * pressure)
    # Mirror the C++: the FD subcell path doesn't populate the passed-in
    # `normal_vector`, so we ignore it and raise `normal_covector` locally
    # using inverse(spatial_metric).
    del normal_vector  # unused; positional-arg placeholder for pypp
    local_inv_spatial_metric = np.linalg.inv(spatial_metric)
    local_normal_vector = np.einsum(
        "ij,j->i", local_inv_spatial_metric, normal_covector
    )
    # M = n^i tilde_S_i.
    normal_dot_tilde_s = np.asarray(np.dot(local_normal_vector, tilde_s))
    # F(M) = n^i n_j F^j(tilde_S_i).
    normal_dot_flux_tilde_s = np.einsum(
        "ij,i->j", flux_tilde_s, normal_covector
    )
    normal_dot_flux_normal_dot_tilde_s = np.asarray(
        np.dot(local_normal_vector, normal_dot_flux_tilde_s)
    )

    return (
        tilde_d,
        tilde_ye,
        tilde_tau,
        tilde_s,
        tilde_b,
        tilde_phi,
        np.asarray(np.dot(flux_tilde_d, normal_covector)),
        np.asarray(np.dot(flux_tilde_ye, normal_covector)),
        np.asarray(np.dot(flux_tilde_tau, normal_covector)),
        normal_dot_flux_tilde_s,
        np.einsum("ij,i->j", flux_tilde_b, normal_covector),
        np.asarray(np.dot(flux_tilde_phi, normal_covector)),
        largest_outgoing_char_speed,
        largest_ingoing_char_speed,
        normal_dot_tilde_s,
        normal_dot_flux_normal_dot_tilde_s,
        advection_speed,
        pressure_flux_coefficient,
        normal_covector,
    )


def _hll_raw_G(lambda_R, lambda_L, f_L, f_R, u_L, u_R):
    """Raw HLL numerical flux G in the interior-normal frame.

    All arguments are already in the interior's convention: `f_R` and `u_R`
    have had their exterior-normal signs flipped where appropriate before
    entering this function. Caller applies any strong/weak-form subtraction
    of `f_L` afterwards.
    """
    return (
        lambda_R * f_L - lambda_L * f_R + lambda_L * lambda_R * (u_R - u_L)
    ) / (lambda_R - lambda_L)


def dg_boundary_terms(
    interior_tilde_d,
    interior_tilde_ye,
    interior_tilde_tau,
    interior_tilde_s,
    interior_tilde_b,
    interior_tilde_phi,
    interior_normal_dot_flux_tilde_d,
    interior_normal_dot_flux_tilde_ye,
    interior_normal_dot_flux_tilde_tau,
    interior_normal_dot_flux_tilde_s,
    interior_normal_dot_flux_tilde_b,
    interior_normal_dot_flux_tilde_phi,
    interior_largest_outgoing_char_speed,
    interior_largest_ingoing_char_speed,
    interior_normal_dot_tilde_s,
    interior_normal_dot_flux_normal_dot_tilde_s,
    interior_advection_speed,
    interior_pressure_flux_coefficient,
    interior_interface_normal_covector,
    exterior_tilde_d,
    exterior_tilde_ye,
    exterior_tilde_tau,
    exterior_tilde_s,
    exterior_tilde_b,
    exterior_tilde_phi,
    exterior_normal_dot_flux_tilde_d,
    exterior_normal_dot_flux_tilde_ye,
    exterior_normal_dot_flux_tilde_tau,
    exterior_normal_dot_flux_tilde_s,
    exterior_normal_dot_flux_tilde_b,
    exterior_normal_dot_flux_tilde_phi,
    exterior_largest_outgoing_char_speed,
    exterior_largest_ingoing_char_speed,
    exterior_normal_dot_tilde_s,
    exterior_normal_dot_flux_normal_dot_tilde_s,
    exterior_advection_speed,
    exterior_pressure_flux_coefficient,
    exterior_interface_normal_covector,
    use_strong_form,
):
    # Signal speeds in interior-normal frame.
    lambda_R = float(
        np.maximum(
            0.0,
            np.maximum(
                interior_largest_outgoing_char_speed,
                -exterior_largest_ingoing_char_speed,
            ),
        )
    )
    lambda_L = float(
        np.minimum(
            0.0,
            np.minimum(
                interior_largest_ingoing_char_speed,
                -exterior_largest_outgoing_char_speed,
            ),
        )
    )

    # Rename to L/R and apply sign flips: on the exterior side, flux-like
    # quantities and quantities linear in the outward normal flip sign
    # (because the exterior packaged them with its own outward normal, which
    # is opposite the interior's). Even-normal-count scalars do not flip.
    D_L = float(interior_tilde_d)
    Ye_L = float(interior_tilde_ye)
    Tau_L = float(interior_tilde_tau)
    Phi_L = float(interior_tilde_phi)
    E_L = Tau_L + D_L
    M_L = float(interior_normal_dot_tilde_s)
    u_L = float(interior_advection_speed)
    Ptilde_L = float(interior_pressure_flux_coefficient)
    fD_L = float(interior_normal_dot_flux_tilde_d)
    fYe_L = float(interior_normal_dot_flux_tilde_ye)
    fTau_L = float(interior_normal_dot_flux_tilde_tau)
    fPhi_L = float(interior_normal_dot_flux_tilde_phi)
    fM_L = float(interior_normal_dot_flux_normal_dot_tilde_s)
    fE_L = fTau_L + fD_L
    S_L = np.asarray(interior_tilde_s)
    B_L = np.asarray(interior_tilde_b)
    fS_L = np.asarray(interior_normal_dot_flux_tilde_s)
    fB_L = np.asarray(interior_normal_dot_flux_tilde_b)
    # Average the two sides' normals: matches the C++ Hllc's swap-invariant
    # convention so that the "flip sign under swap" test holds even when the
    # test helper perturbs the exterior metric.
    n_i = 0.5 * (
        np.asarray(interior_interface_normal_covector)
        - np.asarray(exterior_interface_normal_covector)
    )

    D_R = float(exterior_tilde_d)
    Ye_R = float(exterior_tilde_ye)
    Tau_R = float(exterior_tilde_tau)
    Phi_R = float(exterior_tilde_phi)
    E_R = Tau_R + D_R
    M_R = -float(exterior_normal_dot_tilde_s)
    u_R = -float(exterior_advection_speed)
    Ptilde_R = float(exterior_pressure_flux_coefficient)
    fD_R = -float(exterior_normal_dot_flux_tilde_d)
    fYe_R = -float(exterior_normal_dot_flux_tilde_ye)
    fTau_R = -float(exterior_normal_dot_flux_tilde_tau)
    fPhi_R = -float(exterior_normal_dot_flux_tilde_phi)
    fM_R = float(exterior_normal_dot_flux_normal_dot_tilde_s)
    fE_R = fTau_R + fD_R
    S_R = np.asarray(exterior_tilde_s)
    B_R = np.asarray(exterior_tilde_b)
    fS_R = -np.asarray(exterior_normal_dot_flux_tilde_s)
    fB_R = -np.asarray(exterior_normal_dot_flux_tilde_b)

    delta_lambda = lambda_R - lambda_L
    if lambda_L >= 0.0:
        G_D, G_Ye, G_Tau, G_Phi = fD_L, fYe_L, fTau_L, fPhi_L
        G_S = fS_L
        G_B = fB_L
    elif lambda_R <= 0.0:
        G_D, G_Ye, G_Tau, G_Phi = fD_R, fYe_R, fTau_R, fPhi_R
        G_S = fS_R
        G_B = fB_R
    else:
        E_hll = (lambda_R * E_R - lambda_L * E_L + fE_L - fE_R) / delta_lambda
        M_hll = (lambda_R * M_R - lambda_L * M_L + fM_L - fM_R) / delta_lambda
        fE_hll = (
            lambda_R * fE_L
            - lambda_L * fE_R
            + lambda_L * lambda_R * (E_R - E_L)
        ) / delta_lambda
        fM_hll = (
            lambda_R * fM_L
            - lambda_L * fM_R
            + lambda_L * lambda_R * (M_R - M_L)
        ) / delta_lambda

        # MB05 Eq. 18 quadratic. Numerical Recipes stable-root form (matches
        # PLUTO Src/RHD/hllc.c). No fallback -- MB05 guarantees a real root
        # inside [lambda_L, lambda_R] for physical L/R states.
        a = fE_hll
        b = E_hll + fM_hll
        c = M_hll
        # Defensive sqrt guard for the random-data unit test only; physical
        # states satisfy b^2 - 4ac >= 0 by MB05.
        disc = np.sqrt(max(0.0, b * b - 4.0 * a * c))
        lambda_star = (
            (2.0 * c / (b + disc)) if b >= 0.0 else ((b - disc) / (2.0 * a))
        )

        # Ptilde* from HLL consistency for F(M).
        Ptilde_star = fM_hll - lambda_star * fE_hll

        def star(
            lambda_alpha,
            u_alpha,
            D_alpha,
            Ye_alpha,
            E_alpha,
            M_alpha,
            Ptilde_alpha,
            S_alpha,
        ):
            denom = lambda_alpha - lambda_star
            r = (lambda_alpha - u_alpha) / denom
            D_s = D_alpha * r
            Ye_s = Ye_alpha * r
            E_s = (
                E_alpha * (lambda_alpha - u_alpha)
                + Ptilde_star * lambda_star
                - Ptilde_alpha * u_alpha
            ) / denom
            Tau_s = E_s - D_s
            M_s = (
                M_alpha * (lambda_alpha - u_alpha) + Ptilde_star - Ptilde_alpha
            ) / denom
            S_s = (
                S_alpha * (lambda_alpha - u_alpha)
                + (Ptilde_star - Ptilde_alpha) * n_i
            ) / denom
            return D_s, Ye_s, Tau_s, M_s, S_s

        if lambda_star >= 0.0:
            D_s, Ye_s, Tau_s, _, S_s = star(
                lambda_L, u_L, D_L, Ye_L, E_L, M_L, Ptilde_L, S_L
            )
            G_D = fD_L + lambda_L * (D_s - D_L)
            G_Ye = fYe_L + lambda_L * (Ye_s - Ye_L)
            G_Tau = fTau_L + lambda_L * (Tau_s - Tau_L)
            G_S = fS_L + lambda_L * (S_s - S_L)
        else:
            D_s, Ye_s, Tau_s, _, S_s = star(
                lambda_R, u_R, D_R, Ye_R, E_R, M_R, Ptilde_R, S_R
            )
            G_D = fD_R + lambda_R * (D_s - D_R)
            G_Ye = fYe_R + lambda_R * (Ye_s - Ye_R)
            G_Tau = fTau_R + lambda_R * (Tau_s - Tau_R)
            G_S = fS_R + lambda_R * (S_s - S_R)

        # Magnetic + divergence-cleaning sector: raw HLL flux (HLLC does
        # not add contact resolution here).
        G_Phi = _hll_raw_G(lambda_R, lambda_L, fPhi_L, fPhi_R, Phi_L, Phi_R)
        G_B = _hll_raw_G(lambda_R, lambda_L, fB_L, fB_R, B_L, B_R)

    sub = 0.0 if not use_strong_form else 1.0
    return (
        np.asarray(G_D - sub * fD_L),
        np.asarray(G_Ye - sub * fYe_L),
        np.asarray(G_Tau - sub * fTau_L),
        np.asarray(G_S - sub * fS_L),
        np.asarray(G_B - sub * fB_L),
        np.asarray(G_Phi - sub * fPhi_L),
    )
