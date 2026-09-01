# Distributed under the MIT License.
# See LICENSE.txt for details.

# Python reference implementation of the GR-aware HLLC boundary correction
# via the two-scalar hat transform (plan §4). Mirrors `Hll.py`'s
# argument-order convention (state, flux, temporary, primitive with
# SpatialVelocity at the end); the test framework passes values
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
    # New packaged data for the two-scalar hat transform (plan §3):
    # α at the face and β·n (raw, not mesh-corrected).
    lapse_at_interface = np.asarray(lapse)
    shift_dot_normal = np.asarray(beta_dot_n)

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
        lapse_at_interface,
        shift_dot_normal,
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
    interior_lapse_at_interface,
    interior_shift_dot_normal,
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
    exterior_lapse_at_interface,
    exterior_shift_dot_normal,
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
    S_R = np.asarray(exterior_tilde_s)
    B_R = np.asarray(exterior_tilde_b)
    fS_R = -np.asarray(exterior_normal_dot_flux_tilde_s)
    fB_R = -np.asarray(exterior_normal_dot_flux_tilde_b)

    # Averaged interface geometry (plan §3): alpha symmetric, beta_n
    # antisymmetric like the normal itself.
    alpha = 0.5 * (
        float(interior_lapse_at_interface) + float(exterior_lapse_at_interface)
    )
    beta_n_eff = 0.5 * (
        float(interior_shift_dot_normal) - float(exterior_shift_dot_normal)
    )
    alpha_inv = 1.0 / alpha

    # Hat transform of L/R outer wave speeds, fluxes, and pressure (plan
    # §4.1 steps 3-5). The hatted system satisfies F_hat(E) = M, so the
    # existing MB05 quadratic + star-state math runs directly.
    nu_L = (lambda_L + beta_n_eff) * alpha_inv
    nu_R = (lambda_R + beta_n_eff) * alpha_inv
    v_hat_L = (u_L + beta_n_eff) * alpha_inv
    v_hat_R = (u_R + beta_n_eff) * alpha_inv
    P_hat_L = Ptilde_L * alpha_inv
    P_hat_R = Ptilde_R * alpha_inv

    F_hat_D_L = (fD_L + beta_n_eff * D_L) * alpha_inv
    F_hat_Ye_L = (fYe_L + beta_n_eff * Ye_L) * alpha_inv
    F_hat_Tau_L = (fTau_L + beta_n_eff * Tau_L) * alpha_inv
    F_hat_E_L = F_hat_Tau_L + F_hat_D_L
    F_hat_M_L = (fM_L + beta_n_eff * M_L) * alpha_inv
    F_hat_S_L = (fS_L + beta_n_eff * S_L) * alpha_inv

    F_hat_D_R = (fD_R + beta_n_eff * D_R) * alpha_inv
    F_hat_Ye_R = (fYe_R + beta_n_eff * Ye_R) * alpha_inv
    F_hat_Tau_R = (fTau_R + beta_n_eff * Tau_R) * alpha_inv
    F_hat_E_R = F_hat_Tau_R + F_hat_D_R
    F_hat_M_R = (fM_R + beta_n_eff * M_R) * alpha_inv
    F_hat_S_R = (fS_R + beta_n_eff * S_R) * alpha_inv

    # Selected star state and selected hatted flux.
    if lambda_L >= 0.0:
        D_sel, Ye_sel, Tau_sel = D_L, Ye_L, Tau_L
        S_sel = S_L
        F_hat_D_sel = F_hat_D_L
        F_hat_Ye_sel = F_hat_Ye_L
        F_hat_Tau_sel = F_hat_Tau_L
        F_hat_S_sel = F_hat_S_L
        G_B = fB_L
        G_Phi = fPhi_L
    elif lambda_R <= 0.0:
        D_sel, Ye_sel, Tau_sel = D_R, Ye_R, Tau_R
        S_sel = S_R
        F_hat_D_sel = F_hat_D_R
        F_hat_Ye_sel = F_hat_Ye_R
        F_hat_Tau_sel = F_hat_Tau_R
        F_hat_S_sel = F_hat_S_R
        G_B = fB_R
        G_Phi = fPhi_R
    else:
        # Star region. MB05 quadratic in hatted quantities.
        dnu = nu_R - nu_L
        E_hll = (nu_R * E_R - nu_L * E_L + F_hat_E_L - F_hat_E_R) / dnu
        M_hll = (nu_R * M_R - nu_L * M_L + F_hat_M_L - F_hat_M_R) / dnu
        F_hat_E_hll = (
            nu_R * F_hat_E_L - nu_L * F_hat_E_R + nu_L * nu_R * (E_R - E_L)
        ) / dnu
        F_hat_M_hll = (
            nu_R * F_hat_M_L - nu_L * F_hat_M_R + nu_L * nu_R * (M_R - M_L)
        ) / dnu

        # MB05 Eq. 18 quadratic in hatted quantities. Numerical Recipes
        # stable-root form.
        a = F_hat_E_hll
        b = E_hll + F_hat_M_hll
        c = M_hll
        # Defensive sqrt guard for the random-data unit test.
        disc = np.sqrt(max(0.0, b * b - 4.0 * a * c))
        nu_star = (
            (2.0 * c / (b + disc)) if b >= 0.0 else ((b - disc) / (2.0 * a))
        )

        # Hatted star pressure.
        P_hat_star = F_hat_M_hll - nu_star * F_hat_E_hll

        def star_hat(
            nu_alpha,
            v_hat_alpha,
            D_alpha,
            Ye_alpha,
            E_alpha,
            M_alpha,
            P_hat_alpha,
            S_alpha,
        ):
            denom = nu_alpha - nu_star
            r = (nu_alpha - v_hat_alpha) / denom
            D_s = D_alpha * r
            Ye_s = Ye_alpha * r
            E_s = (
                E_alpha * (nu_alpha - v_hat_alpha)
                + P_hat_star * nu_star
                - P_hat_alpha * v_hat_alpha
            ) / denom
            Tau_s = E_s - D_s
            _M_s = (
                M_alpha * (nu_alpha - v_hat_alpha) + P_hat_star - P_hat_alpha
            ) / denom
            S_s = (
                S_alpha * (nu_alpha - v_hat_alpha)
                + (P_hat_star - P_hat_alpha) * n_i
            ) / denom
            return D_s, Ye_s, Tau_s, S_s

        # Region select in the star fan: nu_star vs w_face =
        # beta_n_eff / alpha, equivalently
        # lambda_star_coord = alpha * nu_star - beta_n_eff vs 0.
        lambda_star_coord = alpha * nu_star - beta_n_eff

        if lambda_star_coord >= 0.0:
            D_s, Ye_s, Tau_s, S_s = star_hat(
                nu_L, v_hat_L, D_L, Ye_L, E_L, M_L, P_hat_L, S_L
            )
            D_sel, Ye_sel, Tau_sel, S_sel = D_s, Ye_s, Tau_s, S_s
            F_hat_D_sel = F_hat_D_L + nu_L * (D_s - D_L)
            F_hat_Ye_sel = F_hat_Ye_L + nu_L * (Ye_s - Ye_L)
            F_hat_Tau_sel = F_hat_Tau_L + nu_L * (Tau_s - Tau_L)
            F_hat_S_sel = F_hat_S_L + nu_L * (S_s - S_L)
        else:
            D_s, Ye_s, Tau_s, S_s = star_hat(
                nu_R, v_hat_R, D_R, Ye_R, E_R, M_R, P_hat_R, S_R
            )
            D_sel, Ye_sel, Tau_sel, S_sel = D_s, Ye_s, Tau_s, S_s
            F_hat_D_sel = F_hat_D_R + nu_R * (D_s - D_R)
            F_hat_Ye_sel = F_hat_Ye_R + nu_R * (Ye_s - Ye_R)
            F_hat_Tau_sel = F_hat_Tau_R + nu_R * (Tau_s - Tau_R)
            F_hat_S_sel = F_hat_S_R + nu_R * (S_s - S_R)

        # Magnetic + divergence-cleaning sector: raw HLL flux (hybrid).
        G_Phi = _hll_raw_G(lambda_R, lambda_L, fPhi_L, fPhi_R, Phi_L, Phi_R)
        G_B = _hll_raw_G(lambda_R, lambda_L, fB_L, fB_R, B_L, B_R)

    # Inverse hat transform: G_coord = alpha * F_hat_sel - beta_n_eff *
    # U_sel. Requires both the selected state and the selected flux.
    G_D = alpha * F_hat_D_sel - beta_n_eff * D_sel
    G_Ye = alpha * F_hat_Ye_sel - beta_n_eff * Ye_sel
    G_Tau = alpha * F_hat_Tau_sel - beta_n_eff * Tau_sel
    G_S = alpha * F_hat_S_sel - beta_n_eff * S_sel

    sub = 0.0 if not use_strong_form else 1.0
    return (
        np.asarray(G_D - sub * fD_L),
        np.asarray(G_Ye - sub * fYe_L),
        np.asarray(G_Tau - sub * fTau_L),
        np.asarray(G_S - sub * fS_L),
        np.asarray(G_B - sub * fB_L),
        np.asarray(G_Phi - sub * fPhi_L),
    )
