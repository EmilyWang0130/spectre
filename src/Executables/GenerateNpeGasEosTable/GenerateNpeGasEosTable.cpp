// Distributed under the MIT License.
// See LICENSE.txt for details.
// Physics: cold relativistic degenerate npe matter + first-order (π²T²)
// electron thermal correction.
//
// Unit conventions (same as ConvertComposeTable output):
//   nb:            1/fm³
//   T:             MeV
//   Ye:            dimensionless
//   pressure:      MeV/fm³
//   eps (ε_H5):    dimensionless  (= E_per_baryon / m_n - 1)
//   cs²:           dimensionless  (c = 1)
//   lepton mu:     MeV
//   dp_depsilon:   1/fm³          (= ∂P/∂(E_per_baryon_MeV) at fixed nb,Ye)
//   zeta:          MeV/fm³        (= ∂P/∂Ye at fixed nb,ε)
//
// Tabulated3d unit conversions applied after reading the H5:
//   pressure  → multiply by 1/pressure_unit
//   dp_depsilon → multiply by neutron_mass_nuclear / pressure_unit_nuclear
//   zeta      → multiply by 1/pressure_unit
//   density   → log(nb * neutron_mass_nuclear / pressure_unit_nuclear)

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "DataStructures/DataVector.hpp"
#include "IO/H5/EosTable.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"

extern "C" void CkRegisterMainModule(void) {}

namespace {

// ---------------------------------------------------------------------------
// Physical constants in nuclear units (MeV, fm).
// "c = 1" here means nuclear natural units: speeds measured in units of c,
// so cs² is dimensionless (cs²/c²). This is NOT geometric units (G=c=M☉=1);
// the H5 file stores quantities in MeV/fm³. Tabulated3d converts to geometric
// units internally when reading the table.
// ---------------------------------------------------------------------------
constexpr double hbarc = hydro::units::nuclear::hbarc;           // MeV·fm
constexpr double mass_e = hydro::units::nuclear::electron_mass;  // MeV
constexpr double mass_p = hydro::units::nuclear::proton_mass;    // MeV
constexpr double mass_n = hydro::units::nuclear::neutron_mass;   // MeV

// ---------------------------------------------------------------------------
// Textbook Fermi-gas functions (Chandrasekhar 1935 notation)
//
//   φ(x) = (1/8π²)[x√(1+x²)(2x²/3 − 1) + ln(x + √(1+x²))]
//   χ(x) = (1/8π²)[x√(1+x²)(1 + 2x²)   − ln(x + √(1+x²))]
//
//   Species pressure:      P_i = (m_i⁴/ħ³) φ(x_i)   [MeV/fm³]
//   Species energy density: e_i = (m_i⁴/ħ³) χ(x_i)   [MeV/fm³]
//   Relativistic param:    x_i = ħ(3π²n_i)^{1/3}/m_i  [dimensionless]
// ---------------------------------------------------------------------------
double phi_fn(double x) {
  const double s = std::sqrt(1.0 + x * x);
  return (1.0 / (8.0 * M_PI * M_PI)) *
         (x * s * (2.0 * x * x / 3.0 - 1.0) + std::log(x + s));
}

double chi_fn(double x) {
  const double s = std::sqrt(1.0 + x * x);
  return (1.0 / (8.0 * M_PI * M_PI)) *
         (x * s * (1.0 + 2.0 * x * x) - std::log(x + s));
}

// x_i = ħ (3π² n)^{1/3} / m    [n in 1/fm³, m in MeV → dimensionless]
double x_param(double n, double m) {
  if (n <= 0.0)
    return 0.0;
  return hbarc * std::cbrt(3.0 * M_PI * M_PI * n) / m;
}

// Cold pressure of one species: (m⁴/ħ³) φ(x)   [MeV/fm³]
double cold_pressure_one(double n, double m) {
  if (n <= 0.0)
    return 0.0;
  const double x = x_param(n, m);
  return (m * m * m * m / (hbarc * hbarc * hbarc)) * phi_fn(x);
}

// Cold energy density of one species: (m⁴/ħ³) χ(x)   [MeV/fm³]
double cold_energy_density_one(double n, double m) {
  if (n <= 0.0)
    return 0.0;
  const double x = x_param(n, m);
  return (m * m * m * m / (hbarc * hbarc * hbarc)) * chi_fn(x);
}

// ---------------------------------------------------------------------------
// Thermal corrections (electron gas, first order in T)
//
// δ(E/baryon) = (π²/2) Ye T² / μ_E           [MeV]
// δP          = (nb/3) δ(E/baryon)            [MeV/fm³]
// σ (entropy per baryon) = π² Ye T / μ_E     [dimensionless]
//
// where μ_E = ħ(3π²n_e)^{1/3}   [MeV] is the electron Fermi energy.
//
// To avoid 0/0 when Ye = 0, rewrite Ye/μ_E = Ye^{2/3} nb^{-1/3} /
// (ħ(3π²)^{1/3}). The functions below use this form.
// ---------------------------------------------------------------------------
double thermal_ye_over_muE(double nb, double Ye) {
  // = Ye / μ_E = Ye^{2/3} nb^{-1/3} / (ħ (3π²)^{1/3})
  const double ye_nb = Ye * nb;
  if (ye_nb <= 0.0)
    return 0.0;
  return std::cbrt(ye_nb * ye_nb) / (nb * hbarc * std::cbrt(3.0 * M_PI * M_PI));
}

// δ(E/baryon) = (π²/2) T² * (Ye/μ_E)   [MeV]
double delta_eps_per_baryon(double nb, double T, double Ye) {
  return (M_PI * M_PI / 2.0) * T * T * thermal_ye_over_muE(nb, Ye);
}

// δP = (nb/3) δ(E/baryon)   [MeV/fm³]
double delta_pressure(double nb, double T, double Ye) {
  return (nb / 3.0) * delta_eps_per_baryon(nb, T, Ye);
}

// ---------------------------------------------------------------------------
// EOS functions: the six quantities written to the H5 file
// ---------------------------------------------------------------------------

// Pressure P(nb, T, Ye)   [MeV/fm³]
double analytic_pressure(double nb, double T, double Ye) {
  const double ne = Ye * nb;
  const double nn = (1.0 - Ye) * nb;
  return cold_pressure_one(ne, mass_e) + cold_pressure_one(ne, mass_p) +
         cold_pressure_one(nn, mass_n) + delta_pressure(nb, T, Ye);
}

// ε_H5 = E_per_baryon / m_n − 1   [dimensionless, CompOSE convention]
// E_per_baryon = e_total / nb   [MeV]
double analytic_specific_internal_energy(double nb, double T, double Ye) {
  const double ne = Ye * nb;
  const double nn = (1.0 - Ye) * nb;
  const double e_cold = cold_energy_density_one(ne, mass_e) +
                        cold_energy_density_one(ne, mass_p) +
                        cold_energy_density_one(nn, mass_n);
  const double E_per_baryon = e_cold / nb + delta_eps_per_baryon(nb, T, Ye);
  return E_per_baryon / mass_n - 1.0;
}

// Adiabatic sound speed squared c_s²   [dimensionless, c = 1]
//
// c_s² = (∂P/∂nb ∂σ/∂T - ∂P/∂T ∂σ/∂nb) /
//        (∂e/∂nb ∂σ/∂T - ∂e/∂T ∂σ/∂nb)
//
// where σ = entropy per baryon. Evaluating with the thermal model:
//
//   ∂P_cold/∂nb = (Ye/3)[m_e x_e²/γ_e + m_p x_p²/γ_p]
//               + ((1-Ye)/3) m_n x_n²/γ_n            [MeV]
//
//   ∂e_cold/∂nb = Ye [m_e γ_e + m_p γ_p]
//               + (1-Ye) m_n γ_n                      [MeV]
//
//   Numerator   = ∂P_cold/∂nb + (2π²/9) Ye T²/μ_E
//   Denominator = ∂e_cold/∂nb + (2π²/3) Ye T²/μ_E
//
// where γ_i = √(1 + x_i²) is the relativistic Lorentz factor at p_F.
double analytic_sound_speed_squared(double nb, double T, double Ye) {
  const double ne = Ye * nb;
  const double nn = (1.0 - Ye) * nb;
  const double xe = x_param(ne, mass_e);
  const double xp = x_param(ne, mass_p);
  const double xn = x_param(nn, mass_n);
  const double ge = std::sqrt(1.0 + xe * xe);
  const double gp = std::sqrt(1.0 + xp * xp);
  const double gn = std::sqrt(1.0 + xn * xn);

  const double dPcold_dnb =
      (Ye / 3.0) * (mass_e * xe * xe / ge + mass_p * xp * xp / gp) +
      ((1.0 - Ye) / 3.0) * (mass_n * xn * xn / gn);

  const double decold_dnb =
      Ye * (mass_e * ge + mass_p * gp) + (1.0 - Ye) * mass_n * gn;

  const double ye_muE = thermal_ye_over_muE(nb, Ye);
  const double thermal_num = (2.0 * M_PI * M_PI / 9.0) * T * T * ye_muE;
  const double thermal_den = (2.0 * M_PI * M_PI / 3.0) * T * T * ye_muE;

  return (dPcold_dnb + thermal_num) / (decold_dnb + thermal_den);
}

// Lepton chemical potential μ_L = ∂F_per_baryon/∂Ye   [MeV]
//
//   μ_L = m_e γ_e + m_p γ_p − m_n γ_n   (cold, from dE_cold/dYe)
//         − (π²/3) T²/μ_E                (thermal, from −d(δε)/dYe)
//
// At beta equilibrium this vanishes: μ_e + μ_p = μ_n.
double analytic_lepton_chemical_potential(double nb, double T, double Ye) {
  const double ne = Ye * nb;
  const double nn = (1.0 - Ye) * nb;
  const double xe = x_param(ne, mass_e);
  const double xp = x_param(ne, mass_p);
  const double xn = x_param(nn, mass_n);
  const double mu_cold = mass_e * std::sqrt(1.0 + xe * xe) +
                         mass_p * std::sqrt(1.0 + xp * xp) -
                         mass_n * std::sqrt(1.0 + xn * xn);
  double mu_thermal = 0.0;
  if (ne > 0.0) {
    const double muE = hbarc * std::cbrt(3.0 * M_PI * M_PI * ne);
    mu_thermal = -(M_PI * M_PI / 3.0) * T * T / muE;
  }
  return mu_cold + mu_thermal;
}

// dp_depsilon = ∂P/∂(E_per_baryon_MeV) at fixed nb,Ye   [1/fm³]
//
// For this thermal model: ∂P/∂T / ∂E_per_baryon/∂T = (nb/3 δP/δT) / δE/δT
// Both T-derivatives come from the electron thermal correction:
//   ∂P/∂T = 2δP/T,  ∂(E/baryon)/∂T = δ(E/baryon)*2/T
// Their ratio = nb/3, independent of T and Ye.
double analytic_dp_depsilon(double nb, double /*T*/, double /*Ye*/) {
  return nb / 3.0;
}

// ζ = ∂P/∂Ye at fixed nb and ε   [MeV/fm³]
//
// From the Jacobian identity evaluated analytically (see Mathematica notebook):
//
//   ζ = (nb/3) [m_n/γ_n − m_p/γ_p − m_e/γ_e]
//
// where γ_i = √(1 + x_i²). The thermal correction cancels in this ratio
// for the linear-in-T² model.
double analytic_zeta(double nb, double /*T*/, double Ye) {
  const double ne = Ye * nb;
  const double nn = (1.0 - Ye) * nb;
  const double xe = x_param(ne, mass_e);
  const double xp = x_param(ne, mass_p);
  const double xn = x_param(nn, mass_n);
  return (nb / 3.0) * (mass_n / std::sqrt(1.0 + xn * xn) -
                       mass_p / std::sqrt(1.0 + xp * xp) -
                       mass_e / std::sqrt(1.0 + xe * xe));
}

// ---------------------------------------------------------------------------
// Grid helpers
// ---------------------------------------------------------------------------
struct GridSpec {
  std::array<double, 2> bounds{};
  size_t number_of_points = 0;
  bool log_spacing = false;
};

std::vector<double> make_grid(const GridSpec& spec) {
  std::vector<double> grid(spec.number_of_points);
  if (spec.number_of_points == 1) {
    grid[0] = spec.bounds[0];
    return grid;
  }
  if (spec.log_spacing) {
    const double lo = std::log(spec.bounds[0]);
    const double hi = std::log(spec.bounds[1]);
    const double dl =
        (hi - lo) / static_cast<double>(spec.number_of_points - 1);
    for (size_t i = 0; i < spec.number_of_points; ++i) {
      grid[i] = std::exp(lo + dl * static_cast<double>(i));
    }
  } else {
    const double dx = (spec.bounds[1] - spec.bounds[0]) /
                      static_cast<double>(spec.number_of_points - 1);
    for (size_t i = 0; i < spec.number_of_points; ++i) {
      grid[i] = spec.bounds[0] + dx * static_cast<double>(i);
    }
  }
  return grid;
}

// Flattening order expected by Tabulated3D:
//   Ye varies fastest, n_b next, T slowest.
//   index = iYe + nYe*(in + nN*iT)
size_t table_index(size_t in, size_t iT, size_t iYe, size_t nN, size_t nYe) {
  return iYe + nYe * (in + nN * iT);
}

// ---------------------------------------------------------------------------
// Main writer
// ---------------------------------------------------------------------------
void write_table(const std::string& output_filename,
                 const std::string& eos_subfile_name, const size_t nN,
                 const size_t nT, const size_t nYe) {
  const GridSpec nb_spec{
      {1.0e-3, 1.0},  // 1/fm³: sub-nuclear to ~6× saturation
      nN,
      true  // log spacing
  };
  const GridSpec T_spec{
      {0.1, 50.0},  // MeV: low to moderate temperature
      nT,
      true  // log spacing
  };
  const GridSpec Ye_spec{
      {0.05, 0.55},  // electron fraction
      nYe,
      false  // linear spacing
  };

  const auto nb_grid = make_grid(nb_spec);
  const auto T_grid = make_grid(T_spec);
  const auto Ye_grid = make_grid(Ye_spec);

  const size_t nN_ = nb_spec.number_of_points;
  const size_t nT_ = T_spec.number_of_points;
  const size_t nYe_ = Ye_spec.number_of_points;
  const size_t ntot = nN_ * nT_ * nYe_;

  DataVector pressure(ntot);
  DataVector specific_internal_energy(ntot);
  DataVector sound_speed_squared(ntot);
  DataVector lepton_chemical_potential(ntot);
  DataVector dp_depsilon(ntot);
  DataVector zeta(ntot);

  for (size_t iT = 0; iT < nT_; ++iT) {
    for (size_t in = 0; in < nN_; ++in) {
      for (size_t iYe = 0; iYe < nYe_; ++iYe) {
        const double nb = nb_grid[in];
        const double T = T_grid[iT];
        const double Ye = Ye_grid[iYe];
        const size_t s = table_index(in, iT, iYe, nN_, nYe_);

        pressure[s] = analytic_pressure(nb, T, Ye);
        specific_internal_energy[s] =
            analytic_specific_internal_energy(nb, T, Ye);
        sound_speed_squared[s] = analytic_sound_speed_squared(nb, T, Ye);
        lepton_chemical_potential[s] =
            analytic_lepton_chemical_potential(nb, T, Ye);
        dp_depsilon[s] = analytic_dp_depsilon(nb, T, Ye);
        zeta[s] = analytic_zeta(nb, T, Ye);
      }
    }
  }

  h5::H5File<h5::AccessType::ReadWrite> h5_file(output_filename, true);
  auto& eos_table = h5_file.insert<h5::EosTable>(
      eos_subfile_name,
      std::vector<std::string>{"number density", "temperature",
                               "electron fraction"},
      std::vector{nb_spec.bounds, T_spec.bounds, Ye_spec.bounds},
      std::vector{nb_spec.number_of_points, T_spec.number_of_points,
                  Ye_spec.number_of_points},
      std::vector{nb_spec.log_spacing, T_spec.log_spacing, Ye_spec.log_spacing},
      false);

  eos_table.write_quantity("pressure", pressure);
  eos_table.write_quantity("specific internal energy",
                           specific_internal_energy);
  eos_table.write_quantity("sound speed squared", sound_speed_squared);
  eos_table.write_quantity("lepton chemical potential",
                           lepton_chemical_potential);
  eos_table.write_quantity("dp_depsilon", dp_depsilon);
  eos_table.write_quantity("zeta", zeta);
}

}  // namespace

int main(int argc, char** argv) {
  namespace bpo = boost::program_options;
  bpo::options_description opts(
      "Generate a SpECTRE Tabulated3D-compatible HDF5 EOS table from the\n"
      "analytical npe ideal Fermi gas model.\n\nOptions");
  // clang-format off
    opts.add_options()
        ("help,h", "Show this help message.")
        ("nN",  bpo::value<size_t>()->default_value(40),
         "Number density (nb) grid points.")
        ("nT",  bpo::value<size_t>()->default_value(20),
         "Temperature grid points.")
        ("nYe", bpo::value<size_t>()->default_value(20),
         "Electron fraction grid points.")
        ("output,o", bpo::value<std::string>()->default_value("npe_gas_eos.h5"),
         "Output HDF5 filename.")
        ("subfile", bpo::value<std::string>()->default_value("npe_gas"),
         "EOS subfile name inside the HDF5 file (without leading /).");
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::parse_command_line(argc, argv, opts), vm);
  bpo::notify(vm);
  if (vm.count("help") != 0) {
    Parallel::printf("%s\n", opts);
    return 0;
  }
  write_table(vm["output"].as<std::string>(), vm["subfile"].as<std::string>(),
              vm["nN"].as<size_t>(), vm["nT"].as<size_t>(),
              vm["nYe"].as<size_t>());
  return 0;
}
