// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "NumericalAlgorithms/Interpolation/MultiCubicSpanInterpolation.hpp"
#include "Utilities/Gsl.hpp"

namespace {

// Fill an axis with `n` uniformly spaced points on [x_min, x_max].
std::vector<double> uniform_axis(const size_t n, const double x_min,
                                 const double x_max) {
  std::vector<double> pts(n);
  const double dx = (x_max - x_min) / static_cast<double>(n - 1);
  for (size_t i = 0; i < n; ++i) {
    pts[i] = x_min + static_cast<double>(i) * dx;
  }
  return pts;
}

// Fill a table's flat storage from an arbitrary scalar function of 1..3 args.
template <size_t Dim, typename Func>
std::vector<double> fill_table(const std::array<std::vector<double>, Dim>& axes,
                               const Index<Dim>& num_points, const Func& f) {
  size_t total = 1;
  for (size_t d = 0; d < Dim; ++d) {
    total *= num_points[d];
  }
  std::vector<double> data(total);
  for (size_t idx = 0; idx < total; ++idx) {
    size_t rem = idx;
    std::array<size_t, Dim> ijk{};
    for (size_t d = 0; d < Dim; ++d) {
      ijk[d] = rem % num_points[d];
      rem /= num_points[d];
    }
    if constexpr (Dim == 1) {
      data[idx] = f(axes[0][ijk[0]]);
    } else if constexpr (Dim == 2) {
      data[idx] = f(axes[0][ijk[0]], axes[1][ijk[1]]);
    } else {
      data[idx] = f(axes[0][ijk[0]], axes[1][ijk[1]], axes[2][ijk[2]]);
    }
  }
  return data;
}

template <size_t Dim>
std::array<gsl::span<const double>, Dim> make_axis_views(
    const std::array<std::vector<double>, Dim>& axes,
    const Index<Dim>& num_points) {
  std::array<gsl::span<const double>, Dim> views{};
  for (size_t d = 0; d < Dim; ++d) {
    views[d] = gsl::span<const double>{axes[d].data(), num_points[d]};
  }
  return views;
}

// A cubic polynomial in 1..3 args used to verify exact reproduction of
// degree <= 3 polynomials by the 4-point-Lagrange tensor product.
double cubic_1d(const double x) {
  return 2.0 - 0.7 * x + 0.3 * x * x - 0.5 * std::pow(x, 3);
}
double cubic_2d(const double x, const double y) {
  return 1.0 + 0.4 * x - 0.6 * y + 0.5 * x * x - 0.3 * x * y + 0.2 * y * y -
         0.15 * std::pow(x, 3) + 0.1 * x * x * y - 0.05 * x * y * y +
         0.2 * std::pow(y, 3);
}
double cubic_3d(const double x, const double y, const double z) {
  return 1.0 + 0.4 * x - 0.6 * y + 0.3 * z + 0.5 * x * y - 0.2 * y * z +
         0.15 * x * z - 0.1 * std::pow(x, 3) + 0.2 * std::pow(y, 3) -
         0.05 * std::pow(z, 3) + 0.1 * x * y * z;
}

// A smooth non-polynomial for convergence tests.
double smooth_1d(const double x) { return std::sin(1.3 * x + 0.2); }
double smooth_2d(const double x, const double y) {
  return std::sin(1.3 * x + 0.2) * std::exp(-0.4 * y);
}
double smooth_3d(const double x, const double y, const double z) {
  return std::sin(1.3 * x + 0.2) * std::exp(-0.4 * y) * std::cos(0.7 * z);
}

// Verify the tabulated 4-point Lagrange interpolant reproduces a cubic
// polynomial exactly, to floating-point tolerance.
template <bool UniformSpacing>
void test_cubic_reproduction() {
  CAPTURE(UniformSpacing);

  {
    Index<1> num_points{6};
    std::array<std::vector<double>, 1> axes{uniform_axis(6, -1.0, 1.0)};
    const auto data = fill_table<1>(axes, num_points,
                                    [](const double x) { return cubic_1d(x); });
    intrp::MultiCubicSpanInterpolation<1, 1, UniformSpacing> interp(
        make_axis_views<1>(axes, num_points),
        gsl::span<const double>{data.data(), data.size()}, num_points);
    for (const double x : {-0.83, -0.51, -0.11, 0.09, 0.31, 0.62, 0.87}) {
      const auto w = interp.get_weights(x);
      CHECK(interp.interpolate(w, 0) == approx(cubic_1d(x)));
    }
  }

  {
    Index<2> num_points{{6, 7}};
    std::array<std::vector<double>, 2> axes{uniform_axis(6, -1.0, 1.0),
                                            uniform_axis(7, -2.0, 2.0)};
    const auto data = fill_table<2>(
        axes, num_points,
        [](const double x, const double y) { return cubic_2d(x, y); });
    intrp::MultiCubicSpanInterpolation<2, 1, UniformSpacing> interp(
        make_axis_views<2>(axes, num_points),
        gsl::span<const double>{data.data(), data.size()}, num_points);
    for (size_t i = 0; i < 5; ++i) {
      const double x = -0.7 + 0.3 * static_cast<double>(i);
      const double y = -1.7 + 0.6 * static_cast<double>(i);
      const auto w = interp.get_weights(x, y);
      CHECK(interp.interpolate(w, 0) == approx(cubic_2d(x, y)));
    }
  }

  {
    Index<3> num_points{{6, 5, 5}};
    std::array<std::vector<double>, 3> axes{uniform_axis(6, -1.0, 1.0),
                                            uniform_axis(5, -1.0, 1.0),
                                            uniform_axis(5, -1.0, 1.0)};
    const auto data = fill_table<3>(
        axes, num_points, [](const double x, const double y, const double z) {
          return cubic_3d(x, y, z);
        });
    intrp::MultiCubicSpanInterpolation<3, 1, UniformSpacing> interp(
        make_axis_views<3>(axes, num_points),
        gsl::span<const double>{data.data(), data.size()}, num_points);
    for (size_t i = 0; i < 5; ++i) {
      const double x = -0.7 + 0.3 * static_cast<double>(i);
      const double y = -0.6 + 0.25 * static_cast<double>(i);
      const double z = -0.5 + 0.2 * static_cast<double>(i);
      const auto w = interp.get_weights(x, y, z);
      CHECK(interp.interpolate(w, 0) == approx(cubic_3d(x, y, z)));
    }
  }
}

// Verify O(h^4) convergence on a smooth non-polynomial function by halving
// the grid spacing and checking the error drops by close to a factor of 16.
template <bool UniformSpacing>
void test_convergence() {
  CAPTURE(UniformSpacing);
  // 1D convergence.
  const double x_target = 0.373;
  double previous_error = 0.0;
  for (size_t n : {8, 16, 32}) {
    Index<1> num_points{n};
    std::array<std::vector<double>, 1> axes{uniform_axis(n, -1.0, 1.0)};
    const auto data = fill_table<1>(axes, num_points, &smooth_1d);
    intrp::MultiCubicSpanInterpolation<1, 1, UniformSpacing> interp(
        make_axis_views<1>(axes, num_points),
        gsl::span<const double>{data.data(), data.size()}, num_points);
    const auto w = interp.get_weights(x_target);
    const double err = std::abs(interp.interpolate(w, 0) - smooth_1d(x_target));
    if (previous_error > 0.0) {
      // Expect ~16x reduction when doubling n; allow some slack for
      // finite-h prefactor effects.
      CHECK(err < previous_error / 8.0);
    }
    previous_error = err;
  }

  // 3D: verify convergence over a factor-4 grid refinement. Between the
  // coarsest and finest grids we expect ~256x error reduction for pure
  // O(h^4); require at least 16x to remain robust across smooth-function
  // constants and target positions. Exact cubic reproduction (checked
  // above) is the strong guarantee of the correct order; this is a
  // sanity check on the tensor-product structure.
  const std::array<double, 3> target_3d{0.212, -0.083, 0.117};
  double err_coarse = 0.0;
  double err_fine = 0.0;
  for (size_t n : {12, 48}) {
    Index<3> num_points{{n, n, n}};
    std::array<std::vector<double>, 3> axes{uniform_axis(n, -1.0, 1.0),
                                            uniform_axis(n, -1.0, 1.0),
                                            uniform_axis(n, -1.0, 1.0)};
    const auto data = fill_table<3>(axes, num_points, &smooth_3d);
    intrp::MultiCubicSpanInterpolation<3, 1, UniformSpacing> interp(
        make_axis_views<3>(axes, num_points),
        gsl::span<const double>{data.data(), data.size()}, num_points);
    const auto w = interp.get_weights(target_3d[0], target_3d[1], target_3d[2]);
    const double err =
        std::abs(interp.interpolate(w, 0) -
                 smooth_3d(target_3d[0], target_3d[1], target_3d[2]));
    if (n == 12) {
      err_coarse = err;
    } else {
      err_fine = err;
    }
  }
  CAPTURE(err_coarse);
  CAPTURE(err_fine);
  CHECK(err_fine < err_coarse / 16.0);
}

// Verify multi-variable indexing gives the same result as running one
// variable at a time.
void test_multi_variable() {
  // 12 points per axis is enough for cubic interpolation of smooth_2d to
  // land inside the smaller tolerance below.
  const size_t n = 12;
  Index<2> num_points{{n, n}};
  std::array<std::vector<double>, 2> axes{uniform_axis(n, -1.0, 1.0),
                                          uniform_axis(n, -1.0, 1.0)};
  constexpr size_t nvars = 3;
  const size_t total = num_points[0] * num_points[1] * nvars;
  std::vector<double> data(total);
  for (size_t j = 0; j < num_points[1]; ++j) {
    for (size_t i = 0; i < num_points[0]; ++i) {
      const size_t flat = i + num_points[0] * j;
      const double x = axes[0][i];
      const double y = axes[1][j];
      data[0 + nvars * flat] = 1.0 + x - y;
      data[1 + nvars * flat] = smooth_2d(x, y);
      data[2 + nvars * flat] = cubic_2d(x, y);
    }
  }
  intrp::MultiCubicSpanInterpolation<2, nvars, true> interp(
      make_axis_views<2>(axes, num_points),
      gsl::span<const double>{data.data(), data.size()}, num_points);
  const double x = 0.14;
  const double y = -0.37;
  const auto w = interp.get_weights(x, y);
  CHECK(interp.interpolate(w, 0) == approx(1.0 + x - y));
  // smooth_2d is not a cubic polynomial; only require it to be close.
  CHECK(std::abs(interp.interpolate(w, 1) - smooth_2d(x, y)) < 5.0e-4);
  CHECK(interp.interpolate(w, 2) == approx(cubic_2d(x, y)));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Interpolation.MultiCubicSpanInterpolation",
                  "[Unit][NumericalAlgorithms]") {
  test_cubic_reproduction<true>();
  test_cubic_reproduction<false>();
  test_convergence<true>();
  test_convergence<false>();
  test_multi_variable();
}
