#pragma once

#include <cfd/core/types.hpp>
#include <cmath>
#include <vector>

namespace cfd::fv {

using core::Scalar;

// Temperature-dependent material property phi(T).
//   Constant:   phi = reference
//   Polynomial: phi = sum_i coefficients[i] T^i
//   Sutherland: phi = reference (T / t_ref)^1.5 (t_ref + s) / (T + s)   (gas viscosity)
//   PowerLaw:   phi = reference (T / t_ref)^exponent
struct PropertyLaw {
    enum class Kind { Constant, Polynomial, Sutherland, PowerLaw };
    Kind kind = Kind::Constant;
    Scalar reference = 0.0;
    std::vector<Scalar> coefficients;
    Scalar t_ref = 273.15;
    Scalar s = 110.4;      // Sutherland temperature [K] (air)
    Scalar exponent = 0.7; // power law

    [[nodiscard]] bool is_constant() const { return kind == Kind::Constant; }

    [[nodiscard]] Scalar operator()(Scalar t) const {
        switch (kind) {
        case Kind::Constant:
            break;
        case Kind::Polynomial: {
            Scalar v = 0.0;
            for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
                v = v * t + *it; // Horner
            }
            return v;
        }
        case Kind::Sutherland:
            return reference * std::pow(t / t_ref, 1.5) * (t_ref + s) / (t + s);
        case Kind::PowerLaw:
            return reference * std::pow(t / t_ref, exponent);
        }
        return reference;
    }
};

} // namespace cfd::fv
