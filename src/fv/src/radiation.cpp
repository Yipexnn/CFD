#include <cfd/fv/radiation.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cfd::fv {

P1Radiation::P1Radiation(const FlowSolver& flow, RadiationSettings settings)
    : geo_(&flow.geometry()), settings_(std::move(settings)) {
    const RadiationSettings& s = settings_;
    if (!(s.absorption > 0.0) || s.scattering < 0.0 || std::abs(s.anisotropy) > 1.0) {
        throw std::invalid_argument(
            "P1Radiation: needs absorption > 0, scattering >= 0, |anisotropy| <= 1");
    }
    gamma_ = 1.0 / (3.0 * (s.absorption + s.scattering) - s.anisotropy * s.scattering);
    const Mesh& m = geo_->mesh();
    const auto nb = static_cast<std::size_t>(m.n_boundary_faces());
    marshak_.assign(nb, 0);
    wall_coeff_.assign(nb, 0.0);
    bc_.assign(nb, {});
    for (std::size_t b = 0; b < nb; ++b) {
        const FlowBoundary kind = flow.boundary_condition(static_cast<Label>(b)).kind;
        if (kind == FlowBoundary::Symmetry || kind == FlowBoundary::Empty) {
            continue;
        }
        const auto patch = static_cast<std::size_t>(geo_->boundary_patch()[b]);
        const auto it = s.patch_emissivity.find(m.patches()[patch].name);
        const Scalar eps = it != s.patch_emissivity.end() ? it->second : s.emissivity;
        if (!(eps > 0.0 && eps <= 1.0)) {
            throw std::invalid_argument("P1Radiation: emissivity must be in (0, 1]");
        }
        marshak_[b] = 1;
        wall_coeff_[b] = eps / (2.0 * (2.0 - eps));
        bc_[b].kind = ScalarBoundary::Kind::Mixed;
        bc_[b].coefficient = wall_coeff_[b] / gamma_;
    }
    for (const auto& [name, eps] : s.patch_emissivity) {
        if (m.find_patch(name) == nullptr) {
            throw std::invalid_argument("P1Radiation: unknown patch '" + name + "'");
        }
        (void)eps;
    }
    g_.assign(static_cast<std::size_t>(m.n_cells()), 0.0);
}

void P1Radiation::set_incident_radiation(std::vector<Scalar> g) {
    if (g.size() != g_.size()) {
        throw std::invalid_argument("P1Radiation::set_incident_radiation: size mismatch");
    }
    g_ = std::move(g);
    geo_->exchange(g_);
}

Scalar P1Radiation::correct(const std::vector<Scalar>& t, const std::vector<Scalar>& t_boundary) {
    const Mesh& m = geo_->mesh();
    const auto nc = g_.size();
    const Scalar a = settings_.absorption;
    for (std::size_t b = 0; b < bc_.size(); ++b) {
        if (marshak_[b] != 0) {
            bc_[b].value = 4.0 * k_sigma * std::pow(t_boundary[b], 4);
        }
    }
    ScalarEquation eq;
    eq.gamma_face.assign(static_cast<std::size_t>(m.n_faces()), gamma_);
    eq.su.resize(nc);
    eq.sp.assign(nc, -a);
    for (std::size_t c = 0; c < nc; ++c) {
        eq.su[c] = 4.0 * a * k_sigma * std::pow(t[c], 4);
    }
    eq.boundary = bc_;
    eq.linear_upwind = false;
    const std::vector<Scalar> no_flux(static_cast<std::size_t>(m.n_faces()), 0.0);
    return solve_scalar(*geo_, no_flux, eq, g_, settings_.solver).initial_residual;
}

void P1Radiation::energy_source(const std::vector<Scalar>& t, std::vector<Scalar>& su,
                                std::vector<Scalar>& sp) const {
    const Scalar a = settings_.absorption;
    su.resize(t.size());
    sp.resize(t.size());
    for (std::size_t c = 0; c < t.size(); ++c) {
        // a (G - 4 sigma T^4) linearised around the current T (Newton): sp = -16 a sigma T^3.
        const Scalar t3 = std::pow(t[c], 3);
        su[c] = a * (g_[c] + 12.0 * k_sigma * t3 * t[c]);
        sp[c] = -16.0 * a * k_sigma * t3;
    }
}

std::vector<Scalar> P1Radiation::wall_heat_flux() const {
    const std::vector<Scalar> gb = boundary_values(*geo_, g_, bc_);
    std::vector<Scalar> q(bc_.size(), 0.0);
    for (std::size_t b = 0; b < q.size(); ++b) {
        if (marshak_[b] != 0) {
            q[b] = wall_coeff_[b] * (bc_[b].value - gb[b]);
        }
    }
    return q;
}

Scalar P1Radiation::patch_heat_flow(const std::string& patch) const {
    const Mesh& m = geo_->mesh();
    const auto* pp = m.find_patch(patch);
    if (pp == nullptr) {
        throw std::invalid_argument("P1Radiation: unknown patch '" + patch + "'");
    }
    const std::vector<Scalar> q = wall_heat_flux();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    Scalar sum = 0.0;
    for (Label i = 0; i < pp->n_faces; ++i) {
        const auto f = static_cast<std::size_t>(pp->start_face + i);
        if (geo_->owned_face(f)) {
            sum += q[f - nif] * geo_->face_areas()[f];
        }
    }
    return geo_->comm().sum(sum);
}

} // namespace cfd::fv
