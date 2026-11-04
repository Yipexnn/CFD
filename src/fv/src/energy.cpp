#include <cfd/fv/energy.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cfd::fv {

EnergyEquation::EnergyEquation(const FlowSolver& flow, EnergySettings settings,
                               const std::map<std::string, ThermalBoundary>& conditions)
    : geo_(&flow.geometry()), settings_(std::move(settings)) {
    const Mesh& m = geo_->mesh();
    for (const auto& [name, c] : conditions) {
        if (m.find_patch(name) == nullptr) {
            throw std::invalid_argument("EnergyEquation: unknown patch '" + name + "'");
        }
    }
    const auto nb = static_cast<std::size_t>(m.n_boundary_faces());
    face_bc_.resize(nb);
    bc_.resize(nb);
    for (std::size_t b = 0; b < nb; ++b) {
        const auto& patch = m.patches()[static_cast<std::size_t>(geo_->boundary_patch()[b])];
        const auto it = conditions.find(patch.name);
        if (it != conditions.end() && !geo_->halo_face(b)) {
            face_bc_[b] = it->second;
        } else if (flow.boundary_condition(static_cast<Label>(b)).kind ==
                   FlowBoundary::VelocityInlet) {
            face_bc_[b] = {ThermalBoundary::Kind::FixedTemperature, settings_.t_ref};
        }
    }
    t_.assign(static_cast<std::size_t>(m.n_cells()), settings_.t_ref);
    t_old_ = t_;
    update_boundary(flow);
}

void EnergyEquation::set_temperature(std::vector<Scalar> t) {
    if (t.size() != t_.size()) {
        throw std::invalid_argument("EnergyEquation::set_temperature: size mismatch");
    }
    t_ = std::move(t);
    t_old_ = t_;
}

void EnergyEquation::set_uniform_temperature(Scalar t) {
    std::fill(t_.begin(), t_.end(), t);
    t_old_ = t_;
}

std::vector<Scalar> EnergyEquation::face_diffusivity(const FlowSolver& flow) const {
    const Mesh& m = geo_->mesh();
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = geo_->weights();
    std::vector<Scalar> gamma(nf, settings_.alpha);
    const TurbulenceModel* turb = flow.turbulence_model();
    if (turb == nullptr) {
        return gamma;
    }
    const auto& nut = turb->nut();
    const auto& nut_b = turb->boundary_nut();
    const Scalar pr_t = settings_.prandtl_t;
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        gamma[f] += (w[f] * nut[p] + (1.0 - w[f]) * nut[n]) / pr_t;
    }
    for (std::size_t f = nif; f < nf; ++f) {
        gamma[f] += nut_b[f - nif] / pr_t;
    }
    return gamma;
}

void EnergyEquation::update_boundary(const FlowSolver& flow) {
    const Mesh& m = geo_->mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const std::vector<Scalar> gamma = face_diffusivity(flow);
    for (std::size_t b = 0; b < face_bc_.size(); ++b) {
        const ThermalBoundary& tb = face_bc_[b];
        switch (tb.kind) {
        case ThermalBoundary::Kind::FixedTemperature:
            bc_[b] = {ScalarBoundary::Kind::FixedValue, tb.value};
            break;
        case ThermalBoundary::Kind::HeatFlux:
            bc_[b] = {ScalarBoundary::Kind::FixedGradient, tb.value / gamma[nif + b]};
            break;
        case ThermalBoundary::Kind::Adiabatic:
            bc_[b] = {ScalarBoundary::Kind::ZeroGradient, 0.0};
            break;
        }
    }
}

Scalar EnergyEquation::correct(const FlowSolver& flow, Scalar dt) {
    if (dt > 0.0) {
        t_old_ = t_;
    }
    update_boundary(flow);
    ScalarEquation eq;
    eq.gamma_face = face_diffusivity(flow);
    eq.su.assign(t_.size(), 0.0);
    eq.sp.assign(t_.size(), 0.0);
    eq.boundary = bc_;
    eq.relaxation = dt > 0.0 ? 1.0 : settings_.relaxation;
    eq.dt = dt;
    eq.old_value = &t_old_;
    return solve_scalar(*geo_, flow.face_flux(), eq, t_, settings_.solver).initial_residual;
}

std::vector<Vec3> EnergyEquation::buoyancy() const {
    std::vector<Vec3> a(t_.size());
    for (std::size_t c = 0; c < t_.size(); ++c) {
        a[c] = -settings_.beta * (t_[c] - settings_.t_ref) * settings_.gravity;
    }
    return a;
}

std::vector<Scalar> EnergyEquation::boundary_temperature() const {
    return boundary_values(*geo_, t_, bc_);
}

Scalar EnergyEquation::patch_heat_flow(const FlowSolver& flow, const std::string& patch) const {
    const Mesh& m = geo_->mesh();
    const mesh::Patch* pp = m.find_patch(patch);
    if (pp == nullptr) {
        throw std::invalid_argument("patch_heat_flow: unknown patch '" + patch + "'");
    }
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const std::vector<Scalar> gamma = face_diffusivity(flow);
    const std::vector<Scalar> tb = boundary_temperature();
    Scalar q = 0.0;
    for (Label i = 0; i < pp->n_faces; ++i) {
        const auto f = static_cast<std::size_t>(pp->start_face + i);
        if (!geo_->owned_face(f)) {
            continue;
        }
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const ThermalBoundary& b = face_bc_[f - nif];
        if (b.kind == ThermalBoundary::Kind::HeatFlux) {
            q += b.value * geo_->face_areas()[f];
        } else if (b.kind == ThermalBoundary::Kind::FixedTemperature) {
            // Conduction into the fluid plus convective inflow of enthalpy.
            q += gamma[f] * geo_->delta_coeffs()[f] * (tb[f - nif] - t_[p]) -
                 flow.face_flux()[f] * tb[f - nif];
        }
    }
    return geo_->comm().sum(q);
}

} // namespace cfd::fv
