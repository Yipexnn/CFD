#include <cfd/fv/energy.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cfd::fv {

EnergyEquation::EnergyEquation(const FlowSolver& flow, EnergySettings settings,
                               const std::map<std::string, ThermalBoundary>& conditions)
    : geo_(&flow.geometry()), settings_(std::move(settings)) {
    init(flow, conditions);
}

EnergyEquation::EnergyEquation(const FlowSolver& flow, const ConjugateMesh& conjugate,
                               EnergySettings settings,
                               const std::map<std::string, ThermalBoundary>& conditions)
    : geo_(conjugate.geometry), conjugate_(&conjugate), settings_(std::move(settings)) {
    if (conjugate.geometry == nullptr || conjugate.fluid == nullptr) {
        throw std::invalid_argument("EnergyEquation: incomplete conjugate mesh");
    }
    init(flow, conditions);
}

void EnergyEquation::init(const FlowSolver& flow,
                          const std::map<std::string, ThermalBoundary>& conditions) {
    const Mesh& m = geo_->mesh();
    for (const auto& [name, c] : conditions) {
        if (m.find_patch(name) == nullptr) {
            throw std::invalid_argument("EnergyEquation: unknown patch '" + name + "'");
        }
    }
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nb = static_cast<std::size_t>(m.n_boundary_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    flow_boundary_.assign(nb, -1);
    solid_.assign(nc, 0);
    if (conjugate_ == nullptr) {
        for (std::size_t b = 0; b < nb; ++b) {
            flow_boundary_[b] = static_cast<Label>(b);
        }
    } else {
        const parallel::SubMesh& fluid = *conjugate_->fluid;
        const auto snif = static_cast<std::size_t>(fluid.mesh.n_internal_faces());
        for (std::size_t sf = snif; sf < fluid.parent_face.size(); ++sf) {
            const auto pf = static_cast<std::size_t>(fluid.parent_face[sf]);
            if (pf >= nif) {
                flow_boundary_[pf - nif] = static_cast<Label>(sf - snif);
            }
        }
        capacity_.assign(nc, 1.0);
        solid_k_.assign(nc, 0.0);
        for (std::size_t c = 0; c < nc; ++c) {
            const auto& zone =
                conjugate_->zone_names[static_cast<std::size_t>(conjugate_->zone_of_cell[c])];
            for (const SolidRegion& s : conjugate_->solids) {
                if (s.zone == zone) {
                    solid_[c] = 1;
                    capacity_[c] = s.rho_cp / settings_.rho_cp;
                    solid_k_[c] = s.conductivity / settings_.rho_cp;
                }
            }
        }
    }
    face_bc_.assign(nb, {});
    bc_.assign(nb, {});
    for (std::size_t b = 0; b < nb; ++b) {
        const auto& patch = m.patches()[static_cast<std::size_t>(geo_->boundary_patch()[b])];
        const auto it = conditions.find(patch.name);
        if (geo_->halo_face(b)) {
            continue;
        }
        if (it != conditions.end()) {
            face_bc_[b] = it->second;
        } else if (flow_boundary_[b] >= 0 &&
                   flow.boundary_condition(flow_boundary_[b]).kind == FlowBoundary::VelocityInlet) {
            face_bc_[b] = {ThermalBoundary::Kind::FixedTemperature, settings_.t_ref};
        }
    }
    t_.assign(nc, settings_.t_ref);
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

std::vector<Scalar> EnergyEquation::face_flux(const FlowSolver& flow) const {
    if (conjugate_ == nullptr) {
        return flow.face_flux();
    }
    const parallel::SubMesh& fluid = *conjugate_->fluid;
    const Mesh& m = geo_->mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto snif = static_cast<std::size_t>(fluid.mesh.n_internal_faces());
    std::vector<Scalar> phi(static_cast<std::size_t>(m.n_faces()), 0.0);
    const auto& sub_phi = flow.face_flux();
    for (std::size_t sf = 0; sf < fluid.parent_face.size(); ++sf) {
        const auto pf = static_cast<std::size_t>(fluid.parent_face[sf]);
        if (sf < snif || pf >= nif) { // fluid/solid interface faces carry no flux
            phi[pf] = fluid.face_flipped[sf] != 0 ? -sub_phi[sf] : sub_phi[sf];
        }
    }
    return phi;
}

std::vector<Scalar> EnergyEquation::face_diffusivity(const FlowSolver& flow) const {
    const Mesh& m = geo_->mesh();
    const auto nc = static_cast<std::size_t>(m.n_cells());
    const auto nf = static_cast<std::size_t>(m.n_faces());
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const auto& w = geo_->weights();
    const Scalar pr_t = settings_.prandtl_t;

    // Turbulent viscosity on the energy mesh.
    std::vector<Scalar> nut(nc, 0.0);
    std::vector<Scalar> nut_b(nf - nif, 0.0);
    if (const TurbulenceModel* turb = flow.turbulence_model()) {
        const auto& cell_nut = turb->nut();
        const auto& face_nut = turb->boundary_nut();
        if (conjugate_ == nullptr) {
            nut = cell_nut;
            nut_b = face_nut;
        } else {
            const auto& map = conjugate_->fluid->parent_cell;
            for (std::size_t c = 0; c < map.size(); ++c) {
                nut[static_cast<std::size_t>(map[c])] = cell_nut[c];
            }
            for (std::size_t b = 0; b < nut_b.size(); ++b) {
                if (flow_boundary_[b] >= 0) {
                    nut_b[b] = face_nut[static_cast<std::size_t>(flow_boundary_[b])];
                }
            }
        }
    }
    const bool variable = !settings_.alpha_law.is_constant();
    auto alpha_c = [&](std::size_t c) {
        return variable ? settings_.alpha_law(t_[c]) : settings_.alpha;
    };
    auto cell_k = [&](std::size_t c) {
        return solid_[c] != 0 ? solid_k_[c] : alpha_c(c) + nut[c] / pr_t;
    };
    std::vector<Scalar> gamma(nf, settings_.alpha);
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        if (solid_[p] == 0 && solid_[n] == 0) {
            gamma[f] = w[f] * alpha_c(p) + (1.0 - w[f]) * alpha_c(n) +
                       (w[f] * nut[p] + (1.0 - w[f]) * nut[n]) / pr_t;
        } else { // harmonic interpolation across material jumps
            const Scalar kp = cell_k(p);
            const Scalar kn = cell_k(n);
            gamma[f] = kp * kn / std::max(w[f] * kn + (1.0 - w[f]) * kp, 1e-300);
        }
    }
    for (std::size_t f = nif; f < nf; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        gamma[f] = solid_[p] != 0 ? solid_k_[p] : alpha_c(p) + nut_b[f - nif] / pr_t;
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
    eq.su = source_su_.empty() ? std::vector<Scalar>(t_.size(), 0.0) : source_su_;
    eq.sp = source_sp_.empty() ? std::vector<Scalar>(t_.size(), 0.0) : source_sp_;
    eq.boundary = bc_;
    eq.gradient = flow.settings().gradient;
    eq.relaxation = dt > 0.0 ? 1.0 : settings_.relaxation;
    eq.dt = dt;
    eq.old_value = &t_old_;
    eq.capacity = capacity_.empty() ? nullptr : &capacity_;
    const std::vector<Scalar> phi = face_flux(flow);
    return solve_scalar(*geo_, phi, eq, t_, settings_.solver).initial_residual;
}

std::vector<Vec3> EnergyEquation::buoyancy() const {
    const auto b = [&](Scalar t) {
        return -settings_.beta * (t - settings_.t_ref) * settings_.gravity;
    };
    std::vector<Vec3> a;
    if (conjugate_ == nullptr) {
        a.reserve(t_.size());
        for (const Scalar t : t_) {
            a.push_back(b(t));
        }
    } else {
        for (const Label c : conjugate_->fluid->parent_cell) {
            a.push_back(b(t_[static_cast<std::size_t>(c)]));
        }
    }
    return a;
}

void EnergyEquation::set_source(std::vector<Scalar> su, std::vector<Scalar> sp) {
    if ((!su.empty() && su.size() != t_.size()) || su.size() != sp.size()) {
        throw std::invalid_argument("EnergyEquation::set_source: size mismatch");
    }
    source_su_ = std::move(su);
    source_sp_ = std::move(sp);
}

std::vector<Scalar> EnergyEquation::fluid_temperature() const {
    if (conjugate_ == nullptr) {
        return t_;
    }
    std::vector<Scalar> t;
    t.reserve(conjugate_->fluid->parent_cell.size());
    for (const Label c : conjugate_->fluid->parent_cell) {
        t.push_back(t_[static_cast<std::size_t>(c)]);
    }
    return t;
}

std::vector<Scalar> EnergyEquation::fluid_viscosity() const {
    if (settings_.nu_law.is_constant()) {
        return {};
    }
    std::vector<Scalar> nu = fluid_temperature();
    for (Scalar& v : nu) {
        v = settings_.nu_law(v);
    }
    return nu;
}

std::vector<Scalar> EnergyEquation::boundary_temperature() const {
    return boundary_values(*geo_, t_, bc_);
}

Scalar EnergyEquation::patch_heat_flow(const FlowSolver& flow, const std::string& patch) const {
    const Mesh& m = geo_->mesh();
    const auto nif = static_cast<std::size_t>(m.n_internal_faces());
    const std::vector<Scalar> gamma = face_diffusivity(flow);
    Scalar q = 0.0;
    if (const mesh::Patch* pp = m.find_patch(patch)) {
        const std::vector<Scalar> tb = boundary_temperature();
        const std::vector<Scalar> phi = face_flux(flow);
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
                // Conduction into the domain plus convective inflow of enthalpy.
                q += gamma[f] * geo_->delta_coeffs()[f] * (tb[f - nif] - t_[p]) -
                     phi[f] * tb[f - nif];
            }
        }
        return geo_->comm().sum(q);
    }
    // Conjugate: heat flow from a solid zone into the fluid (counted where the solid cell is
    // owned, so every interface face is counted once).
    int zone = -1;
    if (conjugate_ != nullptr) {
        for (std::size_t z = 0; z < conjugate_->zone_names.size(); ++z) {
            if (conjugate_->zone_names[z] == patch) {
                zone = static_cast<int>(z);
            }
        }
    }
    if (zone < 0) {
        throw std::invalid_argument("patch_heat_flow: unknown patch or zone '" + patch + "'");
    }
    const auto& zones = conjugate_->zone_of_cell;
    const Label n_owned = geo_->n_owned();
    for (std::size_t f = 0; f < nif; ++f) {
        const auto p = static_cast<std::size_t>(m.owner()[f]);
        const auto n = static_cast<std::size_t>(m.neighbour()[f]);
        const bool p_in = zones[p] == zone;
        const bool n_in = zones[n] == zone;
        if (p_in == n_in) {
            continue;
        }
        const std::size_t s = p_in ? p : n; // solid cell
        const std::size_t o = p_in ? n : p; // other cell
        if (solid_[o] != 0 || static_cast<Label>(s) >= n_owned) {
            continue;
        }
        q += gamma[f] * geo_->delta_coeffs()[f] * (t_[s] - t_[o]);
    }
    return geo_->comm().sum(q);
}

} // namespace cfd::fv
