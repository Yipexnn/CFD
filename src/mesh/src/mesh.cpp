#include <algorithm>
#include <cfd/mesh/mesh.hpp>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cfd::mesh {

namespace {

constexpr Scalar k_tiny = 1e-300;

} // namespace

Mesh::Mesh(std::vector<Vec3> points, std::vector<Label> face_point_offsets,
           std::vector<Label> face_points, std::vector<Label> owner, std::vector<Label> neighbour,
           std::vector<Patch> patches, Label n_cells)
    : points_(std::move(points)), face_point_offsets_(std::move(face_point_offsets)),
      face_points_(std::move(face_points)), owner_(std::move(owner)),
      neighbour_(std::move(neighbour)), patches_(std::move(patches)), n_cells_(n_cells) {
    validate();
    compute_geometry();
}

std::span<const Label> Mesh::face(Label f) const {
    const auto begin = static_cast<std::size_t>(face_point_offsets_[static_cast<std::size_t>(f)]);
    const auto end = static_cast<std::size_t>(face_point_offsets_[static_cast<std::size_t>(f) + 1]);
    return {face_points_.data() + begin, end - begin};
}

const Patch* Mesh::find_patch(const std::string& name) const {
    const auto it = std::find_if(patches_.begin(), patches_.end(),
                                 [&](const Patch& p) { return p.name == name; });
    return it == patches_.end() ? nullptr : &*it;
}

void Mesh::set_patch_type(const std::string& name, PatchType type) {
    const auto it = std::find_if(patches_.begin(), patches_.end(),
                                 [&](const Patch& p) { return p.name == name; });
    if (it == patches_.end()) {
        throw std::invalid_argument("Mesh: unknown patch '" + name + "'");
    }
    it->type = type;
}

void Mesh::set_cell_zones(std::vector<std::string> names, std::vector<int> zone_of_cell) {
    if (zone_of_cell.size() != static_cast<std::size_t>(n_cells_)) {
        throw std::invalid_argument("Mesh::set_cell_zones: size mismatch");
    }
    for (const int z : zone_of_cell) {
        if (z < 0 || z >= static_cast<int>(names.size())) {
            throw std::invalid_argument("Mesh::set_cell_zones: zone index out of range");
        }
    }
    zone_names_ = std::move(names);
    cell_zones_ = std::move(zone_of_cell);
}

void Mesh::validate() const {
    const Label nf = n_faces();
    if (face_point_offsets_.size() != static_cast<std::size_t>(nf) + 1 ||
        face_point_offsets_.front() != 0 ||
        face_point_offsets_.back() != static_cast<Label>(face_points_.size())) {
        throw std::invalid_argument("Mesh: inconsistent face point offsets");
    }
    for (Label f = 0; f < nf; ++f) {
        const auto pts = face(f);
        if (pts.size() < 3) {
            throw std::invalid_argument("Mesh: face with fewer than 3 points");
        }
        for (const Label p : pts) {
            if (p < 0 || p >= n_points()) {
                throw std::invalid_argument("Mesh: face point index out of range");
            }
        }
        const Label o = owner_[static_cast<std::size_t>(f)];
        if (o < 0 || o >= n_cells_) {
            throw std::invalid_argument("Mesh: owner index out of range");
        }
        if (f < n_internal_faces()) {
            const Label n = neighbour_[static_cast<std::size_t>(f)];
            if (n <= o || n >= n_cells_) {
                throw std::invalid_argument(
                    "Mesh: neighbour must satisfy owner < neighbour < n_cells");
            }
        }
    }
    Label next = n_internal_faces();
    for (const Patch& p : patches_) {
        if (p.start_face != next || p.n_faces < 0) {
            throw std::invalid_argument("Mesh: patches must be contiguous after internal faces");
        }
        next += p.n_faces;
    }
    if (next != nf) {
        throw std::invalid_argument("Mesh: patches do not cover all boundary faces");
    }
}

void Mesh::compute_geometry() {
    const Label nf = n_faces();
    face_centres_.assign(static_cast<std::size_t>(nf), Vec3::Zero());
    face_area_vectors_.assign(static_cast<std::size_t>(nf), Vec3::Zero());

#pragma omp parallel for
    for (Label f = 0; f < nf; ++f) {
        const auto pts = face(f);
        const auto np = pts.size();
        const auto fi = static_cast<std::size_t>(f);
        if (np == 3) {
            const Vec3& a = points_[static_cast<std::size_t>(pts[0])];
            const Vec3& b = points_[static_cast<std::size_t>(pts[1])];
            const Vec3& c = points_[static_cast<std::size_t>(pts[2])];
            face_centres_[fi] = (a + b + c) / 3.0;
            face_area_vectors_[fi] = 0.5 * (b - a).cross(c - a);
            continue;
        }
        Vec3 pc = Vec3::Zero();
        for (const Label p : pts) {
            pc += points_[static_cast<std::size_t>(p)];
        }
        pc /= static_cast<Scalar>(np);

        Vec3 sum_n = Vec3::Zero();
        Vec3 sum_ac = Vec3::Zero();
        Scalar sum_a = 0.0;
        for (std::size_t i = 0; i < np; ++i) {
            const Vec3& p0 = points_[static_cast<std::size_t>(pts[i])];
            const Vec3& p1 = points_[static_cast<std::size_t>(pts[(i + 1) % np])];
            const Vec3 n = (p1 - p0).cross(pc - p0);
            const Scalar a = n.norm();
            sum_n += n;
            sum_a += a;
            sum_ac += a * (p0 + p1 + pc);
        }
        face_centres_[fi] = sum_a > k_tiny ? Vec3(sum_ac / (3.0 * sum_a)) : pc;
        face_area_vectors_[fi] = 0.5 * sum_n;
    }

    // Cell geometry: pyramid decomposition about an estimated centre.
    const auto nc = static_cast<std::size_t>(n_cells_);
    std::vector<Vec3> centre_estimate(nc, Vec3::Zero());
    std::vector<Label> n_cell_faces(nc, 0);
    const auto nfi = static_cast<std::size_t>(nf);
    const auto nif = static_cast<std::size_t>(n_internal_faces());
    for (std::size_t f = 0; f < nfi; ++f) {
        const auto o = static_cast<std::size_t>(owner_[f]);
        centre_estimate[o] += face_centres_[f];
        ++n_cell_faces[o];
        if (f < nif) {
            const auto n = static_cast<std::size_t>(neighbour_[f]);
            centre_estimate[n] += face_centres_[f];
            ++n_cell_faces[n];
        }
    }
    for (std::size_t c = 0; c < nc; ++c) {
        if (n_cell_faces[c] > 0) {
            centre_estimate[c] /= static_cast<Scalar>(n_cell_faces[c]);
        }
    }

    std::vector<Scalar> pyr3_vol(nc, 0.0);
    std::vector<Vec3> vol_centre(nc, Vec3::Zero());
    for (std::size_t f = 0; f < nfi; ++f) {
        const Vec3& cf = face_centres_[f];
        const Vec3& sf = face_area_vectors_[f];
        const auto o = static_cast<std::size_t>(owner_[f]);
        const Scalar vo = sf.dot(cf - centre_estimate[o]);
        pyr3_vol[o] += vo;
        vol_centre[o] += vo * (0.75 * cf + 0.25 * centre_estimate[o]);
        if (f < nif) {
            const auto n = static_cast<std::size_t>(neighbour_[f]);
            const Scalar vn = sf.dot(centre_estimate[n] - cf);
            pyr3_vol[n] += vn;
            vol_centre[n] += vn * (0.75 * cf + 0.25 * centre_estimate[n]);
        }
    }

    cell_volumes_.resize(nc);
    cell_centres_.resize(nc);
    for (std::size_t c = 0; c < nc; ++c) {
        cell_volumes_[c] = pyr3_vol[c] / 3.0;
        cell_centres_[c] =
            std::abs(pyr3_vol[c]) > k_tiny ? Vec3(vol_centre[c] / pyr3_vol[c]) : centre_estimate[c];
    }
}

void Mesh::build_cell_faces() const {
    const auto nc = static_cast<std::size_t>(n_cells_);
    cell_face_offsets_.assign(nc + 1, 0);
    const auto nfi = static_cast<std::size_t>(n_faces());
    const auto nif = static_cast<std::size_t>(n_internal_faces());
    for (std::size_t f = 0; f < nfi; ++f) {
        ++cell_face_offsets_[static_cast<std::size_t>(owner_[f]) + 1];
        if (f < nif) {
            ++cell_face_offsets_[static_cast<std::size_t>(neighbour_[f]) + 1];
        }
    }
    for (std::size_t c = 0; c < nc; ++c) {
        cell_face_offsets_[c + 1] += cell_face_offsets_[c];
    }
    cell_faces_.assign(static_cast<std::size_t>(cell_face_offsets_.back()), 0);
    std::vector<Label> fill(cell_face_offsets_.begin(), cell_face_offsets_.end() - 1);
    for (std::size_t f = 0; f < nfi; ++f) {
        const auto o = static_cast<std::size_t>(owner_[f]);
        cell_faces_[static_cast<std::size_t>(fill[o]++)] = static_cast<Label>(f);
        if (f < nif) {
            const auto n = static_cast<std::size_t>(neighbour_[f]);
            cell_faces_[static_cast<std::size_t>(fill[n]++)] = static_cast<Label>(f);
        }
    }
}

const std::vector<Label>& Mesh::cell_face_offsets() const {
    if (cell_face_offsets_.empty()) {
        build_cell_faces();
    }
    return cell_face_offsets_;
}

const std::vector<Label>& Mesh::cell_faces() const {
    if (cell_face_offsets_.empty()) {
        build_cell_faces();
    }
    return cell_faces_;
}

std::string to_string(PatchType type) {
    switch (type) {
    case PatchType::Wall:
        return "wall";
    case PatchType::Inlet:
        return "inlet";
    case PatchType::Outlet:
        return "outlet";
    case PatchType::Symmetry:
        return "symmetry";
    case PatchType::Patch:
        break;
    }
    return "patch";
}

PatchType patch_type_from_string(const std::string& name) {
    if (name == "wall") {
        return PatchType::Wall;
    }
    if (name == "inlet") {
        return PatchType::Inlet;
    }
    if (name == "outlet") {
        return PatchType::Outlet;
    }
    if (name == "symmetry") {
        return PatchType::Symmetry;
    }
    return PatchType::Patch;
}

} // namespace cfd::mesh
