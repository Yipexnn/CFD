#include <algorithm>
#include <cfd/parallel/decomposition.hpp>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace cfd::parallel {

namespace {

using core::Vec3;

void bisect(const std::vector<Vec3>& centres, std::vector<Label>& cells, std::size_t begin,
            std::size_t end, int first_part, int n_parts, std::vector<int>& parts) {
    if (n_parts <= 1 || end - begin <= 1) {
        for (std::size_t i = begin; i < end; ++i) {
            parts[static_cast<std::size_t>(cells[i])] = first_part;
        }
        return;
    }
    Vec3 lo = Vec3::Constant(std::numeric_limits<Scalar>::max());
    Vec3 hi = -lo;
    for (std::size_t i = begin; i < end; ++i) {
        const Vec3& c = centres[static_cast<std::size_t>(cells[i])];
        lo = lo.cwiseMin(c);
        hi = hi.cwiseMax(c);
    }
    Eigen::Index axis = 0;
    (hi - lo).maxCoeff(&axis);
    const int left_parts = n_parts / 2;
    const std::size_t n = end - begin;
    const std::size_t n_left =
        (n * static_cast<std::size_t>(left_parts) + static_cast<std::size_t>(n_parts) / 2) /
        static_cast<std::size_t>(n_parts);
    const auto first = cells.begin() + static_cast<std::ptrdiff_t>(begin);
    std::nth_element(first, first + static_cast<std::ptrdiff_t>(n_left),
                     cells.begin() + static_cast<std::ptrdiff_t>(end), [&](Label a, Label b) {
                         const Scalar ca = centres[static_cast<std::size_t>(a)][axis];
                         const Scalar cb = centres[static_cast<std::size_t>(b)][axis];
                         return ca < cb || (ca == cb && a < b);
                     });
    bisect(centres, cells, begin, begin + n_left, first_part, left_parts, parts);
    bisect(centres, cells, begin + n_left, end, first_part + left_parts, n_parts - left_parts,
           parts);
}

} // namespace

std::vector<int> partition_points(const std::vector<Vec3>& points, int n_parts) {
    if (n_parts < 1) {
        throw std::invalid_argument("partition_rcb: n_parts must be >= 1");
    }
    const std::size_t n = points.size();
    std::vector<int> parts(n, 0);
    std::vector<Label> cells(n);
    std::iota(cells.begin(), cells.end(), Label{0});
    bisect(points, cells, 0, n, 0, n_parts, parts);
    return parts;
}

std::vector<int> partition_rcb(const mesh::Mesh& mesh, int n_parts) {
    return partition_points(mesh.cell_centres(), n_parts);
}

LocalMesh decompose(const mesh::Mesh& global, const std::vector<int>& parts, int part,
                    const Comm& comm) {
    const auto nc = static_cast<std::size_t>(global.n_cells());
    const auto nif = static_cast<std::size_t>(global.n_internal_faces());
    if (parts.size() != nc) {
        throw std::invalid_argument("decompose: parts size mismatch");
    }
    const auto& own = global.owner();
    const auto& nei = global.neighbour();
    const auto& cell_offsets = global.cell_face_offsets();
    const auto& cell_faces = global.cell_faces();
    auto other_cell = [&](std::size_t f, Label c) { return own[f] == c ? nei[f] : own[f]; };

    // Local cell numbering: owned (ascending), then ghosts sorted by (owning part, index).
    std::vector<Label> local_of(nc, -1);
    LocalMesh lm;
    for (std::size_t c = 0; c < nc; ++c) {
        if (parts[c] == part) {
            local_of[c] = static_cast<Label>(lm.global_cell.size());
            lm.global_cell.push_back(static_cast<Label>(c));
        }
    }
    const auto n_owned = static_cast<Label>(lm.global_cell.size());
    std::vector<Label> ghosts;
    for (Label i = 0; i < n_owned; ++i) {
        const Label c = lm.global_cell[static_cast<std::size_t>(i)];
        for (Label k = cell_offsets[static_cast<std::size_t>(c)];
             k < cell_offsets[static_cast<std::size_t>(c) + 1]; ++k) {
            const auto f = static_cast<std::size_t>(cell_faces[static_cast<std::size_t>(k)]);
            if (f < nif) {
                const Label o = other_cell(f, c);
                if (parts[static_cast<std::size_t>(o)] != part) {
                    ghosts.push_back(o);
                }
            }
        }
    }
    std::sort(ghosts.begin(), ghosts.end(), [&](Label a, Label b) {
        return std::make_pair(parts[static_cast<std::size_t>(a)], a) <
               std::make_pair(parts[static_cast<std::size_t>(b)], b);
    });
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    for (const Label g : ghosts) {
        local_of[static_cast<std::size_t>(g)] = static_cast<Label>(lm.global_cell.size());
        lm.global_cell.push_back(g);
    }

    // Faces: (owner, neighbour, global face, flipped).
    struct LocalFace {
        Label owner;
        Label neighbour;
        Label global;
        bool flip;
    };
    std::vector<LocalFace> internal;
    std::vector<LocalFace> halo;
    for (std::size_t f = 0; f < nif; ++f) {
        const Label lo = local_of[static_cast<std::size_t>(own[f])];
        const Label ln = local_of[static_cast<std::size_t>(nei[f])];
        if (lo >= 0 && ln >= 0) {
            if (lo < ln) {
                internal.push_back({lo, ln, static_cast<Label>(f), false});
            } else {
                internal.push_back({ln, lo, static_cast<Label>(f), true});
            }
        } else if (lo >= 0) {
            halo.push_back({lo, -1, static_cast<Label>(f), false});
        } else if (ln >= 0) {
            halo.push_back({ln, -1, static_cast<Label>(f), true});
        }
    }
    std::sort(internal.begin(), internal.end(), [](const LocalFace& a, const LocalFace& b) {
        return std::tie(a.owner, a.neighbour) < std::tie(b.owner, b.neighbour);
    });
    std::vector<LocalFace> faces = internal;
    std::vector<mesh::Patch> patches;
    for (const mesh::Patch& gp : global.patches()) {
        mesh::Patch p = gp;
        p.start_face = static_cast<Label>(faces.size());
        for (Label i = 0; i < gp.n_faces; ++i) {
            const auto f = static_cast<std::size_t>(gp.start_face + i);
            const Label lo = local_of[static_cast<std::size_t>(own[f])];
            if (lo >= 0) {
                faces.push_back({lo, -1, static_cast<Label>(f), false});
            }
        }
        p.n_faces = static_cast<Label>(faces.size()) - p.start_face;
        patches.push_back(p);
    }
    mesh::Patch hp;
    hp.name = k_halo_patch;
    hp.type = mesh::PatchType::Patch;
    hp.start_face = static_cast<Label>(faces.size());
    hp.n_faces = static_cast<Label>(halo.size());
    faces.insert(faces.end(), halo.begin(), halo.end());
    patches.push_back(hp);

    // Points and face connectivity.
    std::vector<Label> point_of(static_cast<std::size_t>(global.n_points()), -1);
    std::vector<Vec3> points;
    std::vector<Label> offsets{0};
    std::vector<Label> face_points;
    std::vector<Label> owner;
    std::vector<Label> neighbour;
    const std::size_t first_halo = faces.size() - halo.size();
    for (std::size_t i = 0; i < faces.size(); ++i) {
        const LocalFace& lf = faces[i];
        const auto gpts = global.face(lf.global);
        const std::size_t start = face_points.size();
        for (const Label gp : gpts) {
            Label& lp = point_of[static_cast<std::size_t>(gp)];
            if (lp < 0) {
                lp = static_cast<Label>(points.size());
                points.push_back(global.points()[static_cast<std::size_t>(gp)]);
            }
            face_points.push_back(lp);
        }
        if (lf.flip) {
            std::reverse(face_points.begin() + static_cast<std::ptrdiff_t>(start),
                         face_points.end());
        }
        offsets.push_back(static_cast<Label>(face_points.size()));
        owner.push_back(lf.owner);
        if (lf.neighbour >= 0) {
            neighbour.push_back(lf.neighbour);
        }
        lm.global_face.push_back(i >= first_halo ? -1 : lf.global);
        lm.face_flipped.push_back(lf.flip ? 1 : 0);
    }
    lm.mesh = mesh::Mesh(std::move(points), std::move(offsets), std::move(face_points),
                         std::move(owner), std::move(neighbour), std::move(patches),
                         static_cast<Label>(lm.global_cell.size()));

    // Halo exchange lists.
    std::map<int, Halo::Neighbour> by_rank;
    for (const Label g : ghosts) {
        const int q = parts[static_cast<std::size_t>(g)];
        by_rank[q].rank = q;
        by_rank[q].recv.push_back(local_of[static_cast<std::size_t>(g)]);
    }
    std::map<int, std::vector<Label>> send_global;
    for (Label i = 0; i < n_owned; ++i) {
        const Label c = lm.global_cell[static_cast<std::size_t>(i)];
        for (Label k = cell_offsets[static_cast<std::size_t>(c)];
             k < cell_offsets[static_cast<std::size_t>(c) + 1]; ++k) {
            const auto f = static_cast<std::size_t>(cell_faces[static_cast<std::size_t>(k)]);
            if (f < nif) {
                const int q = parts[static_cast<std::size_t>(other_cell(f, c))];
                if (q != part) {
                    send_global[q].push_back(c);
                }
            }
        }
    }
    for (auto& [q, cells] : send_global) {
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        by_rank[q].rank = q;
        for (const Label c : cells) {
            by_rank[q].send.push_back(local_of[static_cast<std::size_t>(c)]);
        }
    }
    lm.halo.comm = &comm;
    lm.halo.n_owned = n_owned;
    lm.halo.halo_patch = static_cast<int>(lm.mesh.patches().size()) - 1;
    for (auto& [q, nb] : by_rank) {
        lm.halo.neighbours.push_back(std::move(nb));
    }
    return lm;
}

SubMesh extract_zone(const mesh::Mesh& parent, const std::vector<int>& zone_of_cell,
                     const std::vector<std::string>& zone_names, int zone,
                     const Halo* parent_halo) {
    const auto nc = static_cast<std::size_t>(parent.n_cells());
    const auto nif = static_cast<std::size_t>(parent.n_internal_faces());
    if (zone_of_cell.size() != nc) {
        throw std::invalid_argument("extract_zone: zone size mismatch");
    }
    SubMesh sm;
    std::vector<Label> sub_of(nc, -1);
    for (std::size_t c = 0; c < nc; ++c) {
        if (zone_of_cell[c] == zone) {
            sub_of[c] = static_cast<Label>(sm.parent_cell.size());
            sm.parent_cell.push_back(static_cast<Label>(c));
        }
    }
    const auto& own = parent.owner();
    const auto& nei = parent.neighbour();
    struct SubFace {
        Label owner;
        Label neighbour;
        Label parent;
        bool flip;
    };
    std::vector<SubFace> faces;
    std::vector<std::vector<SubFace>> interface(zone_names.size());
    for (std::size_t f = 0; f < nif; ++f) {
        const Label so = sub_of[static_cast<std::size_t>(own[f])];
        const Label sn = sub_of[static_cast<std::size_t>(nei[f])];
        if (so >= 0 && sn >= 0) {
            faces.push_back({so, sn, static_cast<Label>(f), false}); // order preserved
        } else if (so >= 0) {
            interface[static_cast<std::size_t>(zone_of_cell[static_cast<std::size_t>(nei[f])])]
                .push_back({so, -1, static_cast<Label>(f), false});
        } else if (sn >= 0) {
            interface[static_cast<std::size_t>(zone_of_cell[static_cast<std::size_t>(own[f])])]
                .push_back({sn, -1, static_cast<Label>(f), true});
        }
    }
    std::vector<mesh::Patch> patches;
    for (const mesh::Patch& pp : parent.patches()) {
        mesh::Patch p = pp;
        p.start_face = static_cast<Label>(faces.size());
        for (Label i = 0; i < pp.n_faces; ++i) {
            const auto f = static_cast<std::size_t>(pp.start_face + i);
            const Label so = sub_of[static_cast<std::size_t>(own[f])];
            if (so >= 0) {
                faces.push_back({so, -1, static_cast<Label>(f), false});
            }
        }
        p.n_faces = static_cast<Label>(faces.size()) - p.start_face;
        patches.push_back(p);
    }
    for (std::size_t z = 0; z < zone_names.size(); ++z) {
        if (static_cast<int>(z) == zone) {
            continue;
        }
        mesh::Patch p;
        p.name = zone_names[z];
        p.type = mesh::PatchType::Wall;
        p.start_face = static_cast<Label>(faces.size());
        p.n_faces = static_cast<Label>(interface[z].size());
        faces.insert(faces.end(), interface[z].begin(), interface[z].end());
        patches.push_back(p);
    }

    std::vector<Label> point_of(static_cast<std::size_t>(parent.n_points()), -1);
    std::vector<Vec3> points;
    std::vector<Label> offsets{0};
    std::vector<Label> face_points;
    std::vector<Label> owner;
    std::vector<Label> neighbour;
    for (const SubFace& sf : faces) {
        const std::size_t start = face_points.size();
        for (const Label gp : parent.face(sf.parent)) {
            Label& lp = point_of[static_cast<std::size_t>(gp)];
            if (lp < 0) {
                lp = static_cast<Label>(points.size());
                points.push_back(parent.points()[static_cast<std::size_t>(gp)]);
            }
            face_points.push_back(lp);
        }
        if (sf.flip) {
            std::reverse(face_points.begin() + static_cast<std::ptrdiff_t>(start),
                         face_points.end());
        }
        offsets.push_back(static_cast<Label>(face_points.size()));
        owner.push_back(sf.owner);
        if (sf.neighbour >= 0) {
            neighbour.push_back(sf.neighbour);
        }
        sm.parent_face.push_back(sf.parent);
        sm.face_flipped.push_back(sf.flip ? 1 : 0);
    }
    sm.mesh = mesh::Mesh(std::move(points), std::move(offsets), std::move(face_points),
                         std::move(owner), std::move(neighbour), std::move(patches),
                         static_cast<Label>(sm.parent_cell.size()));

    if (parent_halo != nullptr) {
        sm.halo.comm = parent_halo->comm;
        sm.halo.halo_patch = parent_halo->halo_patch;
        Label n_owned = 0;
        for (Label c = 0; c < parent_halo->n_owned; ++c) {
            n_owned += sub_of[static_cast<std::size_t>(c)] >= 0 ? 1 : 0;
        }
        sm.halo.n_owned = n_owned;
        for (const Halo::Neighbour& nb : parent_halo->neighbours) {
            Halo::Neighbour s;
            s.rank = nb.rank;
            for (const Label c : nb.send) {
                if (sub_of[static_cast<std::size_t>(c)] >= 0) {
                    s.send.push_back(sub_of[static_cast<std::size_t>(c)]);
                }
            }
            for (const Label c : nb.recv) {
                if (sub_of[static_cast<std::size_t>(c)] >= 0) {
                    s.recv.push_back(sub_of[static_cast<std::size_t>(c)]);
                }
            }
            if (!s.send.empty() || !s.recv.empty()) {
                sm.halo.neighbours.push_back(std::move(s));
            }
        }
    } else {
        sm.halo.n_owned = sm.mesh.n_cells();
    }
    return sm;
}

} // namespace cfd::parallel
