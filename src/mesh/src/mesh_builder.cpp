#include <algorithm>
#include <cfd/mesh/mesh_builder.hpp>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace cfd::mesh {

namespace {

// Orientation-independent face key: sorted node indices, padded with -1.
using FaceKey = std::array<Label, 4>;

FaceKey make_key(const std::array<Label, 4>& nodes, int n) {
    FaceKey key{-1, -1, -1, -1};
    // Insertion sort over at most 4 entries (std::sort triggers a GCC 15 false-positive
    // -Warray-bounds on such small fixed-size ranges).
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        const Label v = nodes[i];
        std::size_t j = i;
        for (; j > 0 && key[j - 1] > v; --j) {
            key[j] = key[j - 1];
        }
        key[j] = v;
    }
    return key;
}

struct FaceKeyHash {
    std::size_t operator()(const FaceKey& k) const noexcept {
        std::uint64_t h = 1469598103934665603ULL;
        for (const Label v : k) {
            h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(v));
            h *= 1099511628211ULL;
        }
        return static_cast<std::size_t>(h);
    }
};

struct FaceRecord {
    std::array<Label, 4> nodes{}; // oriented outward from cell0
    int n_nodes = 0;
    Label cell0 = -1;
    Label cell1 = -1;
};

} // namespace

MeshBuilder::MeshBuilder(std::vector<Vec3> points) : points_(std::move(points)) {}

Label MeshBuilder::add_cell(CellShape shape, std::span<const Label> nodes) {
    if (static_cast<int>(nodes.size()) != n_nodes(shape)) {
        throw std::invalid_argument("MeshBuilder: wrong node count for cell shape");
    }
    for (const Label n : nodes) {
        if (n < 0 || n >= static_cast<Label>(points_.size())) {
            throw std::invalid_argument("MeshBuilder: cell node index out of range");
        }
    }
    cell_shapes_.push_back(shape);
    cell_nodes_.insert(cell_nodes_.end(), nodes.begin(), nodes.end());
    cell_node_offsets_.push_back(static_cast<Label>(cell_nodes_.size()));
    return static_cast<Label>(cell_shapes_.size()) - 1;
}

Label MeshBuilder::add_patch(const std::string& name, PatchType type) {
    for (std::size_t i = 0; i < patches_.size(); ++i) {
        if (patches_[i].name == name) {
            patches_[i].type = type;
            return static_cast<Label>(i);
        }
    }
    patches_.push_back({name, type});
    return static_cast<Label>(patches_.size()) - 1;
}

void MeshBuilder::add_boundary_face(std::span<const Label> nodes, const std::string& patch) {
    if (nodes.size() < 3 || nodes.size() > 4) {
        throw std::invalid_argument("MeshBuilder: boundary faces must have 3 or 4 nodes");
    }
    Label index = -1;
    for (std::size_t i = 0; i < patches_.size(); ++i) {
        if (patches_[i].name == patch) {
            index = static_cast<Label>(i);
        }
    }
    if (index < 0) {
        index = add_patch(patch);
    }
    TaggedFace face;
    std::copy(nodes.begin(), nodes.end(), face.nodes.begin());
    face.n_nodes = static_cast<int>(nodes.size());
    face.patch = index;
    tagged_faces_.push_back(face);
}

Mesh MeshBuilder::build(const std::string& default_patch) const {
    const Label nc = n_cells();

    // 1. Collect unique faces.
    std::vector<FaceRecord> faces;
    std::unordered_map<FaceKey, Label, FaceKeyHash> lookup;
    faces.reserve(cell_nodes_.size() / 2);
    lookup.reserve(cell_nodes_.size() / 2);

    for (Label c = 0; c < nc; ++c) {
        const auto base = static_cast<std::size_t>(cell_node_offsets_[static_cast<std::size_t>(c)]);
        for (const LocalFace& lf : local_faces(cell_shapes_[static_cast<std::size_t>(c)])) {
            std::array<Label, 4> nodes{-1, -1, -1, -1};
            for (int i = 0; i < lf.n_nodes; ++i) {
                nodes[static_cast<std::size_t>(i)] =
                    cell_nodes_[base +
                                static_cast<std::size_t>(lf.nodes[static_cast<std::size_t>(i)])];
            }
            const FaceKey key = make_key(nodes, lf.n_nodes);
            const auto [it, inserted] = lookup.try_emplace(key, static_cast<Label>(faces.size()));
            if (inserted) {
                faces.push_back({nodes, lf.n_nodes, c, -1});
            } else {
                FaceRecord& rec = faces[static_cast<std::size_t>(it->second)];
                if (rec.cell1 >= 0) {
                    throw std::runtime_error("MeshBuilder: face shared by more than two cells");
                }
                rec.cell1 = c;
            }
        }
    }

    // 2. Patch assignment for boundary faces.
    std::vector<PatchDef> patches = patches_;
    std::vector<Label> face_patch(faces.size(), -1);
    for (const TaggedFace& tf : tagged_faces_) {
        const auto it = lookup.find(make_key(tf.nodes, tf.n_nodes));
        if (it == lookup.end()) {
            throw std::runtime_error("MeshBuilder: tagged boundary face is not a face of any cell");
        }
        if (faces[static_cast<std::size_t>(it->second)].cell1 < 0) {
            face_patch[static_cast<std::size_t>(it->second)] = tf.patch;
        }
        // Tagged internal faces (e.g. baffles) are ignored.
    }
    Label default_index = -1;
    for (std::size_t f = 0; f < faces.size(); ++f) {
        if (faces[f].cell1 < 0 && face_patch[f] < 0) {
            if (default_index < 0) {
                default_index = static_cast<Label>(patches.size());
                patches.push_back({default_patch, PatchType::Wall});
            }
            face_patch[f] = default_index;
        }
    }

    // 3. Order: internal faces by (owner, neighbour), then boundary faces by (patch, owner).
    std::vector<Label> internal;
    std::vector<Label> boundary;
    for (std::size_t f = 0; f < faces.size(); ++f) {
        (faces[f].cell1 >= 0 ? internal : boundary).push_back(static_cast<Label>(f));
    }
    auto owner_of = [&](Label f) {
        const FaceRecord& r = faces[static_cast<std::size_t>(f)];
        return r.cell1 >= 0 ? std::min(r.cell0, r.cell1) : r.cell0;
    };
    auto neighbour_of = [&](Label f) {
        const FaceRecord& r = faces[static_cast<std::size_t>(f)];
        return std::max(r.cell0, r.cell1);
    };
    std::sort(internal.begin(), internal.end(), [&](Label a, Label b) {
        return std::pair(owner_of(a), neighbour_of(a)) < std::pair(owner_of(b), neighbour_of(b));
    });
    std::stable_sort(boundary.begin(), boundary.end(), [&](Label a, Label b) {
        return std::pair(face_patch[static_cast<std::size_t>(a)], owner_of(a)) <
               std::pair(face_patch[static_cast<std::size_t>(b)], owner_of(b));
    });

    // 4. Emit mesh arrays.
    std::vector<Label> offsets{0};
    std::vector<Label> face_points;
    std::vector<Label> owner;
    std::vector<Label> neighbour;
    offsets.reserve(faces.size() + 1);
    face_points.reserve(faces.size() * 4);
    owner.reserve(faces.size());
    neighbour.reserve(internal.size());

    auto emit = [&](Label f, bool flip) {
        const FaceRecord& r = faces[static_cast<std::size_t>(f)];
        const auto n = static_cast<std::size_t>(r.n_nodes);
        if (flip) {
            face_points.push_back(r.nodes[0]);
            for (std::size_t i = n - 1; i >= 1; --i) {
                face_points.push_back(r.nodes[i]);
            }
        } else {
            face_points.insert(face_points.end(), r.nodes.begin(),
                               r.nodes.begin() + static_cast<std::ptrdiff_t>(n));
        }
        offsets.push_back(static_cast<Label>(face_points.size()));
    };

    for (const Label f : internal) {
        const FaceRecord& r = faces[static_cast<std::size_t>(f)];
        const Label o = owner_of(f);
        emit(f, r.cell0 != o); // stored orientation is outward from cell0
        owner.push_back(o);
        neighbour.push_back(neighbour_of(f));
    }

    std::vector<Patch> out_patches;
    out_patches.reserve(patches.size());
    Label start = static_cast<Label>(internal.size());
    for (const PatchDef& p : patches) {
        out_patches.push_back({p.name, p.type, start, 0});
    }
    for (const Label f : boundary) {
        emit(f, false);
        owner.push_back(faces[static_cast<std::size_t>(f)].cell0);
        ++out_patches[static_cast<std::size_t>(face_patch[static_cast<std::size_t>(f)])].n_faces;
    }
    for (Patch& p : out_patches) {
        p.start_face = start;
        start += p.n_faces;
    }

    return {points_,
            std::move(offsets),
            std::move(face_points),
            std::move(owner),
            std::move(neighbour),
            std::move(out_patches),
            nc};
}

} // namespace cfd::mesh
