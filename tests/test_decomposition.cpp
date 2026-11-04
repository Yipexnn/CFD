#include <algorithm>
#include <cfd/mesh/box_mesher.hpp>
#include <cfd/parallel/decomposition.hpp>
#include <gtest/gtest.h>
#include <map>
#include <vector>

using cfd::core::Label;
using cfd::core::Vec3;
namespace par = cfd::parallel;

namespace {

cfd::mesh::Mesh test_mesh() {
    cfd::mesh::BoxSpec spec;
    spec.nx = 9;
    spec.ny = 7;
    spec.nz = 5;
    spec.max = Vec3(2.0, 1.0, 0.5);
    spec.grading = Vec3(2.0, 1.0, 0.5);
    return cfd::mesh::make_box_mesh(spec);
}

class Decomposition : public ::testing::TestWithParam<int> {};

TEST_P(Decomposition, PartitionIsBalancedAndComplete) {
    const auto mesh = test_mesh();
    const int n = GetParam();
    const std::vector<int> parts = par::partition_rcb(mesh, n);
    std::vector<int> count(static_cast<std::size_t>(n), 0);
    for (const int p : parts) {
        ASSERT_GE(p, 0);
        ASSERT_LT(p, n);
        ++count[static_cast<std::size_t>(p)];
    }
    const auto [lo, hi] = std::minmax_element(count.begin(), count.end());
    EXPECT_LE(*hi - *lo, n);
    EXPECT_EQ(par::partition_rcb(mesh, n), parts); // deterministic
}

TEST_P(Decomposition, LocalMeshesAreConsistent) {
    const auto global = test_mesh();
    const int n = GetParam();
    const std::vector<int> parts = par::partition_rcb(global, n);
    std::vector<par::LocalMesh> locals;
    for (int p = 0; p < n; ++p) {
        locals.push_back(par::decompose(global, parts, p, par::Comm::self()));
    }

    std::vector<int> owned_count(static_cast<std::size_t>(global.n_cells()), 0);
    std::map<std::string, Label> boundary_faces;
    for (int p = 0; p < n; ++p) {
        const par::LocalMesh& lm = locals[static_cast<std::size_t>(p)];
        const auto& m = lm.mesh;
        const Label n_owned = lm.halo.n_owned;
        ASSERT_EQ(lm.global_cell.size(), static_cast<std::size_t>(m.n_cells()));
        for (Label c = 0; c < m.n_cells(); ++c) {
            const Label g = lm.global_cell[static_cast<std::size_t>(c)];
            if (c < n_owned) {
                EXPECT_EQ(parts[static_cast<std::size_t>(g)], p);
                ++owned_count[static_cast<std::size_t>(g)];
            } else {
                EXPECT_NE(parts[static_cast<std::size_t>(g)], p);
            }
            // Ghost cells have all their faces, so their geometry is exact too.
            EXPECT_NEAR(m.cell_volumes()[static_cast<std::size_t>(c)],
                        global.cell_volumes()[static_cast<std::size_t>(g)], 1e-14);
            EXPECT_TRUE(m.cell_centres()[static_cast<std::size_t>(c)].isApprox(
                global.cell_centres()[static_cast<std::size_t>(g)], 1e-12));
        }
        // Patches: global patches in order plus the halo patch of ghost faces.
        ASSERT_EQ(m.patches().size(), global.patches().size() + 1);
        EXPECT_EQ(lm.halo.halo_patch, static_cast<int>(global.patches().size()));
        const auto& hp = m.patches().back();
        EXPECT_EQ(hp.name, par::k_halo_patch);
        for (Label i = 0; i < hp.n_faces; ++i) {
            EXPECT_GE(m.owner()[static_cast<std::size_t>(hp.start_face + i)], n_owned);
            EXPECT_EQ(lm.global_face[static_cast<std::size_t>(hp.start_face + i)], -1);
        }
        for (std::size_t q = 0; q < global.patches().size(); ++q) {
            const auto& lp = m.patches()[q];
            for (Label i = 0; i < lp.n_faces; ++i) {
                const auto f = static_cast<std::size_t>(lp.start_face + i);
                if (m.owner()[f] < n_owned) {
                    ++boundary_faces[lp.name];
                }
                const Label gf = lm.global_face[f];
                EXPECT_TRUE(m.face_area_vectors()[f].isApprox(
                    global.face_area_vectors()[static_cast<std::size_t>(gf)], 1e-12));
            }
        }
        // Internal faces: orientation consistent with the local owner -> neighbour.
        for (Label f = 0; f < m.n_internal_faces(); ++f) {
            const auto fs = static_cast<std::size_t>(f);
            const Vec3 d = m.cell_centres()[static_cast<std::size_t>(m.neighbour()[fs])] -
                           m.cell_centres()[static_cast<std::size_t>(m.owner()[fs])];
            EXPECT_GT(d.dot(m.face_area_vectors()[fs]), 0.0);
        }
    }
    for (const int c : owned_count) {
        EXPECT_EQ(c, 1);
    }
    for (const auto& gp : global.patches()) {
        EXPECT_EQ(boundary_faces[gp.name], gp.n_faces) << gp.name;
    }

    // Send lists of p to q match the receive lists of q from p (as global cells).
    for (int p = 0; p < n; ++p) {
        const auto& lp = locals[static_cast<std::size_t>(p)];
        for (const auto& nb : lp.halo.neighbours) {
            const auto& lq = locals[static_cast<std::size_t>(nb.rank)];
            const auto it =
                std::find_if(lq.halo.neighbours.begin(), lq.halo.neighbours.end(),
                             [&](const par::Halo::Neighbour& x) { return x.rank == p; });
            ASSERT_NE(it, lq.halo.neighbours.end());
            ASSERT_EQ(nb.send.size(), it->recv.size());
            for (std::size_t k = 0; k < nb.send.size(); ++k) {
                EXPECT_LT(nb.send[k], lp.halo.n_owned);
                EXPECT_EQ(lp.global_cell[static_cast<std::size_t>(nb.send[k])],
                          lq.global_cell[static_cast<std::size_t>(it->recv[k])]);
            }
        }
        // Every ghost is received exactly once.
        std::vector<int> received(lp.global_cell.size(), 0);
        for (const auto& nb : lp.halo.neighbours) {
            for (const Label c : nb.recv) {
                ++received[static_cast<std::size_t>(c)];
            }
        }
        for (std::size_t c = 0; c < received.size(); ++c) {
            EXPECT_EQ(received[c], static_cast<Label>(c) < lp.halo.n_owned ? 0 : 1);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Parts, Decomposition, ::testing::Values(1, 2, 3, 4, 7));

} // namespace
