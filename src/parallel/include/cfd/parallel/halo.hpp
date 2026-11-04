#pragma once

#include <cfd/parallel/comm.hpp>
#include <cstring>
#include <type_traits>
#include <vector>

namespace cfd::parallel {

// Ghost-cell exchange pattern of a decomposed mesh. Local cells [0, n_owned) are owned,
// [n_owned, n_cells) are ghost copies of cells owned by neighbouring processes.
struct Halo {
    struct Neighbour {
        int rank = 0;
        std::vector<Label> send; // local owned cells whose values the neighbour needs
        std::vector<Label> recv; // local ghost cells filled from the neighbour
    };

    const Comm* comm = &Comm::self();
    Label n_owned = 0;
    std::vector<Neighbour> neighbours;
    // Index of the boundary patch made of the ghost cells' outer faces (-1: none).
    int halo_patch = -1;

    [[nodiscard]] bool parallel() const { return comm->parallel(); }

    // Copies owned values into the neighbours' ghost cells. T must be a plain value type
    // that can be copied bytewise (Scalar, Vec3, Matrix3d, ...); values.size() must be the
    // local cell count.
    template <typename T> void exchange(std::vector<T>& values) const {
        static_assert(std::is_standard_layout_v<T> && std::is_trivially_destructible_v<T>);
        if (neighbours.empty()) {
            return;
        }
        std::vector<int> peers;
        std::vector<std::vector<char>> send(neighbours.size());
        std::vector<std::vector<char>> recv(neighbours.size());
        for (std::size_t i = 0; i < neighbours.size(); ++i) {
            const Neighbour& nb = neighbours[i];
            peers.push_back(nb.rank);
            send[i].resize(nb.send.size() * sizeof(T));
            for (std::size_t k = 0; k < nb.send.size(); ++k) {
                std::memcpy(send[i].data() + k * sizeof(T),
                            static_cast<const void*>(&values[static_cast<std::size_t>(nb.send[k])]),
                            sizeof(T));
            }
            recv[i].resize(nb.recv.size() * sizeof(T));
        }
        comm->exchange(peers, send, recv);
        for (std::size_t i = 0; i < neighbours.size(); ++i) {
            const Neighbour& nb = neighbours[i];
            for (std::size_t k = 0; k < nb.recv.size(); ++k) {
                std::memcpy(static_cast<void*>(&values[static_cast<std::size_t>(nb.recv[k])]),
                            recv[i].data() + k * sizeof(T), sizeof(T));
            }
        }
    }

    // Reverse of exchange(): adds the ghost values of the neighbours to the owned cells they
    // copy (e.g. overlapping Schwarz corrections).
    void reverse_add(std::vector<Scalar>& values) const {
        if (neighbours.empty()) {
            return;
        }
        std::vector<int> peers;
        std::vector<std::vector<char>> send(neighbours.size());
        std::vector<std::vector<char>> recv(neighbours.size());
        for (std::size_t i = 0; i < neighbours.size(); ++i) {
            const Neighbour& nb = neighbours[i];
            peers.push_back(nb.rank);
            send[i].resize(nb.recv.size() * sizeof(Scalar));
            for (std::size_t k = 0; k < nb.recv.size(); ++k) {
                std::memcpy(send[i].data() + k * sizeof(Scalar),
                            &values[static_cast<std::size_t>(nb.recv[k])], sizeof(Scalar));
            }
            recv[i].resize(nb.send.size() * sizeof(Scalar));
        }
        comm->exchange(peers, send, recv);
        for (std::size_t i = 0; i < neighbours.size(); ++i) {
            const Neighbour& nb = neighbours[i];
            for (std::size_t k = 0; k < nb.send.size(); ++k) {
                Scalar v = 0.0;
                std::memcpy(&v, recv[i].data() + k * sizeof(Scalar), sizeof(Scalar));
                values[static_cast<std::size_t>(nb.send[k])] += v;
            }
        }
    }
};

} // namespace cfd::parallel
