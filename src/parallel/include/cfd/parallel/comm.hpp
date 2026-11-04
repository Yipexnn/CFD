#pragma once

#include <cfd/core/types.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace cfd::parallel {

using core::Label;
using core::Scalar;

// Thin wrapper around MPI_COMM_WORLD. Without MPI support (CFD_HAS_MPI undefined) or before
// MPI is initialised, it behaves as a single process (rank 0 of 1).
class Comm {
public:
    // Process-wide communicator (MPI_COMM_WORLD when MPI is running, serial otherwise).
    [[nodiscard]] static const Comm& world();
    // Always serial (rank 0 of 1), e.g. for per-rank work inside a parallel run.
    [[nodiscard]] static const Comm& self();

    [[nodiscard]] int rank() const { return rank_; }
    [[nodiscard]] int size() const { return size_; }
    [[nodiscard]] bool is_root() const { return rank_ == 0; }
    [[nodiscard]] bool parallel() const { return size_ > 1; }

    [[nodiscard]] Scalar sum(Scalar value) const;
    void sum(Scalar* values, std::size_t n) const; // in place
    [[nodiscard]] Label sum(Label value) const;
    [[nodiscard]] Scalar max(Scalar value) const;
    [[nodiscard]] Scalar min(Scalar value) const;
    void barrier() const;
    // Broadcasts a string from the root to all processes.
    void broadcast(std::string& text) const;

    // Concatenation of every process' values in rank order (root only; empty elsewhere).
    [[nodiscard]] std::vector<Scalar> gather(const std::vector<Scalar>& local) const;
    [[nodiscard]] std::vector<Label> gather(const std::vector<Label>& local) const;

    // Point-to-point exchange of byte buffers with a set of peers (non-blocking, completed
    // before return). send[i] goes to peers[i]; recv[i] (pre-sized) is filled from peers[i].
    void exchange(const std::vector<int>& peers, const std::vector<std::vector<char>>& send,
                  std::vector<std::vector<char>>& recv) const;

private:
    Comm(int rank, int size, bool use_mpi) : rank_(rank), size_(size), mpi_(use_mpi) {}

    int rank_ = 0;
    int size_ = 1;
    bool mpi_ = false;
};

// MPI_Init / MPI_Finalize guard for main(); a no-op without MPI support.
class Environment {
public:
    Environment(int& argc, char**& argv);
    ~Environment();
    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;
    Environment(Environment&&) = delete;
    Environment& operator=(Environment&&) = delete;

    // True if the library was built with MPI support.
    [[nodiscard]] static bool mpi_available();

private:
    bool initialised_ = false;
};

} // namespace cfd::parallel
