#include <algorithm>
#include <cfd/parallel/comm.hpp>
#include <stdexcept>

#ifdef CFD_HAS_MPI
#include <mpi.h>
#endif

namespace cfd::parallel {

namespace {

#ifdef CFD_HAS_MPI
bool mpi_running() {
    int initialised = 0;
    int finalised = 0;
    MPI_Initialized(&initialised);
    MPI_Finalized(&finalised);
    return initialised != 0 && finalised == 0;
}

MPI_Datatype label_type() {
    static_assert(sizeof(Label) == sizeof(int) || sizeof(Label) == sizeof(long long));
    return sizeof(Label) == sizeof(int) ? MPI_INT : MPI_LONG_LONG;
}
#endif

template <typename T>
std::vector<T> gather_impl(const std::vector<T>& local, bool mpi, int rank, int size,
                           [[maybe_unused]] int type_tag) {
    if (!mpi || size == 1) {
        return local;
    }
#ifdef CFD_HAS_MPI
    const MPI_Datatype type = type_tag == 0 ? MPI_DOUBLE : label_type();
    const int n = static_cast<int>(local.size());
    std::vector<int> counts(static_cast<std::size_t>(size));
    MPI_Gather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> displs(static_cast<std::size_t>(size), 0);
    std::vector<T> all;
    if (rank == 0) {
        for (int r = 1; r < size; ++r) {
            displs[static_cast<std::size_t>(r)] =
                displs[static_cast<std::size_t>(r - 1)] + counts[static_cast<std::size_t>(r - 1)];
        }
        all.resize(static_cast<std::size_t>(displs.back() + counts.back()));
    }
    MPI_Gatherv(local.data(), n, type, all.data(), counts.data(), displs.data(), type, 0,
                MPI_COMM_WORLD);
    return all;
#else
    (void)rank;
    return local;
#endif
}

} // namespace

const Comm& Comm::world() {
#ifdef CFD_HAS_MPI
    if (mpi_running()) {
        static const Comm comm = [] {
            int rank = 0;
            int size = 1;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            return Comm(rank, size, true);
        }();
        return comm;
    }
#endif
    return self();
}

const Comm& Comm::self() {
    static const Comm comm(0, 1, false);
    return comm;
}

Scalar Comm::sum(Scalar value) const {
    sum(&value, 1);
    return value;
}

void Comm::sum([[maybe_unused]] Scalar* values, [[maybe_unused]] std::size_t n) const {
#ifdef CFD_HAS_MPI
    if (mpi_ && size_ > 1) {
        MPI_Allreduce(MPI_IN_PLACE, values, static_cast<int>(n), MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
    }
#endif
}

Label Comm::sum(Label value) const {
#ifdef CFD_HAS_MPI
    if (mpi_ && size_ > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, label_type(), MPI_SUM, MPI_COMM_WORLD);
    }
#endif
    return value;
}

Scalar Comm::max(Scalar value) const {
#ifdef CFD_HAS_MPI
    if (mpi_ && size_ > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
#endif
    return value;
}

Scalar Comm::min(Scalar value) const {
#ifdef CFD_HAS_MPI
    if (mpi_ && size_ > 1) {
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    }
#endif
    return value;
}

void Comm::barrier() const {
#ifdef CFD_HAS_MPI
    if (mpi_ && size_ > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
}

void Comm::broadcast([[maybe_unused]] std::string& text) const {
#ifdef CFD_HAS_MPI
    if (mpi_ && size_ > 1) {
        unsigned long long n = text.size();
        MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        text.resize(static_cast<std::size_t>(n));
        MPI_Bcast(text.data(), static_cast<int>(n), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
#endif
}

std::vector<Scalar> Comm::gather(const std::vector<Scalar>& local) const {
    return gather_impl(local, mpi_, rank_, size_, 0);
}

std::vector<Label> Comm::gather(const std::vector<Label>& local) const {
    return gather_impl(local, mpi_, rank_, size_, 1);
}

void Comm::exchange(const std::vector<int>& peers,
                    [[maybe_unused]] const std::vector<std::vector<char>>& send,
                    [[maybe_unused]] std::vector<std::vector<char>>& recv) const {
    if (peers.empty()) {
        return;
    }
#ifdef CFD_HAS_MPI
    if (mpi_ && size_ > 1) {
        std::vector<MPI_Request> requests(2 * peers.size());
        for (std::size_t i = 0; i < peers.size(); ++i) {
            MPI_Irecv(recv[i].data(), static_cast<int>(recv[i].size()), MPI_BYTE, peers[i], 7,
                      MPI_COMM_WORLD, &requests[i]);
        }
        for (std::size_t i = 0; i < peers.size(); ++i) {
            // MPI-2 signatures take a non-const buffer.
            MPI_Isend(const_cast<char*>(send[i].data()), static_cast<int>(send[i].size()), MPI_BYTE,
                      peers[i], 7, MPI_COMM_WORLD, &requests[peers.size() + i]);
        }
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        return;
    }
#endif
    throw std::logic_error("Comm::exchange: peers without a parallel communicator");
}

Environment::Environment([[maybe_unused]] int& argc, [[maybe_unused]] char**& argv) {
#ifdef CFD_HAS_MPI
    int initialised = 0;
    MPI_Initialized(&initialised);
    if (initialised == 0) {
        MPI_Init(&argc, &argv);
        initialised_ = true;
    }
#endif
}

Environment::~Environment() {
#ifdef CFD_HAS_MPI
    if (initialised_) {
        MPI_Finalize();
    }
#endif
}

bool Environment::mpi_available() {
#ifdef CFD_HAS_MPI
    return true;
#else
    return false;
#endif
}

} // namespace cfd::parallel
