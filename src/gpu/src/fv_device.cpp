#include "fv_impl.hpp"

#include <cfd/gpu/krylov.hpp>
#include <stdexcept>
#include <string>

namespace cfd::gpu {

namespace detail {

namespace {

template <typename T> std::vector<T> copy(const T* p, std::size_t n, const char* what) {
    if (n > 0 && p == nullptr) {
        throw std::invalid_argument(std::string("DeviceFv: missing mesh array ") + what);
    }
    return n > 0 ? std::vector<T>(p, p + n) : std::vector<T>{};
}

} // namespace

FvHostMesh::FvHostMesh(const FvMeshView& m)
    : nc(m.n_cells), nif(m.n_internal_faces), nf(m.n_faces) {
    if (nc <= 0 || nif < 0 || nf < nif || m.cell_offsets == nullptr) {
        throw std::invalid_argument("DeviceFv: invalid mesh sizes");
    }
    const auto c = static_cast<std::size_t>(nc);
    const auto f = static_cast<std::size_t>(nf);
    const auto fi = static_cast<std::size_t>(nif);
    own = copy(m.owner, f, "owner");
    nei = copy(m.neighbour, fi, "neighbour");
    start = copy(m.cell_offsets, c + 1, "cell_offsets");
    faces = copy(m.cell_faces, static_cast<std::size_t>(start.back()), "cell_faces");
    w = copy(m.weight, fi, "weight");
    sf = copy(m.sf, 3 * f, "sf");
    cf = copy(m.face_centre, 3 * f, "face_centre");
    cc = copy(m.cell_centre, 3 * c, "cell_centre");
    vol = copy(m.volume, c, "volume");
    kv = copy(m.correction, 3 * f, "correction");
    lsq_d = copy(m.lsq_d, 3 * f, "lsq_d");
    lsq = copy(m.lsq_inverse, 9 * c, "lsq_inverse");
    // Device kernels index nei[] only for internal faces; pad so empty arrays have storage.
    if (nei.empty()) {
        nei.push_back(0);
        w.push_back(0.0);
    }
}

} // namespace detail

DeviceFv::DeviceFv(const FvMeshView& mesh) {
#ifdef CFD_GPU_CUDA
    if (detail::cuda_device_available()) {
        impl_ = detail::make_fv_cuda(mesh);
        return;
    }
#endif
    impl_ = detail::make_fv_openmp(mesh);
}

DeviceFv::~DeviceFv() = default;

void DeviceFv::gradient(const double* phi, const double* bv, int scheme, double* grad) {
    impl_->gradient(phi, bv, scheme, grad);
}

void DeviceFv::vector_gradient(const double* u, const double* ub, int scheme, double* grad) {
    impl_->vector_gradient(u, ub, scheme, grad);
}

void DeviceFv::scalar_corrections(const double* phi, const double* bv, int scheme,
                                  const double* flux, const double* gamma, bool linear_upwind,
                                  double* source) {
    impl_->scalar_corrections(phi, bv, scheme, flux, gamma, linear_upwind, source);
}

void DeviceFv::momentum_corrections(const double* flux, const double* nu_face, bool linear_upwind,
                                    bool limited, bool venkat, const double* eps2, double central,
                                    bool transpose_term, double* source) {
    impl_->momentum_corrections(flux, nu_face, linear_upwind, limited, venkat, eps2, central,
                                transpose_term, source);
}

bool DeviceFv::on_device() const {
    return impl_->on_device();
}

} // namespace cfd::gpu
