#pragma once

// Backends of gpu::DeviceFv (OpenMP host / offload, CUDA).
#include <cfd/gpu/fv_device.hpp>
#include <memory>
#include <vector>

namespace cfd::gpu {

struct DeviceFv::Impl {
    Impl() = default;
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    virtual ~Impl() = default;
    virtual void gradient(const double* phi, const double* bv, bool least_squares,
                          double* grad) = 0;
    virtual void vector_gradient(const double* u, const double* ub, bool least_squares,
                                 double* grad) = 0;
    virtual void scalar_corrections(const double* phi, const double* bv, bool least_squares,
                                    const double* flux, const double* gamma, bool linear_upwind,
                                    double* source) = 0;
    virtual void momentum_corrections(const double* flux, const double* nu_face, bool linear_upwind,
                                      bool limited, bool venkat, const double* eps2,
                                      bool transpose_term, double* source) = 0;
    [[nodiscard]] virtual bool on_device() const = 0;
};

namespace detail {

// Owned host copy of an FvMeshView (sizes checked).
struct FvHostMesh {
    int nc = 0;
    int nif = 0;
    int nf = 0;
    std::vector<int> own;
    std::vector<int> nei;
    std::vector<int> start;
    std::vector<int> faces;
    std::vector<double> w;
    std::vector<double> sf;
    std::vector<double> cf;
    std::vector<double> cc;
    std::vector<double> vol;
    std::vector<double> kv;
    std::vector<double> lsq_d;
    std::vector<double> lsq;

    explicit FvHostMesh(const FvMeshView& m);
    [[nodiscard]] int nb() const { return nf - nif; }
};

std::unique_ptr<DeviceFv::Impl> make_fv_openmp(const FvMeshView& mesh);
#ifdef CFD_GPU_CUDA
std::unique_ptr<DeviceFv::Impl> make_fv_cuda(const FvMeshView& mesh);
#endif

} // namespace detail

} // namespace cfd::gpu
