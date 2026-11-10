// gpu::DeviceFv with CUDA kernels: mesh and work buffers stay in device memory, each call
// copies only the fields.
#include "fv_impl.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace cfd::gpu::detail {

namespace fvcuda {
#define CFD_HD __host__ __device__
#include "fv_cell.inc"
#undef CFD_HD
} // namespace fvcuda

namespace {

constexpr int k_fv_block = 128;

void fv_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
    }
}

__global__ void gradient_kernel(fvcuda::FvArrays m, const double* phi, const double* bv, int stride,
                                int comp, bool lsq, double* out, int ostride, int ooff) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < m.nc) {
        fvcuda::cell_gradient(m, c, phi, bv, stride, comp, lsq, out, ostride, ooff);
    }
}

__global__ void corrected_kernel(fvcuda::FvArrays m, const double* phi, const double* bv,
                                 int stride, int comp, const double* g, double* out, int ostride,
                                 int ooff) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < m.nc) {
        fvcuda::cell_gradient_corrected(m, c, phi, bv, stride, comp, g, out, ostride, ooff);
    }
}

__global__ void limiter_kernel(fvcuda::FvArrays m, const double* u, const double* ub,
                               const double* g, bool venkat, double e0, double e1, double e2,
                               double* lim) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < m.nc) {
        fvcuda::cell_limiter(m, c, u, ub, g, venkat, e0, e1, e2, lim);
    }
}

__global__ void scalar_kernel(fvcuda::FvArrays m, const double* g, const double* flux,
                              const double* gamma, bool lu, double* src) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < m.nc) {
        fvcuda::cell_scalar_corrections(m, c, g, flux, gamma, lu, src);
    }
}

__global__ void momentum_kernel(fvcuda::FvArrays m, const double* g, const double* lim,
                                bool limited, const double* flux, const double* nu, bool lu,
                                double central, const double* u, bool transpose_term, double* src) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < m.nc) {
        fvcuda::cell_momentum_corrections(m, c, g, lim, limited, flux, nu, lu, central, u,
                                          transpose_term, src);
    }
}

std::size_t sz(int n) {
    return static_cast<std::size_t>(n > 1 ? n : 1);
}

class CudaFv final : public DeviceFv::Impl {
public:
    explicit CudaFv(const FvMeshView& view) : h_(view) {
        try {
            m_.nc = h_.nc;
            m_.nif = h_.nif;
            m_.nf = h_.nf;
            m_.own = put(h_.own);
            m_.nei = put(h_.nei);
            m_.start = put(h_.start);
            m_.faces = put(h_.faces);
            m_.w = put(h_.w);
            m_.sf = put(h_.sf);
            m_.cf = put(h_.cf);
            m_.cc = put(h_.cc);
            m_.vol = put(h_.vol);
            m_.kv = put(h_.kv);
            m_.lsq_d = put(h_.lsq_d);
            m_.lsq = put(h_.lsq);
            phi_ = alloc(sz(h_.nc));
            bv_ = alloc(sz(h_.nb()));
            sgrad_ = alloc(3 * sz(h_.nc));
            vphi_ = alloc(3 * sz(h_.nc));
            vbv_ = alloc(3 * sz(h_.nb()));
            vgrad_ = alloc(9 * sz(h_.nc));
            lim_ = alloc(3 * sz(h_.nc));
            flux_ = alloc(sz(h_.nif));
            coef_ = alloc(sz(h_.nif));
            src_ = alloc(3 * sz(h_.nc));
            tmp_ = alloc(3 * sz(h_.nc));
        } catch (...) {
            release();
            throw;
        }
    }

    ~CudaFv() override { release(); }

    void gradient(const double* phi, const double* bv, int scheme, double* grad) override {
        to(phi_, phi, h_.nc);
        to(bv_, bv, h_.nb());
        launch_gradient(phi_, bv_, 1, 0, scheme, sgrad_, 3, 0);
        from(grad, sgrad_, 3 * h_.nc);
    }

    void vector_gradient(const double* u, const double* ub, int scheme, double* grad) override {
        to(vphi_, u, 3 * h_.nc);
        to(vbv_, ub, 3 * h_.nb());
        for (int i = 0; i < 3; ++i) {
            launch_gradient(vphi_, vbv_, 3, i, scheme, vgrad_, 9, 3 * i);
        }
        from(grad, vgrad_, 9 * h_.nc);
        have_vector_ = true;
    }

    void scalar_corrections(const double* phi, const double* bv, int scheme, const double* flux,
                            const double* gamma, bool linear_upwind, double* source) override {
        to(phi_, phi, h_.nc);
        to(bv_, bv, h_.nb());
        launch_gradient(phi_, bv_, 1, 0, scheme, sgrad_, 3, 0);
        to(flux_, flux, h_.nif);
        to(coef_, gamma, h_.nif);
        to(src_, source, h_.nc);
        scalar_kernel<<<grid(), k_fv_block>>>(m_, sgrad_, flux_, coef_, linear_upwind, src_);
        fv_check(cudaGetLastError(), "kernel");
        from(source, src_, h_.nc);
    }

    void momentum_corrections(const double* flux, const double* nu_face, bool linear_upwind,
                              bool limited, bool venkat, const double* eps2, double central,
                              bool transpose_term, double* source) override {
        if (!have_vector_) {
            throw std::logic_error("DeviceFv::momentum_corrections before vector_gradient");
        }
        to(flux_, flux, h_.nif);
        to(coef_, nu_face, h_.nif);
        to(src_, source, 3 * h_.nc);
        const bool use_lim = limited && linear_upwind;
        if (use_lim) {
            limiter_kernel<<<grid(), k_fv_block>>>(m_, vphi_, vbv_, vgrad_, venkat,
                                                   venkat ? eps2[0] : 0.0, venkat ? eps2[1] : 0.0,
                                                   venkat ? eps2[2] : 0.0, lim_);
            fv_check(cudaGetLastError(), "kernel");
        }
        momentum_kernel<<<grid(), k_fv_block>>>(m_, vgrad_, lim_, use_lim, flux_, coef_,
                                                linear_upwind, central, vphi_, transpose_term,
                                                src_);
        fv_check(cudaGetLastError(), "kernel");
        from(source, src_, 3 * h_.nc);
    }

    [[nodiscard]] bool on_device() const override { return true; }

private:
    [[nodiscard]] int grid() const { return (h_.nc + k_fv_block - 1) / k_fv_block; }

    void launch_gradient(const double* phi, const double* bv, int stride, int comp, int scheme,
                         double* out, int ostride, int ooff) {
        if (scheme == 2) { // corrected Gauss: least-squares gradient first
            gradient_kernel<<<grid(), k_fv_block>>>(m_, phi, bv, stride, comp, true, tmp_, 3, 0);
            fv_check(cudaGetLastError(), "kernel");
            corrected_kernel<<<grid(), k_fv_block>>>(m_, phi, bv, stride, comp, tmp_, out, ostride,
                                                     ooff);
            fv_check(cudaGetLastError(), "kernel");
            return;
        }
        gradient_kernel<<<grid(), k_fv_block>>>(m_, phi, bv, stride, comp, scheme == 1, out,
                                                ostride, ooff);
        fv_check(cudaGetLastError(), "kernel");
    }

    template <typename T> const T* put(const std::vector<T>& v) {
        void* p = nullptr;
        const std::size_t bytes = sizeof(T) * (v.empty() ? 1 : v.size());
        fv_check(cudaMalloc(&p, bytes), "malloc");
        owned_.push_back(p);
        if (!v.empty()) {
            fv_check(cudaMemcpy(p, v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice), "copy");
        }
        return static_cast<const T*>(p);
    }

    double* alloc(std::size_t n) {
        void* p = nullptr;
        fv_check(cudaMalloc(&p, sizeof(double) * n), "malloc");
        owned_.push_back(p);
        return static_cast<double*>(p);
    }

    static void to(double* dst, const double* src, int n) {
        if (n > 0) {
            fv_check(cudaMemcpy(dst, src, sizeof(double) * static_cast<std::size_t>(n),
                                cudaMemcpyHostToDevice),
                     "copy");
        }
    }

    static void from(double* dst, const double* src, int n) {
        if (n > 0) {
            fv_check(cudaMemcpy(dst, src, sizeof(double) * static_cast<std::size_t>(n),
                                cudaMemcpyDeviceToHost),
                     "copy");
        }
    }

    void release() {
        for (void* p : owned_) {
            cudaFree(p);
        }
        owned_.clear();
    }

    FvHostMesh h_;
    fvcuda::FvArrays m_{};
    std::vector<void*> owned_;
    double* phi_ = nullptr;
    double* bv_ = nullptr;
    double* sgrad_ = nullptr;
    double* vphi_ = nullptr;
    double* vbv_ = nullptr;
    double* vgrad_ = nullptr;
    double* lim_ = nullptr;
    double* flux_ = nullptr;
    double* coef_ = nullptr;
    double* src_ = nullptr;
    double* tmp_ = nullptr;
    bool have_vector_ = false;
};

} // namespace

std::unique_ptr<DeviceFv::Impl> make_fv_cuda(const FvMeshView& mesh) {
    return std::make_unique<CudaFv>(mesh);
}

} // namespace cfd::gpu::detail
