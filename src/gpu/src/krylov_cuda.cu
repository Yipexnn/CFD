// Jacobi-preconditioned CG / BiCGStab with CUDA kernels (CSR SpMV, fused vector updates,
// block-reduced dot products). All vectors stay in device memory during the solve.
#include <algorithm>
#include <cfd/gpu/krylov.hpp>
#include <cmath>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace cfd::gpu::detail {

namespace {

constexpr int k_block = 256;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
    }
}

__global__ void spmv_kernel(int n, const int* rp, const int* ci, const double* va,
                            const double* x, double* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double s = 0.0;
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            s += va[k] * x[ci[k]];
        }
        y[i] = s;
    }
}

__global__ void dot_kernel(int n, const double* x, const double* y, double* out) {
    __shared__ double cache[k_block];
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        s += x[i] * y[i];
    }
    cache[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            cache[threadIdx.x] += cache[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(out, cache[0]); // double atomicAdd: compute capability >= 6.0
    }
}

// y = a x + b y (+ c z): generic fused update
__global__ void axpbypcz_kernel(int n, double a, const double* x, double b, double* y, double c,
                                const double* z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + b * y[i] + (z != nullptr ? c * z[i] : 0.0);
    }
}

__global__ void mul_kernel(int n, const double* x, const double* d, double* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] * d[i];
    }
}

struct Device {
    int n = 0;
    int grid = 1;
    int dot_grid = 1;
    int* rp = nullptr;
    int* ci = nullptr;
    double* va = nullptr;
    double* dot_out = nullptr;
    std::vector<double*> buffers;

    ~Device() {
        cudaFree(rp);
        cudaFree(ci);
        cudaFree(va);
        cudaFree(dot_out);
        for (double* p : buffers) {
            cudaFree(p);
        }
    }

    double* vec() {
        double* p = nullptr;
        check(cudaMalloc(&p, sizeof(double) * static_cast<std::size_t>(n)), "malloc");
        buffers.push_back(p);
        check(cudaMemset(p, 0, sizeof(double) * static_cast<std::size_t>(n)), "memset");
        return p;
    }

    void spmv(const double* x, double* y) const {
        spmv_kernel<<<grid, k_block>>>(n, rp, ci, va, x, y);
    }

    double dot(const double* x, const double* y) const {
        check(cudaMemset(dot_out, 0, sizeof(double)), "memset");
        dot_kernel<<<dot_grid, k_block>>>(n, x, y, dot_out);
        double h = 0.0;
        check(cudaMemcpy(&h, dot_out, sizeof(double), cudaMemcpyDeviceToHost), "dot");
        return h;
    }

    // y = a x + b y + c z
    void update(double a, const double* x, double b, double* y, double c = 0.0,
                const double* z = nullptr) const {
        axpbypcz_kernel<<<grid, k_block>>>(n, a, x, b, y, c, z);
    }

    void mul(const double* x, const double* d, double* y) const {
        mul_kernel<<<grid, k_block>>>(n, x, d, y);
    }
};

} // namespace

bool cuda_device_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string cuda_device_name() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return "unknown";
    }
    return prop.name;
}

SolveStats solve_cuda(const CsrView& m, const double* b_host, double* x_host, Method method,
                      const SolveControl& control) {
    const int n = m.n;
    SolveStats st;
    st.on_device = true;
    if (n == 0) {
        st.converged = true;
        return st;
    }
    std::vector<double> dinv_h(static_cast<std::size_t>(n), 1.0);
    for (int i = 0; i < n; ++i) {
        for (int k = m.row_ptr[i]; k < m.row_ptr[i + 1]; ++k) {
            if (m.col[k] == i && m.val[k] != 0.0) {
                dinv_h[static_cast<std::size_t>(i)] = 1.0 / m.val[k];
            }
        }
    }
    Device d;
    d.n = n;
    d.grid = (n + k_block - 1) / k_block;
    d.dot_grid = std::min(d.grid, 1024);
    const auto bytes_i = [](int k) { return sizeof(int) * static_cast<std::size_t>(k); };
    const auto bytes_d = [](int k) { return sizeof(double) * static_cast<std::size_t>(k); };
    check(cudaMalloc(&d.rp, bytes_i(n + 1)), "malloc");
    check(cudaMalloc(&d.ci, bytes_i(m.nnz)), "malloc");
    check(cudaMalloc(&d.va, bytes_d(m.nnz)), "malloc");
    check(cudaMalloc(&d.dot_out, sizeof(double)), "malloc");
    check(cudaMemcpy(d.rp, m.row_ptr, bytes_i(n + 1), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d.ci, m.col, bytes_i(m.nnz), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(d.va, m.val, bytes_d(m.nnz), cudaMemcpyHostToDevice), "copy");
    double* b = d.vec();
    double* x = d.vec();
    double* dinv = d.vec();
    check(cudaMemcpy(b, b_host, bytes_d(n), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(x, x_host, bytes_d(n), cudaMemcpyHostToDevice), "copy");
    check(cudaMemcpy(dinv, dinv_h.data(), bytes_d(n), cudaMemcpyHostToDevice), "copy");

    double* r = d.vec();
    d.spmv(x, r);
    d.update(1.0, b, -1.0, r); // r = b - A x
    double res = std::sqrt(d.dot(r, r));
    if (res <= control.abs_tolerance) {
        st.residual = res;
        st.converged = true;
        return st;
    }
    if (method == Method::Cg) {
        double* z = d.vec();
        double* p = d.vec();
        double* q = d.vec();
        d.mul(r, dinv, z);
        d.update(1.0, z, 0.0, p);
        double rz = d.dot(r, z);
        for (st.iterations = 1; st.iterations <= control.max_iterations; ++st.iterations) {
            d.spmv(p, q);
            const double pq = d.dot(p, q);
            if (pq == 0.0) {
                break;
            }
            const double alpha = rz / pq;
            d.update(alpha, p, 1.0, x);
            d.update(-alpha, q, 1.0, r);
            res = std::sqrt(d.dot(r, r));
            if (res <= control.abs_tolerance) {
                st.converged = true;
                break;
            }
            d.mul(r, dinv, z);
            const double rz_new = d.dot(r, z);
            d.update(1.0, z, rz_new / rz, p);
            rz = rz_new;
        }
    } else {
        double* r0 = d.vec();
        double* p = d.vec();
        double* v = d.vec();
        double* s = d.vec();
        double* t = d.vec();
        double* ph = d.vec();
        double* sh = d.vec();
        d.update(1.0, r, 0.0, r0);
        check(cudaMemset(p, 0, bytes_d(n)), "memset");
        check(cudaMemset(v, 0, bytes_d(n)), "memset");
        double rho = 1.0;
        double alpha = 1.0;
        double omega = 1.0;
        for (st.iterations = 1; st.iterations <= control.max_iterations; ++st.iterations) {
            const double rho_new = d.dot(r0, r);
            if (rho_new == 0.0 || omega == 0.0) {
                break;
            }
            const double beta = (rho_new / rho) * (alpha / omega);
            rho = rho_new;
            d.update(1.0, r, beta, p, -beta * omega, v); // p = r + beta (p - omega v)
            d.mul(p, dinv, ph);
            d.spmv(ph, v);
            const double r0v = d.dot(r0, v);
            if (r0v == 0.0) {
                break;
            }
            alpha = rho / r0v;
            d.update(1.0, r, 0.0, s, -alpha, v); // s = r - alpha v
            const double s_norm = std::sqrt(d.dot(s, s));
            if (s_norm <= control.abs_tolerance) {
                d.update(alpha, ph, 1.0, x);
                res = s_norm;
                st.converged = true;
                break;
            }
            d.mul(s, dinv, sh);
            d.spmv(sh, t);
            const double tt = d.dot(t, t);
            omega = tt > 0.0 ? d.dot(t, s) / tt : 0.0;
            d.update(alpha, ph, 1.0, x, omega, sh);
            d.update(1.0, s, 0.0, r, -omega, t); // r = s - omega t
            res = std::sqrt(d.dot(r, r));
            if (res <= control.abs_tolerance) {
                st.converged = true;
                break;
            }
        }
    }
    st.iterations = std::min(st.iterations, control.max_iterations);
    st.residual = res;
    check(cudaGetLastError(), "kernel");
    check(cudaMemcpy(x_host, x, bytes_d(n), cudaMemcpyDeviceToHost), "copy");
    return st;
}

namespace {

__global__ void jacobi_kernel(int n, double w, const double* d, const double* b, const double* t,
                              double* x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += w * d[i] * (b[i] - t[i]);
    }
}

__global__ void dense_mv_kernel(int n, const double* m, const double* b, double* x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) {
            s += m[static_cast<long long>(i) * n + j] * b[j];
        }
        x[i] = s;
    }
}

struct DeviceCsr {
    int n = 0;
    int* rp = nullptr;
    int* ci = nullptr;
    double* va = nullptr;
};

struct DeviceLevel {
    DeviceCsr a, p, r;
    double* dinv = nullptr;
    double* x = nullptr;
    double* b = nullptr;
    double* t = nullptr;
};

class AmgDevice {
public:
    explicit AmgDevice(const AmgView& v) : view_(v) {
        for (const AmgLevelView& l : v.levels) {
            DeviceLevel d;
            d.a = upload(l.a);
            d.p = upload(l.p);
            d.r = upload(l.r);
            std::vector<double> dinv(static_cast<std::size_t>(l.a.n), 1.0);
            for (int i = 0; i < l.a.n; ++i) {
                for (int k = l.a.row_ptr[i]; k < l.a.row_ptr[i + 1]; ++k) {
                    if (l.a.col[k] == i && l.a.val[k] != 0.0) {
                        dinv[static_cast<std::size_t>(i)] = 1.0 / l.a.val[k];
                    }
                }
            }
            d.dinv = alloc<double>(l.a.n);
            check(cudaMemcpy(d.dinv, dinv.data(), sizeof(double) * dinv.size(),
                             cudaMemcpyHostToDevice),
                  "copy");
            d.x = alloc<double>(l.a.n);
            d.b = alloc<double>(l.a.n);
            d.t = alloc<double>(l.a.n);
            levels_.push_back(d);
        }
        const int nc = v.levels.back().a.n;
        coarse_ = alloc<double>(nc * nc);
        check(cudaMemcpy(coarse_, v.coarse_inverse,
                         sizeof(double) * static_cast<std::size_t>(nc) * nc,
                         cudaMemcpyHostToDevice),
              "copy");
    }
    AmgDevice(const AmgDevice&) = delete;
    AmgDevice& operator=(const AmgDevice&) = delete;
    ~AmgDevice() {
        for (void* p : owned_) {
            cudaFree(p);
        }
    }

    static int grid(int n) { return (n + k_block - 1) / k_block; }

    void spmv(const DeviceCsr& m, const double* x, double* y) const {
        if (m.n > 0) {
            spmv_kernel<<<grid(m.n), k_block>>>(m.n, m.rp, m.ci, m.va, x, y);
        }
    }

    void jacobi(const DeviceLevel& l) const {
        for (int s = 0; s < view_.smooth_sweeps; ++s) {
            spmv(l.a, l.x, l.t);
            jacobi_kernel<<<grid(l.a.n), k_block>>>(l.a.n, view_.jacobi_weight, l.dinv, l.b, l.t,
                                                    l.x);
        }
    }

    void vcycle(std::size_t lv) const {
        const DeviceLevel& l = levels_[lv];
        const int n = l.a.n;
        if (lv + 1 == levels_.size()) {
            dense_mv_kernel<<<grid(n), k_block>>>(n, coarse_, l.b, l.x);
            return;
        }
        check(cudaMemset(l.x, 0, sizeof(double) * static_cast<std::size_t>(n)), "memset");
        jacobi(l);
        spmv(l.a, l.x, l.t);
        axpbypcz_kernel<<<grid(n), k_block>>>(n, 1.0, l.b, -1.0, l.t, 0.0, nullptr); // t = b - t
        spmv(l.r, l.t, levels_[lv + 1].b);
        vcycle(lv + 1);
        spmv(l.p, levels_[lv + 1].x, l.t);
        axpbypcz_kernel<<<grid(n), k_block>>>(n, 1.0, l.t, 1.0, l.x, 0.0, nullptr); // x += t
        jacobi(l);
    }

    [[nodiscard]] const DeviceLevel& finest() const { return levels_.front(); }

private:
    template <typename T> T* alloc(long long count) {
        T* p = nullptr;
        check(cudaMalloc(&p, sizeof(T) * static_cast<std::size_t>(std::max(count, 1LL))),
              "malloc");
        check(cudaMemset(p, 0, sizeof(T) * static_cast<std::size_t>(std::max(count, 1LL))),
              "memset");
        owned_.push_back(p);
        return p;
    }
    DeviceCsr upload(const CsrView& m) {
        DeviceCsr d;
        if (m.n == 0 || m.row_ptr == nullptr) {
            return d;
        }
        d.n = m.n;
        d.rp = alloc<int>(m.n + 1);
        d.ci = alloc<int>(m.nnz);
        d.va = alloc<double>(m.nnz);
        check(cudaMemcpy(d.rp, m.row_ptr, sizeof(int) * static_cast<std::size_t>(m.n + 1),
                         cudaMemcpyHostToDevice),
              "copy");
        check(cudaMemcpy(d.ci, m.col, sizeof(int) * static_cast<std::size_t>(m.nnz),
                         cudaMemcpyHostToDevice),
              "copy");
        check(cudaMemcpy(d.va, m.val, sizeof(double) * static_cast<std::size_t>(m.nnz),
                         cudaMemcpyHostToDevice),
              "copy");
        return d;
    }

    const AmgView& view_;
    std::vector<DeviceLevel> levels_;
    double* coarse_ = nullptr;
    std::vector<void*> owned_;
};

} // namespace

SolveStats solve_amg_cuda(const AmgView& amg, const double* b_host, double* x_host,
                          const SolveControl& control) {
    const AmgDevice h(amg);
    const DeviceLevel& f = h.finest();
    const int n = f.a.n;
    Device d; // vectors and dot products of the outer CG
    d.n = n;
    d.grid = (n + k_block - 1) / k_block;
    d.dot_grid = std::min(d.grid, 1024);
    check(cudaMalloc(&d.dot_out, sizeof(double)), "malloc");
    double* b = d.vec();
    double* x = d.vec();
    double* r = d.vec();
    double* p = d.vec();
    double* q = d.vec();
    check(cudaMemcpy(b, b_host, sizeof(double) * static_cast<std::size_t>(n),
                     cudaMemcpyHostToDevice),
          "copy");
    check(cudaMemcpy(x, x_host, sizeof(double) * static_cast<std::size_t>(n),
                     cudaMemcpyHostToDevice),
          "copy");
    SolveStats st;
    st.on_device = true;
    h.spmv(f.a, x, r);
    d.update(1.0, b, -1.0, r); // r = b - A x
    double res = std::sqrt(d.dot(r, r));
    auto precondition = [&]() { // z (= f.x) = V-cycle(r)
        check(cudaMemcpy(f.b, r, sizeof(double) * static_cast<std::size_t>(n),
                         cudaMemcpyDeviceToDevice),
              "copy");
        h.vcycle(0);
    };
    if (res > control.abs_tolerance) {
        precondition();
        d.update(1.0, f.x, 0.0, p);
        double rz = d.dot(r, f.x);
        for (st.iterations = 1; st.iterations <= control.max_iterations; ++st.iterations) {
            h.spmv(f.a, p, q);
            const double pq = d.dot(p, q);
            if (pq == 0.0) {
                break;
            }
            const double alpha = rz / pq;
            d.update(alpha, p, 1.0, x);
            d.update(-alpha, q, 1.0, r);
            res = std::sqrt(d.dot(r, r));
            if (res <= control.abs_tolerance) {
                st.converged = true;
                break;
            }
            precondition();
            const double rz_new = d.dot(r, f.x);
            d.update(1.0, f.x, rz_new / rz, p);
            rz = rz_new;
        }
        st.iterations = std::min(st.iterations, control.max_iterations);
    } else {
        st.converged = true;
    }
    st.residual = res;
    check(cudaGetLastError(), "kernel");
    check(cudaMemcpy(x_host, x, sizeof(double) * static_cast<std::size_t>(n),
                     cudaMemcpyDeviceToHost),
          "copy");
    return st;
}

} // namespace cfd::gpu::detail
