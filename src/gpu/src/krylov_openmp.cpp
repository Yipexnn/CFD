// Jacobi-preconditioned CG / BiCGStab with OpenMP: target offload when compiled with
// CFD_GPU_OMP_TARGET (the OpenMP runtime falls back to the host without a device), host
// threads otherwise. All vectors stay on the device for the whole solve.
#include <algorithm>
#include <cfd/gpu/krylov.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CFD_STR(x) #x

namespace cfd::gpu::detail {

// Host threads.
namespace host {
// Threads only for large vectors (thread start-up dominates small systems).
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__)
#define CFD_LOOP __pragma(omp parallel for if (n > 20000))
#define CFD_LOOP_SUM(var) __pragma(omp parallel for reduction(+ : var) if (n > 20000))
#elif defined(_OPENMP)
#define CFD_LOOP _Pragma("omp parallel for if (n > 20000)")
#define CFD_LOOP_SUM(var) _Pragma(CFD_STR(omp parallel for reduction(+ : var) if (n > 20000)))
#else
#define CFD_LOOP
#define CFD_LOOP_SUM(var)
#endif
#include "krylov_kernels.inc"
#undef CFD_LOOP
#undef CFD_LOOP_SUM
} // namespace host

struct HostSpace {
    using Arrays = host::Arrays;
    static SolveStats cg(const Arrays& a, const double* b, double* x, const double* d, double* r,
                         double* z, double* p, double* q, const SolveControl& c) {
        return host::cg(a, b, x, d, r, z, p, q, c);
    }
    static SolveStats bicgstab(const Arrays& a, const double* b, double* x, const double* d,
                               double* r, double* r0, double* p, double* v, double* s, double* t,
                               double* ph, double* sh, const SolveControl& c) {
        return host::bicgstab(a, b, x, d, r, r0, p, v, s, t, ph, sh, c);
    }
    using AmgLevel = host::AmgLevel;
    static SolveStats amg_cg(const AmgLevel* l, int nl, const double* ci, int sw, double w,
                             const double* b, double* x, double* r, double* p, double* q,
                             const SolveControl& c) {
        return host::amg_cg(l, nl, ci, sw, w, b, x, r, p, q, c);
    }
};

#if defined(CFD_GPU_OMP_TARGET) && defined(_OPENMP)
// OpenMP target offload (arrays mapped once per solve by the caller).
namespace offload {
#define CFD_LOOP _Pragma("omp target teams distribute parallel for")
#define CFD_LOOP_SUM(var)                                                                          \
    _Pragma(CFD_STR(omp target teams distribute parallel for reduction(+ : var)))
#include "krylov_kernels.inc"
#undef CFD_LOOP
#undef CFD_LOOP_SUM
} // namespace offload

struct OffloadSpace {
    using Arrays = offload::Arrays;
    static SolveStats cg(const Arrays& a, const double* b, double* x, const double* d, double* r,
                         double* z, double* p, double* q, const SolveControl& c) {
        return offload::cg(a, b, x, d, r, z, p, q, c);
    }
    static SolveStats bicgstab(const Arrays& a, const double* b, double* x, const double* d,
                               double* r, double* r0, double* p, double* v, double* s, double* t,
                               double* ph, double* sh, const SolveControl& c) {
        return offload::bicgstab(a, b, x, d, r, r0, p, v, s, t, ph, sh, c);
    }
    using AmgLevel = offload::AmgLevel;
    static SolveStats amg_cg(const AmgLevel* l, int nl, const double* ci, int sw, double w,
                             const double* b, double* x, double* r, double* p, double* q,
                             const SolveControl& c) {
        return offload::amg_cg(l, nl, ci, sw, w, b, x, r, p, q, c);
    }
};
#endif

bool openmp_device_available() {
#if defined(CFD_GPU_OMP_TARGET) && defined(_OPENMP)
    return omp_get_num_devices() > 0;
#else
    return false;
#endif
}

SolveStats solve_openmp(const CsrView& m, const double* b, double* x, Method method,
                        const SolveControl& control, bool allow_device) {
    const int n = m.n;
    const int nnz = m.nnz;
    std::vector<double> dinv_v(static_cast<std::size_t>(n), 1.0);
    for (int i = 0; i < n; ++i) {
        for (int k = m.row_ptr[i]; k < m.row_ptr[i + 1]; ++k) {
            if (m.col[k] == i && m.val[k] != 0.0) {
                dinv_v[static_cast<std::size_t>(i)] = 1.0 / m.val[k];
            }
        }
    }
    const int nvec = method == Method::Cg ? 4 : 8;
    std::vector<double> work_v(static_cast<std::size_t>(nvec) * static_cast<std::size_t>(n), 0.0);
    const int* rp = m.row_ptr;
    const int* ci = m.col;
    const double* va = m.val;
    double* dinv = dinv_v.data();
    double* w = work_v.data();
    const int nw = nvec * n;
    SolveStats st;
    auto run = [&](auto space) {
        using Space = decltype(space);
        const typename Space::Arrays a{n, rp, ci, va};
        if (method == Method::Cg) {
            return Space::cg(a, b, x, dinv, w, w + n, w + 2 * n, w + 3 * n, control);
        }
        return Space::bicgstab(a, b, x, dinv, w, w + n, w + 2 * n, w + 3 * n, w + 4 * n, w + 5 * n,
                               w + 6 * n, w + 7 * n, control);
    };
#if defined(CFD_GPU_OMP_TARGET) && defined(_OPENMP)
    if (allow_device && openmp_device_available()) {
#pragma omp target data map(to : rp[0 : n + 1], ci[0 : nnz], va[0 : nnz], b[0 : n], dinv[0 : n])   \
    map(tofrom : x[0 : n]) map(alloc : w[0 : nw])
        { st = run(OffloadSpace{}); }
        st.on_device = true;
        return st;
    }
#endif
    (void)nnz;
    (void)allow_device;
    (void)nw;
    st = run(HostSpace{});
    st.on_device = false;
    return st;
}

SolveStats solve_openmp(const CsrView& m, const double* b, double* x, Method method,
                        const SolveControl& control) {
    return solve_openmp(m, b, x, method, control, true);
}

namespace {

std::vector<double> inverse_diagonal(const CsrView& m) {
    std::vector<double> d(static_cast<std::size_t>(m.n), 1.0);
    for (int i = 0; i < m.n; ++i) {
        for (int k = m.row_ptr[i]; k < m.row_ptr[i + 1]; ++k) {
            if (m.col[k] == i && m.val[k] != 0.0) {
                d[static_cast<std::size_t>(i)] = 1.0 / m.val[k];
            }
        }
    }
    return d;
}

template <typename Space>
SolveStats run_amg(const AmgView& amg, const double* b, double* x, const SolveControl& control,
                   std::vector<std::vector<double>>& dinv, std::vector<std::vector<double>>& work,
                   std::vector<double>& cg_work) {
    using Level = typename Space::AmgLevel;
    using A = typename Space::Arrays;
    const int nl = static_cast<int>(amg.levels.size());
    std::vector<Level> levels(static_cast<std::size_t>(nl));
    for (int l = 0; l < nl; ++l) {
        const auto ls = static_cast<std::size_t>(l);
        const AmgLevelView& v = amg.levels[ls];
        const auto n = static_cast<std::size_t>(v.a.n);
        levels[ls].a = A{v.a.n, v.a.row_ptr, v.a.col, v.a.val};
        levels[ls].p = A{v.p.n, v.p.row_ptr, v.p.col, v.p.val};
        levels[ls].r = A{v.r.n, v.r.row_ptr, v.r.col, v.r.val};
        levels[ls].dinv = dinv[ls].data();
        levels[ls].x = work[ls].data();
        levels[ls].b = work[ls].data() + n;
        levels[ls].t = work[ls].data() + 2 * n;
    }
    const auto n0 = static_cast<std::size_t>(amg.levels[0].a.n);
    double* w = cg_work.data();
    return Space::amg_cg(levels.data(), nl, amg.coarse_inverse, amg.smooth_sweeps,
                         amg.jacobi_weight, b, x, w, w + n0, w + 2 * n0, control);
}

} // namespace

SolveStats solve_amg_openmp(const AmgView& amg, const double* b, double* x,
                            const SolveControl& control, bool allow_device) {
    if (amg.levels.empty()) {
        throw std::invalid_argument("solve_amg_cg: empty hierarchy");
    }
    const std::size_t nl = amg.levels.size();
    std::vector<std::vector<double>> dinv(nl);
    std::vector<std::vector<double>> work(nl);
    for (std::size_t l = 0; l < nl; ++l) {
        dinv[l] = inverse_diagonal(amg.levels[l].a);
        work[l].assign(3 * static_cast<std::size_t>(amg.levels[l].a.n), 0.0);
    }
    const int n = amg.levels[0].a.n;
    std::vector<double> cg_work(3 * static_cast<std::size_t>(n), 0.0);
    SolveStats st;
#if defined(CFD_GPU_OMP_TARGET) && defined(_OPENMP)
    if (allow_device && openmp_device_available()) {
        // Map every level (operators, transfer operators, work vectors) once per solve.
        auto map_csr = [](const CsrView& m, bool to_device) {
            if (m.n == 0 || m.row_ptr == nullptr) {
                return;
            }
            [[maybe_unused]] const int* rp = m.row_ptr;
            [[maybe_unused]] const int* ci = m.col;
            [[maybe_unused]] const double* va = m.val;
            [[maybe_unused]] const int rows = m.n + 1;
            [[maybe_unused]] const int nnz = m.nnz;
            if (to_device) {
#pragma omp target enter data map(to : rp[0 : rows], ci[0 : nnz], va[0 : nnz])
            } else {
#pragma omp target exit data map(release : rp[0 : rows], ci[0 : nnz], va[0 : nnz])
            }
        };
        for (std::size_t l = 0; l < nl; ++l) {
            map_csr(amg.levels[l].a, true);
            map_csr(amg.levels[l].p, true);
            map_csr(amg.levels[l].r, true);
            [[maybe_unused]] double* d = dinv[l].data();
            [[maybe_unused]] double* wk = work[l].data();
            [[maybe_unused]] const int nd = static_cast<int>(dinv[l].size());
            [[maybe_unused]] const int nw = static_cast<int>(work[l].size());
#pragma omp target enter data map(to : d[0 : nd], wk[0 : nw])
        }
        [[maybe_unused]] const double* ci = amg.coarse_inverse;
        [[maybe_unused]] const int nc = amg.levels.back().a.n;
        [[maybe_unused]] const long long ncc = static_cast<long long>(nc) * nc;
        [[maybe_unused]] double* cw = cg_work.data();
        [[maybe_unused]] const int ncw = static_cast<int>(cg_work.size());
#pragma omp target data map(to : b[0 : n], ci[0 : ncc]) map(tofrom : x[0 : n]) map(to : cw[0 : ncw])
        { st = run_amg<OffloadSpace>(amg, b, x, control, dinv, work, cg_work); }
        for (std::size_t l = 0; l < nl; ++l) {
            map_csr(amg.levels[l].a, false);
            map_csr(amg.levels[l].p, false);
            map_csr(amg.levels[l].r, false);
            [[maybe_unused]] double* d = dinv[l].data();
            [[maybe_unused]] double* wk = work[l].data();
            [[maybe_unused]] const int nd = static_cast<int>(dinv[l].size());
            [[maybe_unused]] const int nw = static_cast<int>(work[l].size());
#pragma omp target exit data map(release : d[0 : nd], wk[0 : nw])
        }
        st.on_device = true;
        return st;
    }
#endif
    (void)allow_device;
    st = run_amg<HostSpace>(amg, b, x, control, dinv, work, cg_work);
    st.on_device = false;
    return st;
}

void assemble_openmp(int n_cells, int n_faces, const int* owner, const int* neighbour,
                     const double* flux, const double* gamma, const double* delta_coeff,
                     double* upper, double* lower, double* diag, double* divergence) {
#if defined(CFD_GPU_OMP_TARGET) && defined(_OPENMP)
    if (openmp_device_available()) {
#pragma omp target teams distribute parallel for map(                                              \
        to : owner[0 : n_faces], neighbour[0 : n_faces], flux[0 : n_faces], gamma[0 : n_faces],    \
            delta_coeff[0 : n_faces]) map(from : upper[0 : n_faces], lower[0 : n_faces])           \
    map(tofrom : diag[0 : n_cells], divergence[0 : n_cells])
        for (int f = 0; f < n_faces; ++f) {
            const double fl = flux[f];
            const double g = gamma[f] * delta_coeff[f];
            const double pos = fl > 0.0 ? fl : 0.0;
            const double neg = fl < 0.0 ? fl : 0.0;
            upper[f] = neg - g;
            lower[f] = -pos - g;
            const int p = owner[f];
            const int n = neighbour[f];
#pragma omp atomic
            diag[p] += pos + g;
#pragma omp atomic
            diag[n] += -neg + g;
#pragma omp atomic
            divergence[p] += fl;
#pragma omp atomic
            divergence[n] -= fl;
        }
        return;
    }
#endif
    (void)n_cells;
    for (int f = 0; f < n_faces; ++f) {
        const double fl = flux[f];
        const double g = gamma[f] * delta_coeff[f];
        const double pos = fl > 0.0 ? fl : 0.0;
        const double neg = fl < 0.0 ? fl : 0.0;
        upper[f] = neg - g;
        lower[f] = -pos - g;
        diag[owner[f]] += pos + g;
        diag[neighbour[f]] += -neg + g;
        divergence[owner[f]] += fl;
        divergence[neighbour[f]] -= fl;
    }
}

} // namespace cfd::gpu::detail

namespace cfd::gpu {

SolveStats solve_host(const CsrView& a, const double* b, double* x, Method method,
                      const SolveControl& control) {
    return detail::solve_openmp(a, b, x, method, control, false);
}

} // namespace cfd::gpu