#include <cfd/gpu/krylov.hpp>
#include <stdexcept>
#include <vector>

namespace cfd::gpu {

namespace {

// Owned copy of a CSR matrix (views into it stay valid while it lives).
struct OwnedCsr {
    int n = 0;
    std::vector<int> row_ptr;
    std::vector<int> col;
    std::vector<double> val;

    OwnedCsr() = default;
    explicit OwnedCsr(const CsrView& v) : n(v.n) {
        if (v.row_ptr == nullptr || v.n == 0) {
            n = 0;
            return;
        }
        row_ptr.assign(v.row_ptr, v.row_ptr + v.n + 1);
        col.assign(v.col, v.col + v.nnz);
        val.assign(v.val, v.val + v.nnz);
    }
    [[nodiscard]] CsrView view() const {
        if (n == 0) {
            return {};
        }
        return {n, static_cast<int>(val.size()), row_ptr.data(), col.data(), val.data()};
    }
};

} // namespace

struct DeviceSolver::Impl {
    int n = 0;
    int n_faces = 0;
    std::vector<int> row_ptr;
    std::vector<int> col;
    std::vector<int> diag_pos;
    std::vector<int> upper_pos;
    std::vector<int> lower_pos;
    std::vector<double> val; // host assembly
    // AMG (host path): transfer operators of level 0 and the coarser levels.
    struct Level {
        OwnedCsr a;
        OwnedCsr p;
        OwnedCsr r;
    };
    std::vector<Level> levels;
    std::vector<double> coarse_inverse;
    int sweeps = 2;
    double weight = 0.7;
    void* cuda = nullptr;
};

DeviceSolver::DeviceSolver(const LduPattern& pt) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.n = pt.n;
    m.n_faces = pt.n_faces;
    m.row_ptr.assign(pt.row_ptr, pt.row_ptr + pt.n + 1);
    m.col.assign(pt.col, pt.col + pt.nnz);
    m.diag_pos.assign(pt.diag_pos, pt.diag_pos + pt.n);
    m.upper_pos.assign(pt.upper_pos, pt.upper_pos + pt.n_faces);
    m.lower_pos.assign(pt.lower_pos, pt.lower_pos + pt.n_faces);
    m.val.assign(static_cast<std::size_t>(pt.nnz), 0.0);
#ifdef CFD_GPU_CUDA
    if (detail::cuda_device_available()) {
        m.cuda = detail::cuda_system_create(pt);
    }
#endif
}

DeviceSolver::~DeviceSolver() {
#ifdef CFD_GPU_CUDA
    if (impl_->cuda != nullptr) {
        detail::cuda_system_destroy(impl_->cuda);
    }
#endif
}

bool DeviceSolver::on_device() const {
    return impl_->cuda != nullptr;
}

void DeviceSolver::set_amg(const AmgView& amg) {
    if (amg.levels.empty()) {
        throw std::invalid_argument("DeviceSolver::set_amg: empty hierarchy");
    }
    Impl& m = *impl_;
#ifdef CFD_GPU_CUDA
    if (m.cuda != nullptr) {
        AmgView v = amg;
        v.levels[0].a = {}; // assembled on the device
        detail::cuda_system_set_amg(m.cuda, v);
        return;
    }
#endif
    m.levels.clear();
    for (std::size_t l = 0; l < amg.levels.size(); ++l) {
        Impl::Level lv;
        if (l > 0) {
            lv.a = OwnedCsr(amg.levels[l].a);
        }
        lv.p = OwnedCsr(amg.levels[l].p);
        lv.r = OwnedCsr(amg.levels[l].r);
        m.levels.push_back(std::move(lv));
    }
    const int nc = amg.levels.back().a.n;
    m.coarse_inverse.assign(amg.coarse_inverse,
                            amg.coarse_inverse + static_cast<std::size_t>(nc) * nc);
    m.sweeps = amg.smooth_sweeps;
    m.weight = amg.jacobi_weight;
}

SolveStats DeviceSolver::solve(const double* diag, const double* upper, const double* lower,
                               const double* b, double* x, Method method, bool use_amg,
                               const SolveControl& control) {
    Impl& m = *impl_;
#ifdef CFD_GPU_CUDA
    if (m.cuda != nullptr) {
        return detail::cuda_system_solve(m.cuda, diag, upper, lower, b, x, method, use_amg,
                                         control);
    }
#endif
    // Host assembly of the CSR values.
    for (int i = 0; i < m.n; ++i) {
        m.val[static_cast<std::size_t>(m.diag_pos[static_cast<std::size_t>(i)])] = diag[i];
    }
    for (int f = 0; f < m.n_faces; ++f) {
        const auto fs = static_cast<std::size_t>(f);
        m.val[static_cast<std::size_t>(m.upper_pos[fs])] = upper[f];
        m.val[static_cast<std::size_t>(m.lower_pos[fs])] = lower[f];
    }
    const CsrView a{m.n, static_cast<int>(m.val.size()), m.row_ptr.data(), m.col.data(),
                    m.val.data()};
    if (use_amg && !m.levels.empty()) {
        AmgView v;
        v.smooth_sweeps = m.sweeps;
        v.jacobi_weight = m.weight;
        v.coarse_inverse = m.coarse_inverse.data();
        for (std::size_t l = 0; l < m.levels.size(); ++l) {
            v.levels.push_back(
                {l == 0 ? a : m.levels[l].a.view(), m.levels[l].p.view(), m.levels[l].r.view()});
        }
        return detail::solve_amg_openmp(v, b, x, control, true);
    }
    return detail::solve_openmp(a, b, x, method, control);
}

} // namespace cfd::gpu
