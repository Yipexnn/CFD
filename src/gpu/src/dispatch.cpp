#include <cfd/gpu/krylov.hpp>

namespace cfd::gpu {

SolveStats solve(const CsrView& a, const double* b, double* x, Method method,
                 const SolveControl& control) {
#ifdef CFD_GPU_CUDA
    if (detail::cuda_device_available()) {
        return detail::solve_cuda(a, b, x, method, control);
    }
#endif
    return detail::solve_openmp(a, b, x, method, control);
}

SolveStats solve_amg_cg(const AmgView& amg, const double* b, double* x,
                        const SolveControl& control) {
#ifdef CFD_GPU_CUDA
    if (detail::cuda_device_available()) {
        return detail::solve_amg_cuda(amg, b, x, control);
    }
#endif
    return detail::solve_amg_openmp(amg, b, x, control, true);
}

SolveStats solve_amg_cg_host(const AmgView& amg, const double* b, double* x,
                             const SolveControl& control) {
    return detail::solve_amg_openmp(amg, b, x, control, false);
}

std::string backend_name() {
#if defined(CFD_GPU_CUDA)
    return "cuda";
#elif defined(CFD_GPU_OMP_TARGET)
    return "openmp-offload";
#else
    return "host";
#endif
}

bool device_available() {
#if defined(CFD_GPU_CUDA)
    return detail::cuda_device_available();
#elif defined(CFD_GPU_OMP_TARGET)
    return detail::openmp_device_available();
#else
    return false;
#endif
}

std::string device_description() {
    std::string s = backend_name();
#if defined(CFD_GPU_CUDA)
    s += device_available() ? " (device: " + detail::cuda_device_name() + ")"
                            : " (no CUDA device: host fallback)";
#elif defined(CFD_GPU_OMP_TARGET)
    s += device_available() ? " (offload device present)" : " (no offload device: host fallback)";
#endif
    return s;
}

} // namespace cfd::gpu
