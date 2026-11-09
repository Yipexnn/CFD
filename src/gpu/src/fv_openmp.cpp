// gpu::DeviceFv with OpenMP: target offload with persistent device buffers when compiled with
// CFD_GPU_OMP_TARGET and a device is present, host threads otherwise.
#include "fv_impl.hpp"

#include <algorithm>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(CFD_GPU_OMP_TARGET) && defined(_OPENMP)
#define CFD_FV_OFFLOAD 1
#endif

namespace cfd::gpu::detail {

namespace fvhost {
#define CFD_HD
#include "fv_cell.inc"
#undef CFD_HD
} // namespace fvhost

#ifdef CFD_FV_OFFLOAD
namespace fvoff {
#pragma omp declare target
#define CFD_HD
#include "fv_cell.inc"
#undef CFD_HD
#pragma omp end declare target
} // namespace fvoff
#endif

bool openmp_device_available();

namespace {

// Threads only for large meshes (thread start-up dominates small ones).
#if defined(_OPENMP) && defined(_MSC_VER) && !defined(__clang__)
#define CFD_FV_HOST_LOOP __pragma(omp parallel for if (nc > 20000))
#elif defined(_OPENMP)
#define CFD_FV_HOST_LOOP _Pragma("omp parallel for if (nc > 20000)")
#else
#define CFD_FV_HOST_LOOP
#endif

// Local copies of the mesh pointers (translated to device pointers inside target regions).
#define CFD_FV_LOCALS                                                                              \
    [[maybe_unused]] const int nc = h_.nc;                                                         \
    [[maybe_unused]] const int nif = h_.nif;                                                       \
    [[maybe_unused]] const int nf = h_.nf;                                                         \
    [[maybe_unused]] const int* own = h_.own.data();                                               \
    [[maybe_unused]] const int* nei = h_.nei.data();                                               \
    [[maybe_unused]] const int* start = h_.start.data();                                           \
    [[maybe_unused]] const int* faces = h_.faces.data();                                           \
    [[maybe_unused]] const double* w = h_.w.data();                                                \
    [[maybe_unused]] const double* sf = h_.sf.data();                                              \
    [[maybe_unused]] const double* cf = h_.cf.data();                                              \
    [[maybe_unused]] const double* cc = h_.cc.data();                                              \
    [[maybe_unused]] const double* vol = h_.vol.data();                                            \
    [[maybe_unused]] const double* kv = h_.kv.data();                                              \
    [[maybe_unused]] const double* lsq_dd = h_.lsq_d.data();                                       \
    [[maybe_unused]] const double* lsq_inv = h_.lsq.data();
#define CFD_FV_ARRAYS                                                                              \
    { nc, nif, nf, own, nei, start, faces, w, sf, cf, cc, vol, kv, lsq_dd, lsq_inv }

std::size_t sz(int n) {
    return static_cast<std::size_t>(std::max(n, 1));
}

class OpenMpFv final : public DeviceFv::Impl {
public:
    explicit OpenMpFv(const FvMeshView& view)
        : h_(view), phi_(sz(h_.nc)), bv_(sz(h_.nb())), sgrad_(3 * sz(h_.nc)), vphi_(3 * sz(h_.nc)),
          vbv_(3 * sz(h_.nb())), vgrad_(9 * sz(h_.nc)), lim_(3 * sz(h_.nc)), flux_(sz(h_.nif)),
          coef_(sz(h_.nif)), src_(3 * sz(h_.nc)) {
#ifdef CFD_FV_OFFLOAD
        offload_ = openmp_device_available();
        if (offload_) {
            CFD_FV_LOCALS
            [[maybe_unused]] const int nnei = static_cast<int>(h_.nei.size());
            [[maybe_unused]] const int nfc = static_cast<int>(h_.faces.size());
            [[maybe_unused]] double* phi = phi_.data();
            [[maybe_unused]] double* bv = bv_.data();
            [[maybe_unused]] double* sg = sgrad_.data();
            [[maybe_unused]] double* vphi = vphi_.data();
            [[maybe_unused]] double* vbv = vbv_.data();
            [[maybe_unused]] double* vg = vgrad_.data();
            [[maybe_unused]] double* lim = lim_.data();
            [[maybe_unused]] double* flux = flux_.data();
            [[maybe_unused]] double* coef = coef_.data();
            [[maybe_unused]] double* src = src_.data();
            [[maybe_unused]] const int n1 = static_cast<int>(phi_.size());
            [[maybe_unused]] const int nb1 = static_cast<int>(bv_.size());
            [[maybe_unused]] const int nfl = static_cast<int>(flux_.size());
#pragma omp target enter data map(to : own[0 : nf], nei[0 : nnei], start[0 : nc + 1],              \
                                      faces[0 : nfc], w[0 : nnei], sf[0 : 3 * nf], cf[0 : 3 * nf], \
                                      cc[0 : 3 * nc], vol[0 : nc], kv[0 : 3 * nf],                 \
                                      lsq_dd[0 : 3 * nf], lsq_inv[0 : 9 * nc])                     \
    map(alloc : phi[0 : n1], bv[0 : nb1], sg[0 : 3 * n1], vphi[0 : 3 * n1], vbv[0 : 3 * nb1],      \
            vg[0 : 9 * n1], lim[0 : 3 * n1], flux[0 : nfl], coef[0 : nfl], src[0 : 3 * n1])
        }
#endif
    }

    ~OpenMpFv() override {
#ifdef CFD_FV_OFFLOAD
        if (offload_) {
            CFD_FV_LOCALS
            [[maybe_unused]] const int nnei = static_cast<int>(h_.nei.size());
            [[maybe_unused]] const int nfc = static_cast<int>(h_.faces.size());
            [[maybe_unused]] double* phi = phi_.data();
            [[maybe_unused]] double* bv = bv_.data();
            [[maybe_unused]] double* sg = sgrad_.data();
            [[maybe_unused]] double* vphi = vphi_.data();
            [[maybe_unused]] double* vbv = vbv_.data();
            [[maybe_unused]] double* vg = vgrad_.data();
            [[maybe_unused]] double* lim = lim_.data();
            [[maybe_unused]] double* flux = flux_.data();
            [[maybe_unused]] double* coef = coef_.data();
            [[maybe_unused]] double* src = src_.data();
            [[maybe_unused]] const int n1 = static_cast<int>(phi_.size());
            [[maybe_unused]] const int nb1 = static_cast<int>(bv_.size());
            [[maybe_unused]] const int nfl = static_cast<int>(flux_.size());
#pragma omp target exit data map(                                                                  \
        delete : own[0 : nf], nei[0 : nnei], start[0 : nc + 1], faces[0 : nfc], w[0 : nnei],       \
            sf[0 : 3 * nf], cf[0 : 3 * nf], cc[0 : 3 * nc], vol[0 : nc], kv[0 : 3 * nf],           \
            lsq_dd[0 : 3 * nf], lsq_inv[0 : 9 * nc], phi[0 : n1], bv[0 : nb1], sg[0 : 3 * n1],     \
            vphi[0 : 3 * n1], vbv[0 : 3 * nb1], vg[0 : 9 * n1], lim[0 : 3 * n1], flux[0 : nfl],    \
            coef[0 : nfl], src[0 : 3 * n1])
        }
#endif
    }

    void gradient(const double* phi, const double* bv, bool lsq, double* grad) override {
        upload(phi_, phi, h_.nc);
        upload(bv_, bv, h_.nb());
        gradient_loop(phi_.data(), bv_.data(), 1, 0, lsq, sgrad_.data(), 3, 0);
        download(sgrad_, grad, 3 * h_.nc);
    }

    void vector_gradient(const double* u, const double* ub, bool lsq, double* grad) override {
        upload(vphi_, u, 3 * h_.nc);
        upload(vbv_, ub, 3 * h_.nb());
        for (int i = 0; i < 3; ++i) {
            gradient_loop(vphi_.data(), vbv_.data(), 3, i, lsq, vgrad_.data(), 9, 3 * i);
        }
        download(vgrad_, grad, 9 * h_.nc);
        have_vector_ = true;
    }

    void scalar_corrections(const double* phi, const double* bv, bool lsq, const double* flux,
                            const double* gamma, bool linear_upwind, double* source) override {
        upload(phi_, phi, h_.nc);
        upload(bv_, bv, h_.nb());
        gradient_loop(phi_.data(), bv_.data(), 1, 0, lsq, sgrad_.data(), 3, 0);
        upload(flux_, flux, h_.nif);
        upload(coef_, gamma, h_.nif);
        upload(src_, source, h_.nc);
        CFD_FV_LOCALS
        const double* g = sgrad_.data();
        const double* fl = flux_.data();
        const double* ga = coef_.data();
        double* src = src_.data();
#ifdef CFD_FV_OFFLOAD
        if (offload_) {
#pragma omp target teams distribute parallel for
            for (int c = 0; c < nc; ++c) {
                const fvoff::FvArrays m CFD_FV_ARRAYS;
                fvoff::cell_scalar_corrections(m, c, g, fl, ga, linear_upwind, src);
            }
        } else
#endif
        {
            const fvhost::FvArrays m CFD_FV_ARRAYS;
            CFD_FV_HOST_LOOP
            for (int c = 0; c < nc; ++c) {
                fvhost::cell_scalar_corrections(m, c, g, fl, ga, linear_upwind, src);
            }
        }
        download(src_, source, h_.nc);
    }

    void momentum_corrections(const double* flux, const double* nu_face, bool linear_upwind,
                              bool limited, bool venkat, const double* eps2, double central,
                              bool transpose_term, double* source) override {
        const double e0 = venkat ? eps2[0] : 0.0;
        const double e1 = venkat ? eps2[1] : 0.0;
        const double e2 = venkat ? eps2[2] : 0.0;
        if (!have_vector_) {
            throw std::logic_error("DeviceFv::momentum_corrections before vector_gradient");
        }
        upload(flux_, flux, h_.nif);
        upload(coef_, nu_face, h_.nif);
        upload(src_, source, 3 * h_.nc);
        CFD_FV_LOCALS
        const double* u = vphi_.data();
        const double* ub = vbv_.data();
        const double* g = vgrad_.data();
        double* lim = lim_.data();
        const double* fl = flux_.data();
        const double* nu = coef_.data();
        double* src = src_.data();
        const bool use_lim = limited && linear_upwind;
#ifdef CFD_FV_OFFLOAD
        if (offload_) {
            if (use_lim) {
#pragma omp target teams distribute parallel for
                for (int c = 0; c < nc; ++c) {
                    const fvoff::FvArrays m CFD_FV_ARRAYS;
                    fvoff::cell_limiter(m, c, u, ub, g, venkat, e0, e1, e2, lim);
                }
            }
#pragma omp target teams distribute parallel for
            for (int c = 0; c < nc; ++c) {
                const fvoff::FvArrays m CFD_FV_ARRAYS;
                fvoff::cell_momentum_corrections(m, c, g, lim, use_lim, fl, nu, linear_upwind,
                                                 central, u, transpose_term, src);
            }
        } else
#endif
        {
            const fvhost::FvArrays m CFD_FV_ARRAYS;
            if (use_lim) {
                CFD_FV_HOST_LOOP
                for (int c = 0; c < nc; ++c) {
                    fvhost::cell_limiter(m, c, u, ub, g, venkat, e0, e1, e2, lim);
                }
            }
            CFD_FV_HOST_LOOP
            for (int c = 0; c < nc; ++c) {
                fvhost::cell_momentum_corrections(m, c, g, lim, use_lim, fl, nu, linear_upwind,
                                                  central, u, transpose_term, src);
            }
        }
        download(src_, source, 3 * h_.nc);
    }

    [[nodiscard]] bool on_device() const override { return offload_; }

private:
    void gradient_loop(const double* phi, const double* bv, int stride, int comp, bool lsq,
                       double* out, int ostride, int ooff) {
        CFD_FV_LOCALS
#ifdef CFD_FV_OFFLOAD
        if (offload_) {
#pragma omp target teams distribute parallel for
            for (int c = 0; c < nc; ++c) {
                const fvoff::FvArrays m CFD_FV_ARRAYS;
                fvoff::cell_gradient(m, c, phi, bv, stride, comp, lsq, out, ostride, ooff);
            }
            return;
        }
#endif
        const fvhost::FvArrays m CFD_FV_ARRAYS;
        CFD_FV_HOST_LOOP
        for (int c = 0; c < nc; ++c) {
            fvhost::cell_gradient(m, c, phi, bv, stride, comp, lsq, out, ostride, ooff);
        }
    }

    // Copies n values into the buffer (and to the device).
    void upload(std::vector<double>& buf, const double* data, int n) {
        if (n <= 0) {
            return;
        }
        std::copy(data, data + n, buf.begin());
#ifdef CFD_FV_OFFLOAD
        if (offload_) {
            [[maybe_unused]] double* p = buf.data();
#pragma omp target update to(p[0 : n])
        }
#endif
    }

    // Copies n values of the buffer (from the device) to data.
    void download(std::vector<double>& buf, double* data, int n) {
        if (n <= 0) {
            return;
        }
#ifdef CFD_FV_OFFLOAD
        if (offload_) {
            [[maybe_unused]] double* p = buf.data();
#pragma omp target update from(p[0 : n])
        }
#endif
        std::copy(buf.begin(), buf.begin() + n, data);
    }

    FvHostMesh h_;
    std::vector<double> phi_;
    std::vector<double> bv_;
    std::vector<double> sgrad_;
    std::vector<double> vphi_;
    std::vector<double> vbv_;
    std::vector<double> vgrad_;
    std::vector<double> lim_;
    std::vector<double> flux_;
    std::vector<double> coef_;
    std::vector<double> src_;
    bool offload_ = false;
    bool have_vector_ = false;
};

} // namespace

std::unique_ptr<DeviceFv::Impl> make_fv_openmp(const FvMeshView& mesh) {
    return std::make_unique<OpenMpFv>(mesh);
}

} // namespace cfd::gpu::detail
