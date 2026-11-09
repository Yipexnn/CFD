// Per-cell finite-volume kernels shared by the host, OpenMP-offload and CUDA backends
// (included with CFD_HD defined: __host__ __device__ for CUDA, empty otherwise). Each cell
// gathers over its faces in ascending face order, the order of the serial CPU face loops.
// NOLINTBEGIN
struct FvArrays {
    int nc;
    int nif;
    int nf;
    const int* own;
    const int* nei;
    const int* start;
    const int* faces;
    const double* w;
    const double* sf;
    const double* cf;
    const double* cc;
    const double* vol;
    const double* kv;
    const double* lsq_d;
    const double* lsq;
};

// Gradient of component comp of a field with `stride` values per cell / boundary face, written
// to out[c * ostride + ooff + j].
CFD_HD inline void cell_gradient(const FvArrays& m, int c, const double* phi, const double* bv,
                                 int stride, int comp, bool least_squares, double* out, int ostride,
                                 int ooff) {
    double g0 = 0.0;
    double g1 = 0.0;
    double g2 = 0.0;
    const double pc = phi[c * stride + comp];
    for (int k = m.start[c]; k < m.start[c + 1]; ++k) {
        const int f = m.faces[k];
        if (least_squares) {
            const double dv = f < m.nif
                                  ? phi[m.nei[f] * stride + comp] - phi[m.own[f] * stride + comp]
                                  : bv[(f - m.nif) * stride + comp] - pc;
            g0 += m.lsq_d[3 * f] * dv;
            g1 += m.lsq_d[3 * f + 1] * dv;
            g2 += m.lsq_d[3 * f + 2] * dv;
        } else if (f < m.nif) {
            const double wf = m.w[f];
            const double pf =
                wf * phi[m.own[f] * stride + comp] + (1.0 - wf) * phi[m.nei[f] * stride + comp];
            if (m.own[f] == c) {
                g0 += m.sf[3 * f] * pf;
                g1 += m.sf[3 * f + 1] * pf;
                g2 += m.sf[3 * f + 2] * pf;
            } else {
                g0 -= m.sf[3 * f] * pf;
                g1 -= m.sf[3 * f + 1] * pf;
                g2 -= m.sf[3 * f + 2] * pf;
            }
        } else {
            const double b = bv[(f - m.nif) * stride + comp];
            g0 += m.sf[3 * f] * b;
            g1 += m.sf[3 * f + 1] * b;
            g2 += m.sf[3 * f + 2] * b;
        }
    }
    double* o = out + c * ostride + ooff;
    if (least_squares) {
        const double* a = m.lsq + 9 * c; // column-major
        o[0] = a[0] * g0 + a[3] * g1 + a[6] * g2;
        o[1] = a[1] * g0 + a[4] * g1 + a[7] * g2;
        o[2] = a[2] * g0 + a[5] * g1 + a[8] * g2;
    } else {
        const double v = m.vol[c];
        o[0] = g0 / v;
        o[1] = g1 / v;
        o[2] = g2 / v;
    }
}

// Barth-Jespersen limiter (3 components) of cell c for the vector field u (3 per cell) with
// boundary values ub and gradient g (9 per cell, column-major, g(j, i) = du_i / dx_j).
// venkat: Venkatakrishnan with thresholds e0, e1, e2 (eps^2 per component).
CFD_HD inline void cell_limiter(const FvArrays& m, int c, const double* u, const double* ub,
                                const double* g, bool venkat, double e0, double e1, double e2,
                                double* lim) {
    const double eps2[3] = {e0, e1, e2};
    double lo[3];
    double hi[3];
    for (int i = 0; i < 3; ++i) {
        lo[i] = u[3 * c + i];
        hi[i] = u[3 * c + i];
    }
    for (int k = m.start[c]; k < m.start[c + 1]; ++k) {
        const int f = m.faces[k];
        const double* v =
            f < m.nif ? u + 3 * (m.own[f] == c ? m.nei[f] : m.own[f]) : ub + 3 * (f - m.nif);
        for (int i = 0; i < 3; ++i) {
            lo[i] = v[i] < lo[i] ? v[i] : lo[i];
            hi[i] = v[i] > hi[i] ? v[i] : hi[i];
        }
    }
    double l[3] = {1.0, 1.0, 1.0};
    const double* gc = g + 9 * c;
    for (int k = m.start[c]; k < m.start[c + 1]; ++k) {
        const int f = m.faces[k];
        const double r0 = m.cf[3 * f] - m.cc[3 * c];
        const double r1 = m.cf[3 * f + 1] - m.cc[3 * c + 1];
        const double r2 = m.cf[3 * f + 2] - m.cc[3 * c + 2];
        for (int i = 0; i < 3; ++i) {
            const double d = gc[3 * i] * r0 + gc[3 * i + 1] * r1 + gc[3 * i + 2] * r2;
            if (d < 1e-300 && d > -1e-300) {
                continue;
            }
            const double bound = d > 0.0 ? hi[i] - u[3 * c + i] : lo[i] - u[3 * c + i];
            const double q = venkat ? (bound * bound + eps2[i] + 2.0 * d * bound) /
                                          (bound * bound + 2.0 * d * d + d * bound + eps2[i])
                                    : (bound / d > 0.0 ? bound / d : 0.0);
            l[i] = q < l[i] ? q : l[i];
        }
    }
    for (int i = 0; i < 3; ++i) {
        lim[3 * c + i] = l[i];
    }
}

// Explicit internal-face fluxes of a scalar equation with the cell gradient g (3 per cell).
CFD_HD inline void cell_scalar_corrections(const FvArrays& m, int c, const double* g,
                                           const double* flux, const double* gamma,
                                           bool linear_upwind, double* source) {
    double s = 0.0;
    for (int k = m.start[c]; k < m.start[c + 1]; ++k) {
        const int f = m.faces[k];
        if (f >= m.nif) {
            break; // boundary faces follow the internal ones
        }
        const int p = m.own[f];
        const int n = m.nei[f];
        const double fl = flux[f];
        double e = 0.0;
        if (linear_upwind) {
            const int up = fl >= 0.0 ? p : n;
            const double r0 = m.cf[3 * f] - m.cc[3 * up];
            const double r1 = m.cf[3 * f + 1] - m.cc[3 * up + 1];
            const double r2 = m.cf[3 * f + 2] - m.cc[3 * up + 2];
            e -= fl * (g[3 * up] * r0 + g[3 * up + 1] * r1 + g[3 * up + 2] * r2);
        }
        const double wf = m.w[f];
        const double gf0 = wf * g[3 * p] + (1.0 - wf) * g[3 * n];
        const double gf1 = wf * g[3 * p + 1] + (1.0 - wf) * g[3 * n + 1];
        const double gf2 = wf * g[3 * p + 2] + (1.0 - wf) * g[3 * n + 2];
        e += gamma[f] * (m.kv[3 * f] * gf0 + m.kv[3 * f + 1] * gf1 + m.kv[3 * f + 2] * gf2);
        s = p == c ? s + e : s - e;
    }
    source[c] += s;
}

// Explicit internal-face fluxes of the momentum equation (see DeviceFv::momentum_corrections).
// central > 0: face value blended with linear interpolation (LUST: 0.75), needs u (3 per cell).
CFD_HD inline void cell_momentum_corrections(const FvArrays& m, int c, const double* g,
                                             const double* lim, bool limited, const double* flux,
                                             const double* nu_face, bool linear_upwind,
                                             double central, const double* u, bool transpose_term,
                                             double* source) {
    double s[3] = {0.0, 0.0, 0.0};
    for (int k = m.start[c]; k < m.start[c + 1]; ++k) {
        const int f = m.faces[k];
        if (f >= m.nif) {
            break;
        }
        const int p = m.own[f];
        const int n = m.nei[f];
        const double fl = flux[f];
        double e[3] = {0.0, 0.0, 0.0};
        if (linear_upwind) {
            const int up = fl >= 0.0 ? p : n;
            const double r0 = m.cf[3 * f] - m.cc[3 * up];
            const double r1 = m.cf[3 * f + 1] - m.cc[3 * up + 1];
            const double r2 = m.cf[3 * f + 2] - m.cc[3 * up + 2];
            const double* gu = g + 9 * up;
            for (int i = 0; i < 3; ++i) {
                double d = gu[3 * i] * r0 + gu[3 * i + 1] * r1 + gu[3 * i + 2] * r2;
                if (limited) {
                    d = lim[3 * up + i] * d;
                }
                if (central > 0.0) {
                    const double wc = m.w[f];
                    const double uf = wc * u[3 * p + i] + (1.0 - wc) * u[3 * n + i];
                    d = (1.0 - central) * d + central * (uf - u[3 * up + i]);
                }
                e[i] -= fl * d;
            }
        }
        const double wf = m.w[f];
        double gf[9];
        for (int j = 0; j < 9; ++j) {
            gf[j] = wf * g[9 * p + j] + (1.0 - wf) * g[9 * n + j];
        }
        const double nu = nu_face[f];
        const double k0 = m.kv[3 * f];
        const double k1 = m.kv[3 * f + 1];
        const double k2 = m.kv[3 * f + 2];
        for (int i = 0; i < 3; ++i) {
            e[i] += nu * (gf[3 * i] * k0 + gf[3 * i + 1] * k1 + gf[3 * i + 2] * k2);
        }
        if (transpose_term) {
            const double a0 = m.sf[3 * f];
            const double a1 = m.sf[3 * f + 1];
            const double a2 = m.sf[3 * f + 2];
            for (int i = 0; i < 3; ++i) {
                e[i] += nu * (gf[i] * a0 + gf[i + 3] * a1 + gf[i + 6] * a2);
            }
        }
        for (int i = 0; i < 3; ++i) {
            s[i] = p == c ? s[i] + e[i] : s[i] - e[i];
        }
    }
    for (int i = 0; i < 3; ++i) {
        source[3 * c + i] += s[i];
    }
}
// NOLINTEND
