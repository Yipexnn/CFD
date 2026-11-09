// Krylov kernels; included by krylov_openmp.cpp once per execution space with CFD_LOOP /
// CFD_LOOP_SUM defined (host threads or OpenMP target offload).
// NOLINTBEGIN
struct Arrays {
    int n;
    const int* rp;
    const int* ci;
    const double* va;
};

// y = A x
void spmv(const Arrays& a, const double* x, double* y) {
    const int n = a.n;
    const int* rp = a.rp;
    const int* ci = a.ci;
    const double* va = a.va;
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            s += va[k] * x[ci[k]];
        }
        y[i] = s;
    }
}

double dot(int n, const double* x, const double* y) {
    double sum = 0.0;
    CFD_LOOP_SUM(sum)
    for (int i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

// y = x * d (element-wise)
void scale(int n, const double* x, const double* d, double* y) {
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] * d[i];
    }
}

SolveStats cg(const Arrays& a, const double* b, double* x, const double* dinv, double* r, double* z,
              double* p, double* q, const SolveControl& control) {
    const int n = a.n;
    SolveStats st;
    spmv(a, x, q);
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - q[i];
    }
    double res = std::sqrt(dot(n, r, r));
    if (res <= control.abs_tolerance) {
        st.residual = res;
        st.converged = true;
        return st;
    }
    scale(n, r, dinv, z);
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        p[i] = z[i];
    }
    double rz = dot(n, r, z);
    for (st.iterations = 1; st.iterations <= control.max_iterations; ++st.iterations) {
        spmv(a, p, q);
        const double pq = dot(n, p, q);
        if (pq == 0.0) {
            break;
        }
        const double alpha = rz / pq;
        CFD_LOOP
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        res = std::sqrt(dot(n, r, r));
        if (res <= control.abs_tolerance) {
            st.converged = true;
            break;
        }
        scale(n, r, dinv, z);
        const double rz_new = dot(n, r, z);
        const double beta = rz_new / rz;
        rz = rz_new;
        CFD_LOOP
        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }
    }
    st.iterations = std::min(st.iterations, control.max_iterations);
    st.residual = res;
    return st;
}

// Right-preconditioned BiCGStab (van der Vorst) with Jacobi preconditioning.
SolveStats bicgstab(const Arrays& a, const double* b, double* x, const double* dinv, double* r,
                    double* r0, double* p, double* v, double* s, double* t, double* ph, double* sh,
                    const SolveControl& control) {
    const int n = a.n;
    SolveStats st;
    spmv(a, x, v);
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - v[i];
        r0[i] = r[i];
        p[i] = 0.0;
        v[i] = 0.0;
    }
    double res = std::sqrt(dot(n, r, r));
    if (res <= control.abs_tolerance) {
        st.residual = res;
        st.converged = true;
        return st;
    }
    double rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    for (st.iterations = 1; st.iterations <= control.max_iterations; ++st.iterations) {
        const double rho_new = dot(n, r0, r);
        if (rho_new == 0.0 || omega == 0.0) {
            break;
        }
        const double beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;
        CFD_LOOP
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        scale(n, p, dinv, ph);
        spmv(a, ph, v);
        const double r0v = dot(n, r0, v);
        if (r0v == 0.0) {
            break;
        }
        alpha = rho / r0v;
        CFD_LOOP
        for (int i = 0; i < n; ++i) {
            s[i] = r[i] - alpha * v[i];
        }
        const double s_norm = std::sqrt(dot(n, s, s));
        if (s_norm <= control.abs_tolerance) {
            CFD_LOOP
            for (int i = 0; i < n; ++i) {
                x[i] += alpha * ph[i];
            }
            res = s_norm;
            st.converged = true;
            break;
        }
        scale(n, s, dinv, sh);
        spmv(a, sh, t);
        const double tt = dot(n, t, t);
        omega = tt > 0.0 ? dot(n, t, s) / tt : 0.0;
        CFD_LOOP
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * ph[i] + omega * sh[i];
            r[i] = s[i] - omega * t[i];
        }
        res = std::sqrt(dot(n, r, r));
        if (res <= control.abs_tolerance) {
            st.converged = true;
            break;
        }
    }
    st.iterations = std::min(st.iterations, control.max_iterations);
    st.residual = res;
    return st;
}

// ---- AMG-preconditioned CG ----
struct AmgLevel {
    Arrays a;
    Arrays p; // n_l rows (prolongation from level l + 1)
    Arrays r; // n_{l+1} rows
    const double* dinv;
    double* x;
    double* b;
    double* t;
};

void amg_jacobi(const AmgLevel& lv, int sweeps, double w) {
    const int n = lv.a.n;
    double* x = lv.x;
    const double* b = lv.b;
    double* t = lv.t;
    const double* d = lv.dinv;
    for (int s = 0; s < sweeps; ++s) {
        spmv(lv.a, x, t);
        CFD_LOOP
        for (int i = 0; i < n; ++i) {
            x[i] += w * d[i] * (b[i] - t[i]);
        }
    }
}

// x = M b (dense, row-major n x n)
void dense_mv(int n, const double* m, const double* b, double* x) {
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) {
            s += m[static_cast<long long>(i) * n + j] * b[j];
        }
        x[i] = s;
    }
}

void vcycle(const AmgLevel* levels, int n_levels, int l, const double* coarse_inverse, int sweeps,
            double w) {
    const AmgLevel& lv = levels[l];
    const int n = lv.a.n;
    if (l == n_levels - 1) {
        dense_mv(n, coarse_inverse, lv.b, lv.x);
        return;
    }
    double* x = lv.x;
    double* t = lv.t;
    const double* b = lv.b;
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        x[i] = 0.0;
    }
    amg_jacobi(lv, sweeps, w);
    spmv(lv.a, x, t);
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        t[i] = b[i] - t[i];
    }
    spmv(lv.r, t, levels[l + 1].b);
    vcycle(levels, n_levels, l + 1, coarse_inverse, sweeps, w);
    spmv(lv.p, levels[l + 1].x, t);
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        x[i] += t[i];
    }
    amg_jacobi(lv, sweeps, w);
}

SolveStats amg_cg(const AmgLevel* levels, int n_levels, const double* coarse_inverse, int sweeps,
                  double w, const double* b, double* x, double* r, double* p, double* q,
                  const SolveControl& control) {
    const Arrays& a = levels[0].a;
    const int n = a.n;
    SolveStats st;
    spmv(a, x, q);
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - q[i];
    }
    double res = std::sqrt(dot(n, r, r));
    if (res <= control.abs_tolerance) {
        st.residual = res;
        st.converged = true;
        return st;
    }
    double* zb = levels[0].b;
    double* z = levels[0].x;
    auto precondition = [&]() {
        CFD_LOOP
        for (int i = 0; i < n; ++i) {
            zb[i] = r[i];
        }
        vcycle(levels, n_levels, 0, coarse_inverse, sweeps, w);
    };
    precondition();
    CFD_LOOP
    for (int i = 0; i < n; ++i) {
        p[i] = z[i];
    }
    double rz = dot(n, r, z);
    for (st.iterations = 1; st.iterations <= control.max_iterations; ++st.iterations) {
        spmv(a, p, q);
        const double pq = dot(n, p, q);
        if (pq == 0.0) {
            break;
        }
        const double alpha = rz / pq;
        CFD_LOOP
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        res = std::sqrt(dot(n, r, r));
        if (res <= control.abs_tolerance) {
            st.converged = true;
            break;
        }
        precondition();
        const double rz_new = dot(n, r, z);
        const double beta = rz_new / rz;
        rz = rz_new;
        CFD_LOOP
        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }
    }
    st.iterations = std::min(st.iterations, control.max_iterations);
    st.residual = res;
    return st;
}
// NOLINTEND
