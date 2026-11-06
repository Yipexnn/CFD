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
// NOLINTEND
