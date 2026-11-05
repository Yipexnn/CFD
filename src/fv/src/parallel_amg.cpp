#include <algorithm>
#include <cfd/fv/parallel_amg.hpp>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

namespace cfd::fv {

namespace {

using Triplet = Eigen::Triplet<Scalar, int>;

// Start index of every process' block (size + 1 entries, last = total).
std::vector<Label> block_offsets(const parallel::Comm& comm, Label n_local) {
    std::vector<Scalar> counts(static_cast<std::size_t>(comm.size()), 0.0);
    counts[static_cast<std::size_t>(comm.rank())] = static_cast<Scalar>(n_local);
    comm.sum(counts.data(), counts.size());
    std::vector<Label> offsets(counts.size() + 1, 0);
    for (std::size_t r = 0; r < counts.size(); ++r) {
        offsets[r + 1] = offsets[r] + static_cast<Label>(std::llround(counts[r]));
    }
    return offsets;
}

// Global index of every local column (owned: offset + i, ghosts: from their owners).
std::vector<Label> column_ids(const parallel::Halo& halo, Label n_cols, Label offset) {
    std::vector<Label> ids(static_cast<std::size_t>(n_cols), -1);
    for (Label i = 0; i < halo.n_owned; ++i) {
        ids[static_cast<std::size_t>(i)] = offset + i;
    }
    halo.exchange(ids);
    return ids;
}

struct SparseRow {
    std::vector<std::pair<Label, Scalar>> entries; // (global column, value)
};

// Sends the rows of the halo's send cells to the neighbours; returns the rows of the
// ghost cells (indexed by local cell - n_owned).
std::vector<SparseRow> exchange_rows(const parallel::Halo& halo,
                                     const std::vector<SparseRow>& owned, Label n_local) {
    std::vector<SparseRow> ghosts(static_cast<std::size_t>(n_local - halo.n_owned));
    if (halo.neighbours.empty()) {
        return ghosts;
    }
    std::vector<int> counts(static_cast<std::size_t>(n_local), 0);
    for (Label i = 0; i < halo.n_owned; ++i) {
        counts[static_cast<std::size_t>(i)] =
            static_cast<int>(owned[static_cast<std::size_t>(i)].entries.size());
    }
    halo.exchange(counts);
    constexpr std::size_t entry = sizeof(Label) + sizeof(Scalar);
    std::vector<int> peers;
    std::vector<std::vector<char>> send(halo.neighbours.size());
    std::vector<std::vector<char>> recv(halo.neighbours.size());
    for (std::size_t k = 0; k < halo.neighbours.size(); ++k) {
        const auto& nb = halo.neighbours[k];
        peers.push_back(nb.rank);
        for (const Label c : nb.send) {
            for (const auto& [col, value] : owned[static_cast<std::size_t>(c)].entries) {
                const std::size_t pos = send[k].size();
                send[k].resize(pos + entry);
                std::memcpy(send[k].data() + pos, &col, sizeof(Label));
                std::memcpy(send[k].data() + pos + sizeof(Label), &value, sizeof(Scalar));
            }
        }
        std::size_t n = 0;
        for (const Label c : nb.recv) {
            n += static_cast<std::size_t>(counts[static_cast<std::size_t>(c)]);
        }
        recv[k].resize(n * entry);
    }
    halo.comm->exchange(peers, send, recv);
    for (std::size_t k = 0; k < halo.neighbours.size(); ++k) {
        std::size_t pos = 0;
        for (const Label c : halo.neighbours[k].recv) {
            SparseRow& row = ghosts[static_cast<std::size_t>(c - halo.n_owned)];
            for (int e = 0; e < counts[static_cast<std::size_t>(c)]; ++e) {
                Label col = 0;
                Scalar value = 0.0;
                std::memcpy(&col, recv[k].data() + pos, sizeof(Label));
                std::memcpy(&value, recv[k].data() + pos + sizeof(Label), sizeof(Scalar));
                row.entries.emplace_back(col, value);
                pos += entry;
            }
        }
    }
    return ghosts;
}

} // namespace

void ParallelAmg::setup(const CsrMatrix& a, const parallel::Halo& halo) {
    comm_ = halo.comm;
    levels_.clear();
    coarse_solver_.reset();
    const parallel::Comm& comm = *comm_;

    Level first;
    first.a = a;
    first.halo = halo;
    std::vector<Label> offsets = block_offsets(comm, static_cast<Label>(a.rows()));
    first.offset = offsets[static_cast<std::size_t>(comm.rank())];
    first.n_global = offsets.back();
    levels_.push_back(std::move(first));

    for (int l = 0; l + 1 < settings_.max_levels; ++l) {
        Level& lv = levels_.back();
        const auto no = static_cast<Label>(lv.a.rows());
        const auto nl = static_cast<Label>(lv.a.cols());
        DenseVector d = DenseVector::Zero(no);
        for (Label i = 0; i < no; ++i) {
            d[i] = lv.a.coeff(i, i);
        }
        lv.inv_diag = d.unaryExpr([](Scalar v) { return v != 0.0 ? 1.0 / v : 0.0; });
        if (lv.n_global <= settings_.coarse_size) {
            break;
        }

        // Local aggregation of the owned block.
        const CsrMatrix a_oo = lv.a.leftCols(no);
        int n_agg = 0;
        const std::vector<int> agg =
            no > 0 ? amg_aggregate(a_oo, d, settings_.strength_threshold, n_agg)
                   : std::vector<int>{};
        const std::vector<Label> coarse_offsets = block_offsets(comm, n_agg);
        const Label n_coarse = coarse_offsets.back();
        if (n_coarse == 0 || n_coarse * 10 >= lv.n_global * 9) {
            break; // no useful coarsening
        }
        const Label coffset = coarse_offsets[static_cast<std::size_t>(comm.rank())];

        // Smoothed prolongation P = (I - omega D^-1 A_oo) P_tent (process-local couplings).
        Scalar rho = 0.0;
        for (Label i = 0; i < no; ++i) {
            Scalar row = 0.0;
            for (CsrMatrix::InnerIterator it(a_oo, i); it; ++it) {
                row += std::abs(it.value());
            }
            rho = std::max(rho, row * std::abs(lv.inv_diag[i]));
        }
        const Scalar omega = rho > 0.0 ? 4.0 / (3.0 * rho) : 0.0;
        std::vector<SparseRow> p_rows(static_cast<std::size_t>(no)); // global coarse columns
        std::vector<Triplet> p_triplets;
        for (Label i = 0; i < no; ++i) {
            std::map<Label, Scalar> row;
            row[agg[static_cast<std::size_t>(i)]] += 1.0;
            for (CsrMatrix::InnerIterator it(a_oo, i); it; ++it) {
                row[agg[static_cast<std::size_t>(it.col())]] -= omega * lv.inv_diag[i] * it.value();
            }
            for (const auto& [c, v] : row) {
                p_triplets.emplace_back(static_cast<int>(i), static_cast<int>(c), v);
                p_rows[static_cast<std::size_t>(i)].entries.emplace_back(coffset + c, v);
            }
        }
        lv.p.resize(no, n_agg);
        lv.p.setFromTriplets(p_triplets.begin(), p_triplets.end());
        lv.r = CsrMatrix(lv.p.transpose());

        // Galerkin coarse rows A_c(I, :) = sum_i P(i, I) sum_j a_ij P(j, :).
        const std::vector<SparseRow> ghost_rows = exchange_rows(lv.halo, p_rows, nl);
        std::vector<std::map<Label, Scalar>> coarse_rows(static_cast<std::size_t>(n_agg));
        for (Label i = 0; i < no; ++i) {
            std::map<Label, Scalar> ap; // row i of A P (global coarse columns)
            for (CsrMatrix::InnerIterator it(lv.a, i); it; ++it) {
                const auto j = static_cast<Label>(it.col());
                const SparseRow& pj = j < no ? p_rows[static_cast<std::size_t>(j)]
                                             : ghost_rows[static_cast<std::size_t>(j - no)];
                for (const auto& [col, v] : pj.entries) {
                    ap[col] += it.value() * v;
                }
            }
            for (CsrMatrix::InnerIterator pit(lv.p, i); pit; ++pit) {
                auto& target = coarse_rows[static_cast<std::size_t>(pit.col())];
                for (const auto& [col, v] : ap) {
                    target[col] += pit.value() * v;
                }
            }
        }

        // Coarse local numbering: owned, then ghosts sorted by (owner, global index).
        auto owner_of = [&](Label gid) {
            return static_cast<int>(
                       std::upper_bound(coarse_offsets.begin(), coarse_offsets.end(), gid) -
                       coarse_offsets.begin()) -
                   1;
        };
        std::vector<Label> ghost_ids;
        std::map<int, std::vector<Label>> send_ids; // rank -> my coarse rows it needs
        for (Label I = 0; I < n_agg; ++I) {
            for (const auto& [col, v] : coarse_rows[static_cast<std::size_t>(I)]) {
                if (col < coffset || col >= coffset + n_agg) {
                    ghost_ids.push_back(col);
                    send_ids[owner_of(col)].push_back(I);
                }
            }
        }
        std::sort(ghost_ids.begin(), ghost_ids.end());
        ghost_ids.erase(std::unique(ghost_ids.begin(), ghost_ids.end()), ghost_ids.end());
        std::map<Label, Label> local_of;
        for (std::size_t g = 0; g < ghost_ids.size(); ++g) {
            local_of[ghost_ids[g]] = n_agg + static_cast<Label>(g);
        }

        Level coarse;
        coarse.offset = coffset;
        coarse.n_global = n_coarse;
        coarse.halo.comm = &comm;
        coarse.halo.n_owned = n_agg;
        std::map<int, parallel::Halo::Neighbour> neighbours;
        for (const Label gid : ghost_ids) { // sorted by gid = sorted by (owner, gid)
            const int q = owner_of(gid);
            neighbours[q].rank = q;
            neighbours[q].recv.push_back(local_of[gid]);
        }
        for (auto& [q, rows] : send_ids) {
            std::sort(rows.begin(), rows.end());
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            neighbours[q].rank = q;
            neighbours[q].send = rows;
        }
        for (auto& [q, nb] : neighbours) {
            coarse.halo.neighbours.push_back(std::move(nb));
        }
        std::vector<Triplet> c_triplets;
        for (Label I = 0; I < n_agg; ++I) {
            for (const auto& [col, v] : coarse_rows[static_cast<std::size_t>(I)]) {
                const Label lc =
                    col >= coffset && col < coffset + n_agg ? col - coffset : local_of[col];
                c_triplets.emplace_back(static_cast<int>(I), static_cast<int>(lc), v);
            }
        }
        coarse.a.resize(n_agg, n_agg + static_cast<Label>(ghost_ids.size()));
        coarse.a.setFromTriplets(c_triplets.begin(), c_triplets.end());
        levels_.push_back(std::move(coarse));
    }
    Level& last = levels_.back();
    if (last.inv_diag.size() != last.a.rows()) {
        last.inv_diag = DenseVector::Zero(last.a.rows());
        for (Label i = 0; i < static_cast<Label>(last.a.rows()); ++i) {
            const Scalar v = last.a.coeff(i, i);
            last.inv_diag[i] = v != 0.0 ? 1.0 / v : 0.0;
        }
    }
    setup_coarsest();
}

void ParallelAmg::setup_coarsest() {
    const Level& c = levels_.back();
    const parallel::Comm& comm = *comm_;
    const std::vector<Label> ids = column_ids(c.halo, static_cast<Label>(c.a.cols()), c.offset);
    std::vector<Label> rows;
    std::vector<Label> cols;
    std::vector<Scalar> vals;
    for (int i = 0; i < c.a.outerSize(); ++i) {
        for (CsrMatrix::InnerIterator it(c.a, i); it; ++it) {
            rows.push_back(c.offset + i);
            cols.push_back(ids[static_cast<std::size_t>(it.col())]);
            vals.push_back(it.value());
        }
    }
    const std::vector<Label> all_rows = comm.gather(rows);
    const std::vector<Label> all_cols = comm.gather(cols);
    const std::vector<Scalar> all_vals = comm.gather(vals);
    if (comm.is_root()) {
        std::vector<Eigen::Triplet<Scalar>> t;
        t.reserve(all_vals.size());
        for (std::size_t k = 0; k < all_vals.size(); ++k) {
            t.emplace_back(static_cast<int>(all_rows[k]), static_cast<int>(all_cols[k]),
                           all_vals[k]);
        }
        Eigen::SparseMatrix<Scalar> m(c.n_global, c.n_global);
        m.setFromTriplets(t.begin(), t.end());
        // Symmetrise (both processes of a coupling compute it with their own rounding).
        Eigen::SparseMatrix<Scalar> sym = 0.5 * (m + Eigen::SparseMatrix<Scalar>(m.transpose()));
        coarse_solver_ = std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<Scalar>>>();
        coarse_solver_->compute(sym);
        if (coarse_solver_->info() != Eigen::Success) {
            coarse_solver_.reset();
        }
    }
}

void ParallelAmg::solve_coarsest(const DenseVector& b, DenseVector& x) const {
    const Level& c = levels_.back();
    const parallel::Comm& comm = *comm_;
    std::vector<Scalar> full(static_cast<std::size_t>(c.n_global), 0.0);
    std::copy(b.data(), b.data() + b.size(), full.begin() + c.offset);
    comm.sum(full.data(), full.size());
    std::vector<Scalar> y(full.size(), 0.0);
    if (comm.is_root()) {
        if (coarse_solver_) {
            const Eigen::Map<const DenseVector> fb(full.data(), c.n_global);
            const DenseVector sol = coarse_solver_->solve(fb);
            std::copy(sol.data(), sol.data() + sol.size(), y.begin());
        }
    }
    comm.sum(y.data(), y.size());
    x = Eigen::Map<const DenseVector>(y.data() + c.offset, b.size());
    if (!coarse_solver_ && comm.is_root() && c.n_global > 0) {
        // Singular coarse matrix: fall back to smoothing (rare).
    }
}

void ParallelAmg::smooth(const Level& lv, const DenseVector& b, DenseVector& x,
                         bool forward) const {
    const auto no = static_cast<Label>(lv.a.rows());
    std::vector<Scalar> ext(static_cast<std::size_t>(lv.a.cols()), 0.0);
    std::copy(x.data(), x.data() + no, ext.begin());
    lv.halo.exchange(ext);
    auto relax = [&](Label i) {
        Scalar sum = b[i];
        for (CsrMatrix::InnerIterator it(lv.a, i); it; ++it) {
            if (it.col() != i) {
                sum -= it.value() * ext[static_cast<std::size_t>(it.col())];
            }
        }
        ext[static_cast<std::size_t>(i)] = sum * lv.inv_diag[i];
    };
    if (forward) {
        for (Label i = 0; i < no; ++i) {
            relax(i);
        }
    } else {
        for (Label i = no - 1; i >= 0; --i) {
            relax(i);
        }
    }
    x = Eigen::Map<const DenseVector>(ext.data(), no);
}

void ParallelAmg::cycle(std::size_t level, const DenseVector& b, DenseVector& x) const {
    const Level& lv = levels_[level];
    if (level + 1 == levels_.size()) {
        solve_coarsest(b, x);
        return;
    }
    for (int s = 0; s < settings_.pre_smooth; ++s) {
        smooth(lv, b, x, true);
    }
    std::vector<Scalar> ext(static_cast<std::size_t>(lv.a.cols()), 0.0);
    std::copy(x.data(), x.data() + x.size(), ext.begin());
    lv.halo.exchange(ext);
    const DenseVector residual =
        b - lv.a * Eigen::Map<const DenseVector>(ext.data(), static_cast<Eigen::Index>(ext.size()));
    const DenseVector rc = lv.r * residual;
    DenseVector ec = DenseVector::Zero(rc.size());
    cycle(level + 1, rc, ec);
    x += lv.p * ec;
    for (int s = 0; s < settings_.post_smooth; ++s) {
        smooth(lv, b, x, false);
    }
}

void ParallelAmg::apply(const DenseVector& r, DenseVector& z) const {
    z = DenseVector::Zero(r.size());
    if (levels_.empty()) {
        z = r;
        return;
    }
    cycle(0, r, z);
}

} // namespace cfd::fv
