# Phase 3 – Turbulence (k-ω SST)

Goal: RANS turbulence with the k-ω SST model (Menter 2003, OpenFOAM variant), automatic wall treatment and y+ output.

## Design
- **FlowSolver**
  - momentum diffusion with ν_eff = ν + ν_t (face-interpolated), plus the explicit ∇·(ν_eff (∇U)ᵀ) term
  - `body_force` (driving pressure gradient for periodic-like cases)
  - new boundary kind `Empty`: zero flux, zero gradient. Only valid for directions with a single cell layer (1D/2D cases).
- **TurbulenceModel** interface: `correct(const FlowSolver&)` and `nut()`; wall ν_t per boundary face. FlowSolver calls it after each SIMPLE iteration / PISO step.
- **Scalar transport** helper: convection (upwind / linear-upwind), diffusion with Γ_f, implicit/explicit sources, fixed-value / zero-gradient BCs, fixed cell values, under-relaxation.
- **Wall distance**: Poisson method (∇²φ = −1), plus exact normal distance for wall-adjacent cells.
- **k-ω SST**
  - F1/F2 blending, production limiter, cross-diffusion term
  - ω in wall cells: √(ω_vis² + ω_log²) (blended)
  - k at walls: zero gradient
  - ν_t,wall from the log law when y+ > 11 (nutk wall function), otherwise 0
  - production in wall cells from the log law when y+ > 11
  - inlet k/ω from turbulence intensity and viscosity ratio
- **y+ field** for wall-adjacent cells.

## Validation
- **Fully developed channel, Re_τ = 395** (1D: one cell in x/z, Empty sides, symmetry at the centreline, body force u_τ²/h):
  - force balance: wall shear u_τ = 1 within 1 %
  - log law at y+ ≈ 30–150 within 5 %
  - U_c+ ≈ 20 and U_b+ ≈ 17.5 (DNS, Moser et al. 1999) within 6 %
- Same channel with a coarse mesh (wall functions, y+ ≈ 30–60): U_b+ within 8 %
- **Backward-facing step**: optional/slow, after a performance pass (AMG)

## Definition of Done
- [ ] CI green, Phase 2 laminar tests unchanged
- [x] Channel validations pass (Re_τ 395 resolved: U_b+ 17.9, U_c+ 20.1; Re_τ 2000 wall functions: U_b+ 22.4 vs Dean 21.7)
- [x] y+ reported (KOmegaSst::y_plus)
