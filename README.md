# StencilCore.jl

[![Build Status](https://github.com/vlc1/StencilCore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vlc1/StencilCore.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://vlc1.github.io/StencilCore.jl/dev/)

The shared **type vocabulary** at the root of a three-package stack for
variable-coefficient Cartesian stencils and the symbolic algebra that builds
them. StencilCore owns the types; it has no assembly and depends only on
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).

```
StencilCore         stencil types: AccessStyle, AbstractStencil, AbstractTerm{T},
  │                                ArrayOrTermLike, StaticShift, LinearStencil,
  │                                StarStencil, Stencil, as_linear / as_star
  │                  scalar CAS:   AbstractScalar{T}, Symbolic, Constant, Scaling,
  │                                Null, Unity, Scalar, @symbolic,
  │                                simplify, materialize, differentiate
  ├── StencilAssembly      CSC assembly  (build / assemble / update!)
  └── StencilCalculus      term-level CAS (simplify, differentiate, materialize) + bridge
```

## What it provides

- **`AccessStyle`** trait (`ColumnAccess` / `RowAccess`) + `AbstractStencil{S, T}` —
  `T` is the coefficient eltype (the linear-map space, mirroring `Unity{T}` in
  scalar-land); accessed via `Base.eltype(::AbstractStencil)`.
- **`AbstractTerm{T}`** — a dimension-/size-less array-like with element type
  `T` — and `ArrayOrTermLike{T} = Union{AbstractArray{T}, AbstractTerm{T}}`, the
  coefficient type of every stencil (a stencil is *assemblable* with a concrete
  array coefficient, *symbolic* with a term coefficient).
- **`StaticPair{D,O}` / `StaticShift`** — type-level lattice offsets with a
  `+`/`-`/`*` algebra, the zero shift `ô`, and basis shifts `ê₁ … ê₉`.
- **`LinearStencil`** (contiguous offsets along one axis), the interlaced
  **`StarStencil`** (whole star per cell, explicit diagonal), and the general
  **`Stencil`** (arbitrary reverse-lex offset list).
- **`as_linear` / `as_star`** — narrow a general `Stencil` to an assemblable
  type by a shift-pattern match and a verbatim coefficient copy.
- **`AbstractScalar{T}`** — the cell-level scalar algebra. Concrete leaves
  `Symbolic`, `Constant`, `Scaling`, `Null`, `Unity` plus the interior
  `Scalar` node, with operator overloads, a structural `simplify`,
  `materialize`, and a Jacobian-aware `differentiate` (including the
  `SVector → SMatrix` self-derivative case).

## Install

The three packages are unregistered and resolve each other through relative
`[sources]` paths, so clone them **side by side**:

```
git clone …/StencilCore
git clone …/StencilAssembly
git clone …/StencilCalculus
```

Then `]dev /path/to/StencilCore` (or add the others, whose `[sources]` point at
`../StencilCore`).

## Example

```julia
using StencilCore, StaticArrays
# A 1-D forward difference stencil (offsets 0,1; one SVector of coeffs per cell).
st = LinearStencil{1}(SUnitRange(0, 1), fill(SVector(-1.0, 1.0), 5))

# Type-level offsets read like lattice vectors:
3ê₁ + ê₂                      # StaticShift{Tuple{StaticPair{1,3}, StaticPair{2,1}}}

# The scalar CAS, on its own:
@symbolic τ Float64
differentiate(2τ + τ * τ, τ)                  # 2 + 2τ  (simplified)
differentiate(Constant(2.0), τ)               # Null{Float64}() — no dependence

# Vector-valued symbols: the Jacobian lands in the matching square SMatrix.
@symbolic x SVector{2, Float64}
differentiate(2x, x)                          # Scaling{SMatrix{2,2,Float64,4}}(2)
```

See [`AGENTS.md`](AGENTS.md) for the canonical design decisions.
