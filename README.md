# StencilCore.jl

[![Build Status](https://github.com/vlc1/StencilCore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vlc1/StencilCore.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://vlc1.github.io/StencilCore.jl/dev/)

The shared **type vocabulary** at the root of a three-package stack for
variable-coefficient Cartesian stencils and the symbolic algebra that builds
them. StencilCore owns the types; it has no assembly and depends only on
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl).

```
StencilCore              types: AccessStyle, AbstractStencil, AbstractTerm{T},
  │                             ArrayOrTermLike, StaticShift, LinearStencil,
  │                             StarStencil, Stencil, as_linear / as_star
  ├── StencilAssembly      CSC assembly  (build / assemble / update!)
  └── StencilCalculus      symbolic CAS  (simplify, differentiate, materialize)
```

## What it provides

- **`AccessStyle`** trait (`ColumnAccess` / `RowAccess`) + `AbstractStencil{S}`.
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
```

See [`AGENTS.md`](AGENTS.md) for the canonical design decisions.
