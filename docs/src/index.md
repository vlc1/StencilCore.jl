# StencilCore.jl

StencilCore is the shared **type vocabulary** at the root of a small stack for
building and assembling sparse operators on structured (Cartesian) meshes. It
owns the types and a small scalar CAS; it depends only on
[StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) and
[AbstractTrees.jl](https://github.com/JuliaCollections/AbstractTrees.jl).

```
StencilCore              stencil types: AccessStyle, AbstractStencil, AbstractTerm{T},
  │                                     ArrayOrTermLike, StaticShift, LinearStencil,
  │                                     StarStencil, Stencil, as_linear / as_star
  │                      scalar CAS:    AbstractScalar{T}, Symbolic, Constant,
  │                                     Scaling, Scalar, Null, Unity,
  │                                     simplify, materialize, differentiate
  ├── StencilAssembly      CSC assembly  (build / assemble / update!)
  └── StencilCalculus      term-level CAS (simplify, differentiate, materialize) + bridge
```

You probably want one of the leaves: **[StencilCalculus](https://vlc1.github.io/StencilCalculus.jl/dev/)**
to *build* a stencil by differentiating a symbolic grid expression, or
**[StencilAssembly](https://vlc1.github.io/StencilAssembly.jl/dev/)** to
*assemble* a stencil into a sparse matrix. This page explains the vocabulary
both of them speak.

## Why a stencil needs a vocabulary

On a **structured mesh**, a discrete field is stored as an ordinary N-D array
and the connectivity is implicit: the neighbours of node `(i, j)` are
`(i±1, j)` and `(i, j±1)`, reached by *offsetting the index*. A great many
numerical operations — finite differences, Laplacians — are **stencils**: the
same local formula relating a cell to its neighbours, applied at every point.

A stencil, then, is just a collection of `(offset, coefficient)` pairs. The two
facts about that collection live at very different levels:

- the **offsets** are *structure* — a tiny, compile-time-known set that fixes
  the sparsity pattern and the index arithmetic;
- the **coefficients** are *data* — arbitrary numbers (often position-dependent,
  e.g. a density-weighted gradient `(ϕ[i] − ϕ[i−1]) / ρ[i]`).

!!! note "Motto"
    **Type parameters are structure; values are data.**

StencilCore encodes that split directly: offsets are type-level
([`StaticShift`](@ref)), coefficients are ordinary (or lazy) arrays, and a
stencil's element type is an `SVector` of all the per-offset coefficients on a
single column.

## Two parallel algebras

The CAS half of StencilCore is built from two sibling type hierarchies that
mirror each other:

| role                      | `AbstractScalar` (Core)                                  | `AbstractTerm` (StencilCalculus)                       |
|---------------------------|----------------------------------------------------------|--------------------------------------------------------|
| named substitution leaf   | [`Symbolic`](@ref)`{S, T}`                               | `Slot{S, T}`                                           |
| literal carrier           | [`Constant`](@ref)`{T}` with `val::T`                    | — (literals enter terms via `Fill(Constant(…))`)       |
| scaled identity           | [`Scaling`](@ref)`{T, V<:Number}` (`val * one(T)`)       | —                                                      |
| interior tree node        | [`Scalar`](@ref)`{F, A<:Tuple{Vararg{AbstractScalar}}}`  | `Term{F, A<:Tuple{Vararg{AbstractTerm}}}`              |
| additive identity         | [`Null`](@ref)`{T}` (structural)                         | `Zero{T}`                                              |
| multiplicative identity   | [`Unity`](@ref)`{T}` (structural; requires `one(T)`)     | `One{T}`                                               |
| broadcast bridge          | —                                                        | `Fill{T}` (lives in StencilCalculus, wraps an `AbstractScalar` *or* a literal)|

An [`AbstractTerm{T}`](@ref) is *array-like* — a dimension-/size-less
collection that materializes to a per-cell array. An [`AbstractScalar{T}`](@ref)
is *not* array-like — it materializes to a single value. The two are bridged
in StencilCalculus by the `Fill` term, which broadcasts a scalar across the
grid. Scalars never appear inside a `Term` directly — they enter via `Fill`,
so `Term.args` stays `Tuple{Vararg{AbstractTerm}}`.

## The pieces

- **[`AccessStyle`](@ref)** — a trait (`ColumnAccess` / `RowAccess`) recording
  whether a stencil's coefficients are anchored at the column or the row index;
  `ColumnAccess` is what a compressed-sparse-column matrix wants.
- **[`AbstractTerm`](@ref)`{T}`** — "a dimension-/size-less array-like object
  whose element type is `T`", and `ArrayOrTermLike{T} = Union{AbstractArray{T},
  AbstractTerm{T}}`, the coefficient type of every stencil. A stencil is
  *assemblable* with a concrete-array coefficient and *symbolic* with a term
  coefficient.
- **[`AbstractScalar`](@ref)`{T}`** — the cell-level scalar algebra. Concrete
  leaves ([`Symbolic`](@ref), [`Constant`](@ref), [`Scaling`](@ref),
  [`Null`](@ref), [`Unity`](@ref)) plus the interior [`Scalar`](@ref) node
  have their own operator overloads, a *structural* [`simplify`](@ref)
  rewriter (identities by dispatch on `Null`/`Unity`; folding combines values
  but does not match on them), a [`materialize`](@ref) reduction, and a
  Jacobian-aware [`differentiate`](@ref) chain rule — all parallel to the
  term side, all living entirely in scalar-land.
- **[`StaticPair`](@ref)`{D,O}` / [`StaticShift`](@ref)** — type-level lattice
  offsets with a `+`/`-`/`*` algebra, the zero shift [`ô`](@ref), and basis
  shifts `ê₁ … ê₉`. They read like lattice vectors:

  ```julia
  using StencilCore
  3ê₁ + ê₂        # StaticShift{Tuple{StaticPair{1,3}, StaticPair{2,1}}}
  ```

- **[`LinearStencil`](@ref)** (contiguous offsets along one axis), the
  interlaced **[`StarStencil`](@ref)** (the whole star per cell, with the
  diagonal as an explicit slot), and the general **[`Stencil`](@ref)** (an
  arbitrary reverse-lex offset list — the form differentiation produces).
- **[`as_linear`](@ref) / [`as_star`](@ref)** — narrow a general `Stencil` to an
  assemblable `LinearStencil` / `StarStencil` by a shift-pattern match and a
  verbatim coefficient copy.

```julia
using StencilCore, StaticArrays
# A 1-D forward-difference stencil: offsets 0 and 1, one SVector of
# coefficients per column.
st = LinearStencil{1}(SUnitRange(0, 1), fill(SVector(-1.0, 1.0), 5))

# The scalar CAS, on its own:
@symbolic τ Float64
α = Constant(2)
differentiate(τ * α + α, τ)        # === Constant{Float64}(2.0)
                                   # eltype promotes via Scalar(*); broadcasts to `2.0` per cell once wrapped in Fill
```

See the [Guide](@ref) for worked examples and the [API reference](@ref) for the
full surface.
