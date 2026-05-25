# Guide

This guide builds the StencilCore types by hand. In normal use you would obtain
a stencil from [StencilCalculus](https://vlc1.github.io/StencilCalculus.jl/dev/)
(by differentiation) and hand it to
[StencilAssembly](https://vlc1.github.io/StencilAssembly.jl/dev/) (to assemble a
matrix); here we look under the hood.

## Lattice offsets

Offsets are type-level: a [`StaticShift`](@ref) is a normalized collection of
[`StaticPair`](@ref)`{D,O}` (axis `D`, offset `O`). They support an algebra and
print like lattice vectors, with [`ô`](@ref) the zero shift and `ê₁ … ê₉` the
unit shifts:

```julia
using StencilCore

-2ê₁              # StaticShift{Tuple{StaticPair{1,-2}}}
3ê₁ + ê₂          # StaticShift{Tuple{StaticPair{1,3}, StaticPair{2,1}}}
ê₁ - ê₁           # ô  (the zero shift)
```

Same-axis pairs are summed, zero offsets dropped, and the result is sorted by
axis — so the representation is canonical.

## The scalar algebra

A scalar — a timestep, a Reynolds number, a literal `2.0` — has no spatial
extent. StencilCore models it directly with [`AbstractScalar{T}`](@ref) and a
small CAS that mirrors the term-side one. Six concrete types:

- [`Symbolic{S,T}`](@ref) — a named substitution leaf, like a `Slot` but with
  one materialized value instead of an array.
- [`Constant{T}`](@ref) — a literal leaf carrying its `val::T`. `T` is any
  concrete type — `Number`, `SVector`, anything. This is what numeric (and
  non-Number) literals canonicalise to at the operator boundary.
- [`Scaling{T,V<:Number}`](@ref) — a numerical coefficient times the
  multiplicative identity of shape `T`: materialises to `val * one(T)`. Compact
  carrier for "a scalar multiple of `I`", produced by the chain rule when
  differentiating vector-valued expressions.
- [`Scalar{F,A,T}`](@ref) — an interior tree node `fn(args…)`.
- [`Null{T}`](@ref) / [`Unity{T}`](@ref) — type-level structural `0` / `1`.
  `Unity` requires `one(T)` defined (`Number`, square `SMatrix`); its outer
  ctor routes an SVector through to its square Jacobian space.

The operator overloads build `Scalar` trees, with numeric literals
canonicalising to `Constant`:

```julia
@symbolic τ Float64
α = Constant(2.0)

τ * α + α              # Scalar(+, (Scalar(*, (τ, α)), α))
typeof(τ * α + α)      # Scalar{typeof(+), …, Float64}
eltype(τ * α + α)      # Float64
```

[`simplify`](@ref) post-walks a scalar tree with the same rule-rewriter shape
as the term side. Identities are **purely structural** — `Null + x → x`,
`x * Unity → x` (with an eltype-preservation gate), `x * Null → Null` — never
inspecting `.val`. Folding combines values; it does not match on them.

```julia
simplify(Null{Float64}() + τ)              # τ
simplify(τ * Unity{Float64}())             # τ
simplify(Constant(2.0) + Constant(3.0))    # Constant(5.0)
simplify(τ * Constant(1.0))                # stays a Scalar — numerical `1`
                                           # is not a structural identity
```

[`differentiate`](@ref) on a scalar tree returns an `AbstractScalar` whose
eltype is the **Jacobian** of `eltype(s)` w.r.t. `eltype(v)`. For Number
variables the Jacobian is a Number; for `SVector{N}` variables it is the
square `SMatrix{N, N}`. Mixed shape-classes are rejected at the top level.

```julia
@symbolic τ Float64
differentiate(sin(τ) * τ, τ)               # cos(τ)*τ + sin(τ)*1 simplified
differentiate(Constant(2.0), τ)            # Null{Float64}() — no dependence

# Vector-valued: ∂(2x)/∂x = 2I as a Scaling-carried SMatrix.
@symbolic x SVector{2, Float64}
differentiate(2x, x)                       # Scaling{SMatrix{2,2,Float64,4}}(2)
```

[`materialize`](@ref) reduces a scalar tree to a single value, substituting
`Symbolic` leaves from a `NamedTuple`:

```julia
materialize(τ * Constant(3.0), (τ = 4.0,))  # 12.0
materialize(Unity{SMatrix{2,2,Float64,4}}())  # the 2×2 identity matrix
```

These scalars become **term coefficients** once StencilCalculus wraps them in
a `Fill` — see its docs for the bridge.

## Linear and star stencils

A [`LinearStencil`](@ref) has contiguous offsets along one mesh axis. Its
coefficient is one `SVector` per cell, holding every per-offset coefficient on
that column in ascending-offset order:

```julia
using StencilCore, StaticArrays
n = 5
# offsets 0,1 along axis 1; coefficients (-1, 1) at every column.
fwd = LinearStencil{1}(SUnitRange(0, 1), fill(SVector(-1.0, 1.0), n))
```

A [`StarStencil`](@ref) is the N-D star with symmetric reach `−L … +L` per axis,
stored **interlaced**: a single `SVector{2NL+1}` per cell, in reverse-lex offset
order, with the diagonal as the explicit middle slot — so a free diagonal term
(Helmholtz, an unsteady term) has a home.

```julia
n1, n2 = 5, 4
# 2-D five-point Laplacian (L = 1 ⇒ M = 5):
# (axis2,−1), (axis1,−1), diagonal, (axis1,+1), (axis2,+1).
lap = StarStencil{1}(fill(SVector(-1.0, -1.0, 4.0, -1.0, -1.0), n1, n2))
```

## The general stencil and narrowing

[`Stencil`](@ref) is the lingua franca: an explicit reverse-lex list of
`StaticShift` offsets plus a matching `SVector` coefficient. It is *not*
assembled directly — [`as_linear`](@ref) / [`as_star`](@ref) narrow it to an
assemblable type by matching the offset pattern and reusing the coefficient
verbatim:

```julia
st = Stencil(ColumnAccess, (-2ê₁, -ê₁, ô), fill(SVector(1.0, -4.0, 3.0), n))
ln = as_linear(st)        # LinearStencil{1, -2, 3, …}
ln.term === st.term       # true — verbatim copy
```

A `Stencil` whose offsets form the canonical star pattern narrows with
[`as_star`](@ref) instead; mismatched patterns raise an `ArgumentError`.
