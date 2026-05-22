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
