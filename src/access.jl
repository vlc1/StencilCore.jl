# Access-style trait + abstract stencil supertype.
#
# `AccessStyle` discriminates how a stencil's coefficient arrays are
# anchored (column vs row). It lives as a *type parameter on the
# stencil* (the trailing slot, by convention); assemblers dispatch on
# it to enforce that the stencil's anchoring matches the target sparse
# format. The trait is never consulted at element-access time —
# coefficient reads are plain `getindex`.

"""
    AccessStyle

Holy trait reporting how a stencil's coefficient arrays are anchored:

- [`ColumnAccess`](@ref): `term[c_idx...][k]` is the value at
  **column** mesh position `c_idx`. Required for assembly into
  `SparseMatrixCSC`.
- [`RowAccess`](@ref): `term[r_idx...][k]` is the value at **row**
  mesh position `r_idx`. Required for assembly into a row-major
  format (CSR; not yet implemented).

The trait is a **type parameter on the stencil**, not a runtime field.
Assemblers dispatch on it; mismatching the access style and the target
sparse format is a dispatch-time error (`MethodError`).
"""
abstract type AccessStyle end

"""
    ColumnAccess <: AccessStyle

Coefficients anchored at the **column** mesh position — required for
compressed-sparse-column (`SparseMatrixCSC`) assembly. See [`AccessStyle`](@ref).
"""
struct ColumnAccess <: AccessStyle end

"""
    RowAccess <: AccessStyle

Coefficients anchored at the **row** mesh position — reserved for a future
compressed-sparse-row backend. See [`AccessStyle`](@ref).
"""
struct RowAccess    <: AccessStyle end

"""
    AbstractStencil{T}

Abstract supertype for every stencil — both the *diagonal* stencils that
live in pointwise-land ([`AbstractPointwise`](@ref) and its subtypes) and the
*neighborhood* stencils that carry an [`AccessStyle`](@ref) and a tuple of
offsets ([`NeighborhoodStencil`](@ref)). The single type parameter `T` is
the *linear-map space* — the element type of the per-cell coefficient,
mirroring how `Unity{T}` carries the multiplicative-identity space in
scalar-land.

For a `LinearStencil` / `StarStencil` whose coefficient stores `SVector{L, F}`
per cell, `T === F` (not `SVector{L, F}`). The match rule for applying a
stencil to an `AbstractPointwise{U}` is `T === _unity_space(U)`: scalar-on-
scalar for `U <: Number`, `SMatrix{N, N, F}`-on-`SVector{N, F}` for
vector-valued fields.

Provides a `Base.eltype` accessor — subtypes inherit it without redefining.
The [`AccessStyle`](@ref) trait is defined on [`NeighborhoodStencil`](@ref),
not on `AbstractStencil` itself, since diagonal stencils have no offsets.
"""
abstract type AbstractStencil{T} end

Base.eltype(::Type{<:AbstractStencil{T}}) where {T} = T
Base.eltype(st::AbstractStencil) = eltype(typeof(st))

"""
    NeighborhoodStencil{T, S<:AccessStyle} <: AbstractStencil{T}

Abstract supertype for stencils that carry off-diagonal offsets and therefore
require an [`AccessStyle`](@ref) anchor. Concrete subtypes — `Stencil`,
`LinearStencil`, `StarStencil` — declare `<: NeighborhoodStencil{T, S}`,
carrying the coefficient element type `T` as their second-to-last type
parameter and the access style `S` as their **last** type parameter.

Provides the [`AccessStyle`](@ref) trait accessor; subtypes inherit it
without redefining.
"""
abstract type NeighborhoodStencil{T, S<:AccessStyle} <: AbstractStencil{T} end

AccessStyle(st::NeighborhoodStencil) = AccessStyle(typeof(st))
AccessStyle(::Type{<:NeighborhoodStencil{T, S}}) where {T, S} = S()
