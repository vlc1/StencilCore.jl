# Concrete stencil types with a *relaxed* coefficient field: the
# coefficient may be a concrete array (assemblable) or a symbolic term
# (`materialize` first). Both are `ArrayOrTermLike{E}` with element type
# `E<:SVector{L}` (array-of-structs: one SVector of all per-offset
# coefficients per column). The CSC assembler (StencilAssembly) adds
# methods constrained to concrete-array coefficients.

"""
    LinearStencil{D, O, L, E<:SVector{L}, A<:ArrayOrTermLike{E}, S<:AccessStyle}
        <: AbstractStencil{S}

Variable-coefficient stencil with **contiguous** offsets, aligned with mesh
dimension `D`. Offsets are **diagonal indices**: for a column `j` and a row
`i` the diagonal number is `k = j − i`. For column `c` at mesh position `p_c`
and offset `δ`, the matrix entry lands on row `p_c − δ` with coefficient
`term[p_c][δ − O + 1]` (column-anchored under `S = ColumnAccess`); each
element `term[p_c]` is the `SVector{L}` of all `L` per-offset coefficients on
that column in **ascending offset order** (`term[p_c][1]` ↦ offset `O`).

Type parameters:
- `D`: mesh dim along which the stencil acts (`D ≥ 1`; for a concrete
  coefficient also `D ≤ ndims(term)`).
- `O = δ_min`, `L = δ_max − δ_min + 1` — static via `SUnitRange`.
- `E<:SVector{L}`: coefficient element type (the per-column `SVector`);
  the scalar is `eltype(E)`.
- `A<:ArrayOrTermLike{E}`: the coefficient container — a concrete
  `AbstractArray{E}` (assemblable) or a symbolic `AbstractPointwise{E}`.
  **`N` (grid rank) is not a stencil parameter**: it is `ndims(A)` for a
  concrete coefficient, and unknown (resolved at `materialize`) for a
  symbolic one.
- `S<:AccessStyle`: coefficient anchoring (`ColumnAccess` for CSC).

The inner constructors bind `E = SVector{L, T}`; the default outer
constructor defaults `S` to `ColumnAccess`. A catch-all reports
`ArgumentError`s for non-`SUnitRange` offsets and ill-typed coefficients.
"""
struct LinearStencil{D, O, L, E<:SVector{L}, A<:ArrayOrTermLike{E}, S<:AccessStyle} <: AbstractStencil{S}
    offsets::SUnitRange{O, L}
    term::A

    # Concrete-array coefficient: ndims N available ⇒ enforce D ≤ N.
    function LinearStencil{D}(
        ::Type{S},
        offsets::SUnitRange{O, L},
        term::AbstractArray{SVector{L, T}, N},
    ) where {D, S<:AccessStyle, O, L, T, N}
        D isa Int && D >= 1 || throw(ArgumentError(
            "stencil dimension D must be a positive Int (got $D)"))
        D <= N || throw(ArgumentError(
            "stencil dimension D=$D exceeds coef-array dimension N=$N"))
        new{D, O, L, SVector{L, T}, typeof(term), S}(offsets, term)
    end

    # Symbolic-term coefficient: grid rank unknown ⇒ no D ≤ N check.
    function LinearStencil{D}(
        ::Type{S},
        offsets::SUnitRange{O, L},
        term::AbstractPointwise{SVector{L, T}},
    ) where {D, S<:AccessStyle, O, L, T}
        D isa Int && D >= 1 || throw(ArgumentError(
            "stencil dimension D must be a positive Int (got $D)"))
        new{D, O, L, SVector{L, T}, typeof(term), S}(offsets, term)
    end
end

# Default outer constructor: bare 2-arg form forwards with ColumnAccess.
LinearStencil{D}(offsets, term) where {D} = LinearStencil{D}(ColumnAccess, offsets, term)

# Friendly outer constructor: reports specific errors when neither inner
# method matched (coefficient not an Array/Term of SVector{L}).
function LinearStencil{D}(::Type{S}, offsets::SUnitRange{O, L}, term) where {D, S<:AccessStyle, O, L}
    term isa ArrayOrTermLike || throw(ArgumentError(
        "term must be an AbstractArray or AbstractPointwise with eltype SVector{$L, T} " *
        "(got $(typeof(term)))"))
    E = eltype(term)
    E <: SVector || throw(ArgumentError(
        "term eltype must be SVector{$L, T} (got eltype $E); each element is " *
        "the column's coefficients in ascending-offset order"))
    length(E) == L || throw(ArgumentError(
        "term eltype must be SVector{$L, T} to match offsets length L=$L " *
        "(got SVector length $(length(E)))"))
    throw(ArgumentError("LinearStencil could not be constructed; term = $term"))
end

# Fallback for non-SUnitRange offsets.
function LinearStencil{D}(::Type{S}, offsets, term) where {D, S<:AccessStyle}
    throw(ArgumentError(
        "offsets must be a StaticArrays.SUnitRange (contiguous unit-stride). " *
        "Got $(typeof(offsets)). Construct one via SUnitRange(δ_min, δ_max), " *
        "and supply term as an AbstractArray or AbstractPointwise whose elements are " *
        "SVector{L} of coefficients in ascending-offset order (element[1] is for δ_min)."))
end

# Validate a star's (L, M) and return the derived grid rank N = (M-1)/(2L).
function _star_dims(L, M::Int)
    (L isa Int && L >= 1) || throw(ArgumentError(
        "stencil reach L must be a positive Int (got $L)"))
    (M - 1) % (2L) == 0 || throw(ArgumentError(
        "coefficient SVector length M=$M must equal 2NL+1 for an integer " *
        "grid rank N (L=$L); (M-1) must be divisible by 2L=$(2L)"))
    N = (M - 1) ÷ (2L)
    N >= 1 || throw(ArgumentError(
        "derived grid rank N=(M-1)/(2L)=$N must be ≥ 1 (M=$M, L=$L)"))
    return N
end

"""
    StarStencil{L, N, M, E<:SVector{M}, A<:ArrayOrTermLike{E}, S<:AccessStyle}
        <: AbstractStencil{S}

N-D variable-coefficient star-shaped stencil with symmetric reach `−L … +L`
along every mesh dimension, stored **interlaced**: a single coefficient
`term::A` whose element at each cell is the `SVector{M}` of the whole star,
`M = 2NL + 1`. The entries are in **reverse-lexicographic offset order**
(axis `N` most significant — the `CartesianIndex` order) with the diagonal
as the explicit middle slot `(M+1)/2`. Unlike a per-axis decomposition, the
diagonal is a single free coefficient (Helmholtz `k²`, parabolic `∂ₜ`),
**not** a sum of per-axis centers.

For `L = 2`, `N = 3` (`M = 13`) the slot ↦ offset map is:

    1:(d3,-2) 2:(d3,-1) 3:(d2,-2) 4:(d2,-1) 5:(d1,-2) 6:(d1,-1) 7:(diag)
    8:(d1,+1) 9:(d1,+2) 10:(d2,+1) 11:(d2,+2) 12:(d3,+1) 13:(d3,+2)

Type parameters:
- `L ≥ 1` per-axis reach; `M = 2NL + 1` whole-star offset count.
- `N` grid rank, kept explicit; checked to equal `(M-1)/(2L)` (and the
  coefficient array's `ndims`, when concrete).
- `E<:SVector{M}` coefficient element type; scalar `eltype(E)`.
- `A<:ArrayOrTermLike{E}`: the (single) coefficient container — concrete
  array (assemblable) or symbolic term.
- `S<:AccessStyle`.
"""
struct StarStencil{L, N, M, E<:SVector{M}, A<:ArrayOrTermLike{E}, S<:AccessStyle} <: AbstractStencil{S}
    term::A

    # Concrete-array coefficient: grid rank N = ndims; cross-check M = 2NL+1.
    function StarStencil{L}(
        ::Type{S},
        term::AbstractArray{SVector{M, T}, N},
    ) where {L, S<:AccessStyle, M, T, N}
        Nd = _star_dims(L, M)
        Nd == N || throw(ArgumentError(
            "coefficient array ndims=$N must equal grid rank (M-1)/(2L)=$Nd " *
            "(M=$M, L=$L)"))
        new{L, N, M, SVector{M, T}, typeof(term), S}(term)
    end

    # Symbolic-term coefficient: grid rank derived from (L, M).
    function StarStencil{L}(
        ::Type{S},
        term::AbstractPointwise{SVector{M, T}},
    ) where {L, S<:AccessStyle, M, T}
        N = _star_dims(L, M)
        new{L, N, M, SVector{M, T}, typeof(term), S}(term)
    end
end

# Default outer constructor: bare 1-arg form forwards with ColumnAccess.
StarStencil{L}(term) where {L} = StarStencil{L}(ColumnAccess, term)

# Friendly outer constructor: reports specific errors when neither inner
# method matched (coefficient not an Array/Term of SVector).
function StarStencil{L}(::Type{S}, term) where {L, S<:AccessStyle}
    term isa ArrayOrTermLike || throw(ArgumentError(
        "term must be an AbstractArray or AbstractPointwise with eltype SVector{M, T} " *
        "(got $(typeof(term)))"))
    E = eltype(term)
    E <: SVector || throw(ArgumentError(
        "term eltype must be SVector{M, T} (got eltype $E); each element is the " *
        "whole star's coefficients in reverse-lex order with the diagonal mid-slot"))
    _star_dims(L, length(E))  # throws on a bad (L, M); otherwise:
    throw(ArgumentError("StarStencil could not be constructed; term = $term"))
end
