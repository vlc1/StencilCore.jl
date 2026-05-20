# Concrete stencil types with a *relaxed* coefficient field: the
# coefficient may be a concrete array (assemblable) or a symbolic term
# (`materialize` first). Both are `ArrayOrTermLike{E}` with element type
# `E<:SVector{L}` (array-of-structs: one SVector of all per-offset
# coefficients per column). The CSC assembler (CartesianOperators) adds
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
  `AbstractArray{E}` (assemblable) or a symbolic `AbstractTerm{E}`.
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
        term::AbstractTerm{SVector{L, T}},
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
        "term must be an AbstractArray or AbstractTerm with eltype SVector{$L, T} " *
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
        "and supply term as an AbstractArray or AbstractTerm whose elements are " *
        "SVector{L} of coefficients in ascending-offset order (element[1] is for δ_min)."))
end

"""
    StarStencil{L, N, M, E<:SVector{M}, C<:NTuple{N, ArrayOrTermLike{E}}, S<:AccessStyle}
        <: AbstractStencil{S}

N-D variable-coefficient star-shaped stencil with symmetric reach `−L … +L`
along every mesh dimension. Per-axis offsets are diagonal indices; offset `δ`
along axis `d` lands at row coord `c_d − δ`, identity elsewhere. The diagonal
sums per-axis δ=0 contributions: `A[r, r] = Σ_d terms[d][c][L + 1]`.

`terms[d][c_idx...][k]` is the coefficient at column `c_idx` for axis `d`,
offset `δ = k − L − 1` (under `S = ColumnAccess`).

Type parameters:
- `L ≥ 1` per-axis reach; `M = 2L + 1` offsets per axis.
- `N` axis count = tuple length = grid rank (kept: fixed by construction
  even for a symbolic coefficient).
- `E<:SVector{M}`: per-axis coefficient element type; scalar `eltype(E)`.
- `C<:NTuple{N, ArrayOrTermLike{E}}`: one coefficient container per axis
  (array or term; per-axis concrete types may differ).
- `S<:AccessStyle`.
"""
struct StarStencil{L, N, M, E<:SVector{M}, C<:NTuple{N, ArrayOrTermLike{E}}, S<:AccessStyle} <: AbstractStencil{S}
    terms::C

    # All-array coefficients.
    function StarStencil{L}(
        ::Type{S},
        terms::NTuple{N, AbstractArray{SVector{M, T}, N}},
    ) where {L, S<:AccessStyle, N, M, T}
        L isa Int && L >= 1 || throw(ArgumentError(
            "stencil reach L must be a positive Int (got $L)"))
        M == 2L + 1 || throw(ArgumentError(
            "per-axis SVector length must be 2L+1=$(2L + 1) (got $M)"))
        new{L, N, M, SVector{M, T}, typeof(terms), S}(terms)
    end

    # All-term (symbolic) coefficients.
    function StarStencil{L}(
        ::Type{S},
        terms::NTuple{N, AbstractTerm{SVector{M, T}}},
    ) where {L, S<:AccessStyle, N, M, T}
        L isa Int && L >= 1 || throw(ArgumentError(
            "stencil reach L must be a positive Int (got $L)"))
        M == 2L + 1 || throw(ArgumentError(
            "per-axis SVector length must be 2L+1=$(2L + 1) (got $M)"))
        new{L, N, M, SVector{M, T}, typeof(terms), S}(terms)
    end
end

# Default outer constructor: bare 1-arg form forwards with ColumnAccess.
StarStencil{L}(terms) where {L} = StarStencil{L}(ColumnAccess, terms)

# Friendly outer constructor: reports specific errors for ill-typed tuples.
function StarStencil{L}(::Type{S}, terms::Tuple) where {L, S<:AccessStyle}
    L isa Int && L >= 1 || throw(ArgumentError(
        "stencil reach L must be a positive Int (got $L)"))
    M_expected = 2L + 1
    all(c -> c isa ArrayOrTermLike, terms) || throw(ArgumentError(
        "each per-axis term must be an AbstractArray or AbstractTerm of " *
        "SVector{$M_expected} (got $(map(typeof, terms)))"))
    Es = map(eltype, terms)
    all(E -> E <: SVector, Es) || throw(ArgumentError(
        "each per-axis term eltype must be SVector{$M_expected, T} (got eltypes $Es)"))
    all(E -> length(E) == M_expected, Es) || throw(ArgumentError(
        "each per-axis term eltype must be SVector{$M_expected, T} to match " *
        "2L+1=$M_expected (got SVector lengths $(map(length, Es)))"))
    Ts = map(eltype, Es)
    all(==(first(Ts)), Ts) || throw(ArgumentError(
        "all terms must share the same scalar eltype (got $Ts)"))
    throw(ArgumentError("StarStencil could not be constructed; terms = $terms"))
end

# Catch-all for non-Tuple terms.
function StarStencil{L}(::Type{S}, terms) where {L, S<:AccessStyle}
    throw(ArgumentError(
        "terms must be an NTuple{N, ArrayOrTermLike{SVector{M, T}}} " *
        "(got $(typeof(terms)))"))
end
