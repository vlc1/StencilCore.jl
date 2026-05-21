# General offset-list stencil + narrowing to LinearStencil / StarStencil.
#
# `Stencil` is the lingua franca produced by symbolic differentiation: an
# explicit reverse-lex-ordered list of `StaticShift` offsets plus a matching
# `SVector{M}`-valued coefficient. It is not assembled directly; `as_linear` /
# `as_star` narrow it to an optimized, assemblable stencil by a shift-pattern
# match and a verbatim `term` copy (the layouts coincide by construction).

# Single-axis decomposition of a StaticShift (the zero shift → axis 0, offset 0).
_naxes(::StaticShift{P}) where {P} = length(P.parameters)
_shift_axis(::StaticShift{Tuple{}}) = 0
_shift_axis(::StaticShift{Tuple{StaticPair{D, O}}}) where {D, O} = D
_shift_off(::StaticShift{Tuple{}}) = 0
_shift_off(::StaticShift{Tuple{StaticPair{D, O}}}) where {D, O} = O

"""
    Stencil{M, C<:NTuple{M,StaticShift}, E<:SVector{M}, A<:ArrayOrTermLike{E}, S<:AccessStyle}
        <: AbstractStencil{S}

General offset-list stencil: `M` offsets `shifts` (reverse-lexicographically
ordered) and a matching `SVector{M}`-valued coefficient `term`. The form
produced by symbolic differentiation; **not assembled directly** — narrowed to
a `LinearStencil` / `StarStencil` via [`as_linear`](@ref) / [`as_star`](@ref),
which reuse `term` verbatim.
"""
struct Stencil{M, C<:NTuple{M, StaticShift}, E<:SVector{M}, A<:ArrayOrTermLike{E}, S<:AccessStyle} <: AbstractStencil{S}
    shifts::C
    term::A

    # Access style via a positional Type tag (S is the trailing type param).
    function Stencil(
        ::Type{S},
        shifts::NTuple{M, StaticShift},
        term::AbstractArray{SVector{M, T}, N},
    ) where {S<:AccessStyle, M, T, N}
        new{M, typeof(shifts), SVector{M, T}, typeof(term), S}(shifts, term)
    end

    function Stencil(
        ::Type{S},
        shifts::NTuple{M, StaticShift},
        term::AbstractTerm{SVector{M, T}},
    ) where {S<:AccessStyle, M, T}
        new{M, typeof(shifts), SVector{M, T}, typeof(term), S}(shifts, term)
    end
end

# Default outer constructor: ColumnAccess.
Stencil(shifts, term) = Stencil(ColumnAccess, shifts, term)

# Friendly error when shift count and SVector length disagree (or bad types).
function Stencil(::Type{S}, shifts, term) where {S<:AccessStyle}
    throw(ArgumentError(
        "Stencil needs shifts::NTuple{M, StaticShift} and term an AbstractArray " *
        "or AbstractTerm whose elements are SVector{M} with matching M " *
        "(got $(typeof(shifts)) and $(typeof(term)))"))
end

"""
    as_linear(st::Stencil{…,S}) -> LinearStencil{D, …, S}

Narrow a `Stencil` whose offsets are single-axis (same axis `D`) and contiguous
to the equivalent `LinearStencil{D}`, reusing `term` verbatim. Throws if the
offsets are multi-axis, span several axes, or are not contiguous-ascending.
"""
function as_linear(st::Stencil{M, C, E, A, S}) where {M, C, E, A, S}
    shifts = st.shifts
    all(s -> _naxes(s) <= 1, shifts) || throw(ArgumentError(
        "Stencil offsets are not single-axis; cannot narrow to LinearStencil"))
    D = 0
    for s in shifts
        a = _shift_axis(s)
        a == 0 && continue
        if D == 0
            D = a
        elseif D != a
            throw(ArgumentError(
                "Stencil offsets span axes $D and $a; cannot narrow to LinearStencil"))
        end
    end
    D == 0 && throw(ArgumentError(
        "Stencil carries only the zero offset; axis is ambiguous for LinearStencil"))
    offs = map(_shift_off, shifts)
    for i in 2:M
        offs[i] == offs[i - 1] + 1 || throw(ArgumentError(
            "Stencil offsets are not contiguous-ascending; cannot narrow to LinearStencil"))
    end
    LinearStencil{D}(S, SUnitRange(offs[1], offs[M]), st.term)
end

# Expected (axis, offset) at slot k of the canonical reverse-lex star of
# reach L and rank N (M = 2NL+1). Axis 0 marks the diagonal.
function _expected_star_slot(k::Int, L::Int, N::Int)
    NL = N * L
    if k <= NL
        b = (k - 1) ÷ L
        return (N - b, -L + (k - 1) % L)
    elseif k == NL + 1
        return (0, 0)
    else
        j = k - (NL + 1)
        b = (j - 1) ÷ L
        return (b + 1, (j - 1) % L + 1)
    end
end

"""
    as_star(st::Stencil{…,S}) -> StarStencil{L, …, S}

Narrow a `Stencil` whose offsets form the canonical reverse-lex star pattern
(every axis `1..N` with symmetric reach `−L..L`, plus the diagonal) to the
equivalent `StarStencil{L}`, reusing `term` verbatim. Throws otherwise.
"""
function as_star(st::Stencil{M, C, E, A, S}) where {M, C, E, A, S}
    shifts = st.shifts
    all(s -> _naxes(s) <= 1, shifts) || throw(ArgumentError(
        "Stencil offsets are not single-axis; cannot narrow to StarStencil"))
    N = maximum(_shift_axis, shifts)
    L = maximum(s -> abs(_shift_off(s)), shifts)
    (N >= 1 && L >= 1 && M == 2N * L + 1) || throw(ArgumentError(
        "Stencil offsets do not form a star: M=$M, derived N=$N, L=$L (need M = 2NL+1)"))
    for k in 1:M
        ax, off = _expected_star_slot(k, L, N)
        (_shift_axis(shifts[k]) == ax && _shift_off(shifts[k]) == off) || throw(ArgumentError(
            "Stencil offsets do not match the canonical star order at slot $k"))
    end
    StarStencil{L}(S, st.term)
end
