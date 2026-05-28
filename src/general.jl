# General offset-list stencil + narrowing to LinearStencil / StarStencil.
#
# `Stencil` is the lingua franca produced by symbolic differentiation: an
# explicit reverse-lex-ordered list of `StaticShift` offsets plus a *matching
# tuple of per-offset coefficients* (structure-of-arrays — the `terms` layout
# mirrors `shifts`). It is not assembled directly; `as_linear` / `as_star`
# narrow it to an optimized, assemblable `LinearStencil` / `StarStencil`, which
# is the point where the representation switches to array-of-structs (a single
# `SVector{M}`-valued coefficient) via `_interlace`.

# Single-axis decomposition of a StaticShift (the zero shift → axis 0, offset 0).
_naxes(::StaticShift{P}) where {P} = length(P.parameters)
_shift_axis(::StaticShift{Tuple{}}) = 0
_shift_axis(::StaticShift{Tuple{StaticPair{D, O}}}) where {D, O} = D
_shift_off(::StaticShift{Tuple{}}) = 0
_shift_off(::StaticShift{Tuple{StaticPair{D, O}}}) where {D, O} = O

"""
    Stencil{M, C<:NTuple{M,StaticShift}, A<:NTuple{M,ArrayOrTermLike}, T, S<:AccessStyle}
        <: AbstractStencil{S, T}

General offset-list stencil in **structure-of-arrays** form: `M` offsets
`shifts` (reverse-lexicographically ordered) and a parallel `M`-tuple `terms`
of per-offset coefficients (`terms[k]` is the coefficient at offset
`shifts[k]`). Each coefficient is an `ArrayOrTermLike` — a concrete array or a
symbolic term — with a *scalar* element type, and all coefficients must share
the same `eltype === T`.

The form produced by symbolic differentiation; **not assembled directly** —
narrowed to a `LinearStencil` / `StarStencil` via [`as_linear`](@ref) /
[`as_star`](@ref), which is where the layout switches to array-of-structs (one
`SVector{M}`-valued coefficient) by `_interlace`ing `terms`.
"""
struct Stencil{M, C<:NTuple{M, StaticShift}, A<:NTuple{M, ArrayOrTermLike}, T, S<:AccessStyle} <: NeighborhoodStencil{T, S}
    shifts::C
    terms::A

    # Access style via a positional Type tag (S is the trailing type param;
    # T is the common coefficient eltype, second-to-last). This *inferring*
    # form derives T from the first non-wildcard coefficient.
    function Stencil(
        ::Type{S},
        shifts::NTuple{M, StaticShift},
        terms::NTuple{M, ArrayOrTermLike},
    ) where {S<:AccessStyle, M}
        M >= 1 || throw(ArgumentError("Stencil needs at least one offset"))
        # Wildcards (Fill-wrapped Null/Unity, IdentityStencil, and any
        # Pointwise/Shifted reaching only wildcard leaves) materialize to
        # `zero(T)`/`one(T)` of any surrounding T via promotion (the
        # Bool-shape discipline); they do not fix T. Derive T from the
        # non-wildcard coefficients.
        ix = findfirst(!_is_eltype_wildcard, terms)
        ix === nothing && throw(ArgumentError(
            "Stencil cannot derive a coefficient eltype: every coefficient " *
            "is a structural wildcard. Either include at least one " *
            "non-wildcard coefficient, or use `Stencil{T}(S, shifts, terms)` " *
            "to pin T explicitly."))
        T = eltype(terms[ix])
        for (k, t) in enumerate(terms)
            _is_eltype_wildcard(t) && continue
            eltype(t) === T || throw(ArgumentError(
                "Stencil coefficients must share eltype; derived T = $(T) " *
                "(from terms[$(ix)]) but terms[$(k)] has eltype $(eltype(t))"))
        end
        new{M, typeof(shifts), typeof(terms), T, S}(shifts, terms)
    end

    # Explicit-T form: caller supplies the coefficient eltype as a leading
    # `Type{T}` argument. Useful when every coefficient is a wildcard
    # (e.g. `differentiate(f, f)` whose only coefficient is `IdentityStencil`).
    # Wildcards bypass the uniformity check; non-wildcard coefficients must
    # agree with the supplied T.
    #
    # The Stencil struct's first type parameter is `M` (offset count), so the
    # `{T}` curly-brace syntax binds `M`, not `T`. We therefore pass T as a
    # positional `::Type{T}` argument; the `S<:AccessStyle` constraint on the
    # inferring form keeps the two dispatch paths disjoint.
    function Stencil(
        ::Type{T},
        ::Type{S},
        shifts::NTuple{M, StaticShift},
        terms::NTuple{M, ArrayOrTermLike},
    ) where {T, S<:AccessStyle, M}
        M >= 1 || throw(ArgumentError("Stencil needs at least one offset"))
        for (k, t) in enumerate(terms)
            _is_eltype_wildcard(t) && continue
            eltype(t) === T || throw(ArgumentError(
                "Stencil(T, …) coefficients must have eltype T = $(T); " *
                "terms[$(k)] has eltype $(eltype(t))"))
        end
        new{M, typeof(shifts), typeof(terms), T, S}(shifts, terms)
    end
end

# Trait: a coefficient that materializes to `zero(T)`/`one(T)` of any
# surrounding T via promotion, and therefore does not pin a Stencil's
# coefficient eltype. Default `false`; StencilCalculus overrides for
# `Fill{<:Null}` (a.k.a. `Zero`) and `Fill{<:Unity}` to carry the existing
# Bool-shape discipline through to the Stencil eltype-uniformity check.
_is_eltype_wildcard(_) = false

# Default outer constructor: ColumnAccess.
Stencil(shifts, terms) = Stencil(ColumnAccess, shifts, terms)

# Friendly error when shifts and terms disagree in length, or are not tuples.
function Stencil(::Type{S}, shifts, terms) where {S<:AccessStyle}
    throw(ArgumentError(
        "Stencil needs shifts::NTuple{M, StaticShift} and terms::NTuple{M, " *
        "ArrayOrTermLike} of equal length M (got $(typeof(shifts)) and " *
        "$(typeof(terms)))"))
end

"""
    _interlace(terms::NTuple{M, ArrayOrTermLike}) -> ArrayOrTermLike{<:SVector{M}}

Combine `M` per-offset (structure-of-arrays) coefficients into the single
array-of-structs `SVector{M}`-valued coefficient that `LinearStencil` /
`StarStencil` store — the representation switch performed by narrowing.

The concrete-array method (stack element-wise into an array of `SVector{M}`)
lives here; the symbolic-term method (`Term(SVector, terms)`) is added by the
StencilCalculus package.
"""
function _interlace end

function _interlace(terms::NTuple{M, AbstractArray}) where {M}
    ax = axes(first(terms))
    all(t -> axes(t) == ax, terms) || throw(ArgumentError(
        "coefficient arrays must share axes to interlace into an SVector coefficient"))
    map(SVector, terms...)
end

"""
    as_linear(st::Stencil{…,S}) -> LinearStencil{D, …, S}

Narrow a `Stencil` whose offsets are single-axis (same axis `D`) and contiguous
to the equivalent `LinearStencil{D}`, interlacing `terms` into the single
`SVector{D}`-valued coefficient. Throws if the offsets are multi-axis, span
several axes, or are not contiguous-ascending.
"""
function as_linear(st::Stencil{M, C, A, T, S}) where {M, C, A, T, S}
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
    LinearStencil{D}(S, SUnitRange(offs[1], offs[M]), _interlace(st.terms))
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
equivalent `StarStencil{L}`, interlacing `terms` into the single `SVector{M}`-
valued coefficient. Throws otherwise.
"""
function as_star(st::Stencil{M, C, A, T, S}) where {M, C, A, T, S}
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
    StarStencil{L}(S, _interlace(st.terms))
end
