# Type-level offsets: `StaticPair{D,O}` and `StaticShift`.
#
# Shifts enter a grid expression only through the DSL operator functors
# (`FwdDiff{D}`, `BwdDiff{D}`, …) whose dimension `D` is a type
# parameter, so the offset is compile-time known — on the same footing
# as `LinearStencil`'s `O`/`L`. Encoding the offset in the type system
# lets composition, normalization, and the `+`/`-`/`*` algebra happen at
# the type level, with the normal-form invariants enforced by the
# constructor.

"""
    StaticPair{D, O}

A single offset of magnitude `O` along mesh dimension `D` (both
compile-time `Int`s). Singleton. Alias: [`SPair`](@ref).
"""
struct StaticPair{D, O} end
const SPair = StaticPair

"""
    dim(p), offset(p)

Dimension `D` and offset `O` of a [`StaticPair`](@ref) (works on the
instance or the type).
"""
dim(::StaticPair{D, O}) where {D, O}        = D
dim(::Type{StaticPair{D, O}}) where {D, O}  = D
offset(::StaticPair{D, O}) where {D, O}       = O
offset(::Type{StaticPair{D, O}}) where {D, O} = O

# Insert a pair into an already-normalized (sorted-by-D, unique-D,
# nonzero-O) tuple, preserving those invariants.
_insert(p::StaticPair, ::Tuple{}) = offset(p) == 0 ? () : (p,)
function _insert(
    p::StaticPair{D, O},
    t::Tuple{StaticPair{D2, O2}, Vararg{StaticPair}},
) where {D, O, D2, O2}
    head = t[1]
    rest = Base.tail(t)
    if D == D2
        s = O + O2
        return s == 0 ? rest : (StaticPair{D, s}(), rest...)
    elseif D < D2
        return O == 0 ? t : (p, t...)
    else
        return (head, _insert(p, rest)...)
    end
end

# Normalize an arbitrary tuple of pairs: combine same-D (summing
# offsets), drop zero offsets, sort ascending by D.
_normalize(::Tuple{}) = ()
_normalize(t::Tuple{StaticPair, Vararg{StaticPair}}) =
    _insert(t[1], _normalize(Base.tail(t)))

"""
    StaticShift{P<:Tuple{Vararg{StaticPair}}}

A lattice offset: a normalized collection of [`StaticPair`](@ref)s.
Invariants (enforced by the constructor): pairs are sorted ascending by
dimension, no two pairs share a dimension (same-dimension pairs are
summed), and no pair has zero offset (dropped). The empty shift
`StaticShift{Tuple{}}` is the identity.

Construct from pairs, or via the basis symbols `ê₁ … ê₉` and the
`+`/`-`/`*Int` algebra:

```julia
3ê₁ + ê₂            # StaticShift{Tuple{StaticPair{1,3}, StaticPair{2,1}}}
```

Alias: [`SShift`](@ref).
"""
struct StaticShift{P<:Tuple{Vararg{StaticPair}}}
    pairs::P
    StaticShift(pairs::Tuple{Vararg{StaticPair}}) =
        (np = _normalize(pairs); new{typeof(np)}(np))
end
const SShift = StaticShift

StaticShift(pairs::StaticPair...) = StaticShift(pairs)

# --- Algebra (type-level via the normalizing constructor) ---

Base.:+(a::StaticShift, b::StaticShift) = StaticShift((a.pairs..., b.pairs...))

Base.:-(::StaticPair{D, O}) where {D, O} = StaticPair{D, -O}()
Base.:-(a::StaticShift)                  = StaticShift(map(-, a.pairs))
Base.:-(a::StaticShift, b::StaticShift)  = a + (-b)

Base.:*(k::Integer, ::StaticPair{D, O}) where {D, O} = StaticPair{D, k * O}()
Base.:*(k::Integer, a::StaticShift) = StaticShift(map(p -> k * p, a.pairs))
Base.:*(a::StaticShift, k::Integer) = k * a

Base.iszero(::StaticShift{Tuple{}}) = true
Base.iszero(::StaticShift)          = false

# --- Display: render as a sum of `O·êD` basis terms ---

const _SUBSCRIPTS = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')
_subscript(n::Integer) = join(_SUBSCRIPTS[d - '0' + 1] for d in string(n))
_basis_symbol(D::Integer) = string('ê', _subscript(D))

function Base.show(io::IO, s::StaticShift)
    if isempty(s.pairs)
        print(io, "𝟎")
        return
    end
    first = true
    for p in s.pairs
        O = offset(p)
        if first
            O < 0 && print(io, "-")
            first = false
        else
            print(io, O < 0 ? " - " : " + ")
        end
        a = abs(O)
        a == 1 || print(io, a)
        print(io, _basis_symbol(dim(p)))
    end
end

# Zero shift (identity) and basis shifts ê₁ … ê₉ (unit offset per dim).
const ô  = StaticShift()
const ê₁ = StaticShift((StaticPair{1, 1}(),))
const ê₂ = StaticShift((StaticPair{2, 1}(),))
const ê₃ = StaticShift((StaticPair{3, 1}(),))
const ê₄ = StaticShift((StaticPair{4, 1}(),))
const ê₅ = StaticShift((StaticPair{5, 1}(),))
const ê₆ = StaticShift((StaticPair{6, 1}(),))
const ê₇ = StaticShift((StaticPair{7, 1}(),))
const ê₈ = StaticShift((StaticPair{8, 1}(),))
const ê₉ = StaticShift((StaticPair{9, 1}(),))
