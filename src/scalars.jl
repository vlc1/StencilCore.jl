# Scalar algebra: a hierarchy parallel to AbstractTerm but **without spatial
# extent**. An AbstractScalar{T} materializes to a single value of type `T`;
# an AbstractTerm{T} materializes to an array whose cells hold values of type
# `T`. The two are bridged by `Fill` (defined in StencilCalculus), which wraps
# an AbstractScalar inside an AbstractTerm so it broadcasts over the grid.
#
# Concrete subtypes mirror the term-side ones:
#   Symbolic ↔ Slot   (named, runtime-substituted leaf)
#   Scaling  ↔ Fill   (literal carrier — Scaling{V,T} represents `val * one(T)`,
#                      collapsing the former Const/Unity duet into one type)
#   Scalar   ↔ Term   (interior tree node)
#   Null     ↔ Zero   (additive identity / structural zero)
#
# `T` is required to be concrete (`_assert_concrete`), matching the term side.

"""
    AbstractScalar{T}

Supertype for cell-level scalar expressions. Reaches a single value of
type `T` at `materialize` time (no axes). Concrete subtypes:
[`Symbolic`](@ref), [`Scaling`](@ref), [`Scalar`](@ref), [`Null`](@ref).
Sibling of, **not** subtype of, [`AbstractTerm`](@ref).
"""
abstract type AbstractScalar{T} end

Base.eltype(::Type{<:AbstractScalar{T}}) where {T} = T
Base.eltype(s::AbstractScalar) = eltype(typeof(s))

"""
    Symbolic{S, T}()

Named, runtime-substituted scalar parameter `S` (a `Symbol`) of concrete type
`T` (default `Float64`). Materializes to the value supplied at the keyword `S`
of the `pairs` NamedTuple — like [`Slot`](@ref) but without per-cell indexing.
"""
struct Symbolic{S, T} <: AbstractScalar{T}
    Symbolic{S, T}() where {S, T} = (_assert_concrete(:Symbolic, T); new{S, T}())
end
Symbolic{S}() where {S} = Symbolic{S, Float64}()

"""
    Scaling{V, T}(val) / Scaling(val) / Scaling(T::Type)

Literal scalar leaf representing the value `val * one(T)`. `V` is the type of
the stored `val` and must satisfy `V <: eltype(T)` (for type stability at
materialize time). `Λ` is an alias.

Constructors:
- `Scaling{T}(val)`            — explicit element type `T`.
- `Scaling(val)`               — `T = typeof(val)`.
- `Scaling(T::Type = Float64)` — `val = one(T)` (the multiplicative identity).
"""
struct Scaling{V, T} <: AbstractScalar{T}
    val::V

    function Scaling{T}(val::V) where {V, T}
        _assert_concrete(:Scaling, T)
        V <: eltype(T) || throw(ArgumentError(
            "Scaling: V=$V is not a subtype of eltype(T)=$(eltype(T))"))
        new{V, T}(val)
    end
end

"""
    Λ

Alias for [`Scaling`](@ref).
"""
const Λ = Scaling

Scaling(T::Type = Float64) = Scaling{T}(one(T))
Scaling(val::V) where {V}  = Scaling{V}(val)

"""
    Null{T}()

Type-level additive identity / structural zero for [`AbstractScalar`](@ref):
the scalar-side analogue of [`AbstractTerm`](@ref) `Zero`. Materializes to
`zero(T)`; lets the scalar `simplify` and `differentiate` rules collapse by
dispatch.
"""
struct Null{T} <: AbstractScalar{T}
    Null{T}() where {T} = (_assert_concrete(:Null, T); new{T}())
end
Null(T::Type)       = Null{T}()
Null(::T) where {T} = Null{T}()

"""
    Scalar(fn, args::Tuple{Vararg{AbstractScalar}})

Interior node of a scalar-tree: applies `fn` to scalar `args` component-wise.
The element type `T = Base.promote_op(fn, eltype.(args)...)` is computed
**at construction**; a `Union{}` result throws (the node is unconstructable).
Term-side analogue: [`AbstractTerm`](@ref) `Term`.
"""
struct Scalar{F, A<:Tuple{Vararg{AbstractScalar}}, T} <: AbstractScalar{T}
    fn::F
    args::A
    Scalar{F, A, T}(fn::F, args::A) where {F, A<:Tuple{Vararg{AbstractScalar}}, T} =
        new{F, A, T}(fn, args)
end

function Scalar(fn::F, args::A) where {F, A<:Tuple{Vararg{AbstractScalar}}}
    T = Base.promote_op(fn, map(eltype, args)...)
    T === Union{} && throw(ArgumentError(
        "unconstructable Scalar: $(fn) over eltypes $(map(eltype, args)) has " *
        "no result type (Base.promote_op returned Union{})"))
    Scalar{F, A, T}(fn, args)
end

# Promote a numeric literal to a scalar leaf; used by operator overloads on the
# Number↔AbstractScalar boundary. Always wraps as `Scaling` (the literal leaf).
asscalar(s::AbstractScalar) = s
asscalar(x::Number)         = Scaling(x)
Base.convert(::Type{<:AbstractScalar}, x::Number) = Scaling(x)

# --- Operator overloads ------------------------------------------------------
# Every binary op among {AbstractScalar, Number} (with at least one
# AbstractScalar) lifts into a `Scalar` tree. Numeric literals canonicalise to
# `Scaling` first.

for op in (:+, :-, :*, :/, :\, :^, :min, :max)
    @eval Base.$op(a::AbstractScalar, b::AbstractScalar) = Scalar($op, (a, b))
    @eval Base.$op(a::AbstractScalar, b::Number)         = Scalar($op, (a, Scaling(b)))
    @eval Base.$op(a::Number,         b::AbstractScalar) = Scalar($op, (Scaling(a), b))
end
for op in (:-, :+, :exp, :sin, :cos, :tan, :log, :sqrt, :abs)
    @eval Base.$op(a::AbstractScalar) = Scalar($op, (a,))
end

# --- Constructor macro -------------------------------------------------------

"""
    @symbolic name [T = Float64]

Bind `name` to `Symbolic{:name, T}()`. `@symbolic τ` ≡
`τ = Symbolic{:τ, Float64}()`; `@symbolic τ Float32` ≡
`τ = Symbolic{:τ, Float32}()`. Term-side analogue: [`@slot`](@ref).
"""
macro symbolic(name, T = :Float64)
    name isa Symbol || throw(ArgumentError("@symbolic expects a variable name, got `$(name)`"))
    :($(esc(name)) = $Symbolic{$(QuoteNode(name)), $(esc(T))}())
end

# --- Display -----------------------------------------------------------------
# Scalars render without going through `simplify` (no rewriter at this layer
# yet). Leaves: Symbolic prints its symbol; Scaling prints its stored `val`;
# Null prints the `0` glyph (type-agnostic, matching the term-side Zero).
# Scalar interior nodes render infix when the op is in `_INFIX`, else as a
# call.

const _SCALAR_INFIX = (:+, :-, :*, :/, :\, :^)
_scalar_callsym(f) = nameof(f)

Base.show(io::IO, s::AbstractScalar) = _scalar_show(io, s)

_scalar_show(io::IO, ::Symbolic{S}) where {S} = print(io, S)
_scalar_show(io::IO, s::Scaling)              = show(io, s.val)
_scalar_show(io::IO, ::Null)                  = print(io, '0')

function _scalar_show(io::IO, s::Scalar)
    op, args = _scalar_callsym(s.fn), s.args
    if length(args) == 2 && op in _SCALAR_INFIX
        print(io, '(')
        _scalar_show(io, args[1])
        print(io, ' ', op, ' ')
        _scalar_show(io, args[2])
        print(io, ')')
    elseif length(args) == 1 && op === :-
        print(io, '-')
        _scalar_show(io, args[1])
    else
        print(io, op, '(')
        for (i, a) in enumerate(args)
            i == 1 || print(io, ", ")
            _scalar_show(io, a)
        end
        print(io, ')')
    end
end
