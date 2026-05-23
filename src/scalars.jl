# Scalar algebra: a hierarchy parallel to AbstractTerm but **without spatial
# extent**. An AbstractScalar{T} materializes to a single value of type `T`;
# an AbstractTerm{T} materializes to an array whose cells hold values of type
# `T`. The two are bridged by `Fill` (defined in StencilCalculus), which wraps
# an AbstractScalar inside an AbstractTerm so it broadcasts over the grid.
#
# Concrete subtypes mirror the term-side ones:
#   Symbolic ↔ Slot   (named, runtime-substituted leaf)
#   Const    ↔ Fill   (literal carrier — Const lives in scalar-land, Fill in
#                      term-land and may wrap a Const)
#   Scalar   ↔ Term   (interior tree node)
#   Null     ↔ Zero   (additive identity / structural zero)
#   Unity    ↔ One    (multiplicative identity / structural one)
#
# `T` is required to be concrete (`_assert_concrete`), matching the term side.

"""
    AbstractScalar{T}

Supertype for cell-level scalar expressions. Reaches a single value of
type `T` at `materialize` time (no axes). Concrete subtypes:
[`Symbolic`](@ref), [`Const`](@ref), [`Scalar`](@ref), [`Null`](@ref),
[`Unity`](@ref). Sibling of, **not** subtype of, [`AbstractTerm`](@ref).
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
    Const(value) / Const{T}(value)

Literal scalar leaf carrying its `value` in a runtime field. The element
type is `typeof(value)` (always concrete).
"""
struct Const{T} <: AbstractScalar{T}
    val::T
    Const{T}(val) where {T} = (_assert_concrete(:Const, T); new{T}(val))
end
Const(val) = Const{typeof(val)}(val)

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

"""
    Unity{T}()

Type-level multiplicative identity for [`AbstractScalar`](@ref): the
scalar-side analogue of [`AbstractTerm`](@ref) `One`. Materializes to
`one(T)`.
"""
struct Unity{T} <: AbstractScalar{T}
    Unity{T}() where {T} = (_assert_concrete(:Unity, T); new{T}())
end

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
# Number↔AbstractScalar boundary. Always wraps as `Const` (the literal leaf).
asscalar(s::AbstractScalar) = s
asscalar(x::Number)         = Const(x)
Base.convert(::Type{<:AbstractScalar}, x::Number) = Const(x)

# Shift-invariance: a scalar has no spatial position, so any shift is a no-op.
Base.getindex(s::AbstractScalar)                = s
Base.getindex(s::AbstractScalar, ::StaticShift) = s

# --- Operator overloads ------------------------------------------------------
# Every binary op among {AbstractScalar, Number} (with at least one
# AbstractScalar) lifts into a `Scalar` tree. Numeric literals canonicalise to
# `Const` first.

for op in (:+, :-, :*, :/, :\, :^, :min, :max)
    @eval Base.$op(a::AbstractScalar, b::AbstractScalar) = Scalar($op, (a, b))
    @eval Base.$op(a::AbstractScalar, b::Number)         = Scalar($op, (a, Const(b)))
    @eval Base.$op(a::Number,         b::AbstractScalar) = Scalar($op, (Const(a), b))
end
for op in (:-, :+, :exp, :sin, :cos, :tan, :log, :sqrt, :abs)
    @eval Base.$op(a::AbstractScalar) = Scalar($op, (a,))
end

# --- Constructor macros ------------------------------------------------------

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

"""
    @const name value

Bind `name` to `Const(value)`. `@const α 1` ≡ `α = Const(1)`. Defined via the
`var"@const"` function form because `const` is a reserved word (so a plain
`macro const` would not parse).
"""
function var"@const"(__source__::LineNumberNode, __module__::Module, name, value)
    name isa Symbol || throw(ArgumentError("@const expects a variable name, got `$(name)`"))
    :($(esc(name)) = $Const($(esc(value))))
end

# --- Display -----------------------------------------------------------------
# Scalars render without going through `simplify` (no rewriter at this layer
# yet). Leaves: Symbolic prints its symbol; Const prints its value; Null/Unity
# print `0`/`1` glyphs (type-agnostic, matching the term-side Zero/One).
# Scalar interior nodes render infix when the op is in `_INFIX`, else as a call.

const _SCALAR_INFIX = (:+, :-, :*, :/, :\, :^)
_scalar_callsym(f) = nameof(f)

Base.show(io::IO, s::AbstractScalar) = _scalar_show(io, s)

_scalar_show(io::IO, ::Symbolic{S}) where {S} = print(io, S)
_scalar_show(io::IO, c::Const)                = show(io, c.val)
_scalar_show(io::IO, ::Null)                  = print(io, '0')
_scalar_show(io::IO, ::Unity)                 = print(io, '1')

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
