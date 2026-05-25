# Scalar algebra: a hierarchy parallel to AbstractTerm but **without spatial
# extent**. An AbstractScalar{T} materializes to a single value of type `T`;
# an AbstractTerm{T} materializes to an array whose cells hold values of type
# `T`. The two are bridged by `Fill` (defined in StencilCalculus), which wraps
# an AbstractScalar inside an AbstractTerm so it broadcasts over the grid.
#
# Concrete leaves:
#   Symbolic{S, T}     — named, runtime-substituted variable
#   Constant{T}        — literal value carrier (any concrete `T`)
#   Scaling{T, V}      — `val * one(T)`; numerical coefficient × shape `T`,
#                        with `V <: Number` (stored val type)
#   Null{T}            — structural additive zero (dispatched-on)
#   Unity{T}           — structural multiplicative one (dispatched-on);
#                        requires `one(T)` to be defined ("square scalar")
# Interior node:
#   Scalar             — `fn` applied to scalar children
#
# `simplify` only inspects type-level information (Null/Unity by dispatch);
# field values (`.val`) are folded but never matched on. Numerical zeros /
# ones encoded in `.val` are not collapsed unless the value is encoded
# statically (e.g. via `Static`'s `StaticFloat64`).

"""
    AbstractScalar{T}

Supertype for cell-level scalar expressions. Reaches a single value of
type `T` at `materialize` time (no axes). Concrete subtypes:
[`Symbolic`](@ref), [`Constant`](@ref), [`Scaling`](@ref), [`Unity`](@ref),
[`Null`](@ref), [`Scalar`](@ref). Sibling of, **not** subtype of,
[`AbstractTerm`](@ref).
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
    Scaling{T, V <: Number}(val)

Literal scalar leaf representing the value `val * one(T)`. Two type parameters:

- `T`           — the materialized container type (the shape `one(·)` is taken
  of). Listed first so the one-curly inner ctor `Scaling{T}(val)` binds the
  curly argument to the shape tag. `T = T_raw` whenever `V <: eltype(T_raw)`;
  otherwise `T` widens via numeric promotion (e.g. `Scaling{Float32}(1.0)`
  yields `T = Float64`).
- `V <: Number` — type of the stored `val`.

`Λ` is an alias.

The isotropic boundary form `Scaling(val::Number)` is **not** provided —
numeric literals at the operator boundary canonicalise to [`Constant`](@ref)
instead. Every `Scaling` carries an explicit shape tag.

Constructors:
- `Scaling{T}(val)`                           — explicit shape tag `T`.
- `Scaling(T::Type)`                          — unit `Scaling` in the *linear-
  map space* of `T`: a `Number` stays `T`; an `SVector{N, F}` maps to
  `SMatrix{N, N, F}` (its Jacobian space). `val = one(eltype(·))`.
- `Scaling(T::Type, val)`                     — same `T → linear-map space`
  mapping, with explicit `val` stored as-is (`V = typeof(val)`).
- `Scaling(::Type{<:AbstractScalar{S}})`      — convenience: delegates to
  `Scaling(S)`.
- `Scaling(::Type{<:AbstractScalar{S}}, val)` — delegates to `Scaling(S, val)`.
"""
struct Scaling{T, V <: Number} <: AbstractScalar{T}
    val::V

    function Scaling{Traw}(val::V) where {Traw, V <: Number}
        E = promote_type(eltype(Traw), V)
        T = _similar(Traw, E)
        _assert_concrete(:Scaling, T)
        new{T, V}(val)
    end
end

# Two-curly form delegates through the canonicalizing one-curly inner ctor —
# so `Scaling{T, V}(val)` and `Scaling{T}(val)` produce identical results when
# T is in its canonical form.
Scaling{T, V}(val::V) where {T, V <: Number} = Scaling{T}(val)

# Shape-aware container synthesis: "what container looks like `S` but with
# scalar element type `E`?". Number shapes collapse to `E`; StaticArray shapes
# defer to StaticArrays.similar_type. Other `S` ⇒ MethodError.
_similar(::Type{S}, ::Type{E}) where {S <: Number, E}      = E
_similar(::Type{S}, ::Type{E}) where {S <: StaticArray, E} = similar_type(S, E)

# Linear-map space of a value-space type `T` — the type whose `one(·)` is the
# multiplicative identity for things in `T`'s algebra. Number stays itself
# (scalar 1 is the identity). SVector{N, F} maps to the canonical square
# SMatrix{N, N, F, N*N} (its Jacobian). Fallback returns `T` (square SMatrix
# is its own identity space; user-defined types are the user's problem —
# `one(T)` must work for materialization).
_unity_space(::Type{T}) where {T <: Number}      = T
_unity_space(::Type{SVector{N, F}}) where {N, F} = similar_type(SMatrix{N, N, F}, F)
_unity_space(::Type{T}) where {T}                = T

"""
    Λ

Alias for [`Scaling`](@ref).
"""
const Λ = Scaling

# Value-space ctors. `Scaling(T)` and `Scaling(T, val)` both route the `T`
# through `_unity_space` so e.g. `Scaling(SVector{N, F})` lands in the square
# SMatrix space (where `one(·)` is the Jacobian identity). Type-first
# convention matches the curly form `Scaling{T}(val)`.
function Scaling(::Type{T}) where {T}
    Tout = _unity_space(T)
    Scaling{Tout}(one(eltype(Tout)))
end
function Scaling(::Type{T}, val::Number) where {T}
    Scaling{_unity_space(T)}(val)
end

# Symbol-anchored ctors delegate to the value-space form.
Scaling(::Type{<:AbstractScalar{S}}) where {S}              = Scaling(S)
Scaling(::Type{<:AbstractScalar{S}}, val::Number) where {S} = Scaling(S, val)

"""
    Constant{T}(val) / Constant(val)

Literal scalar leaf carrying a value `val::T` as-is. Materializes to `val`
(no `one(·)` multiplication, unlike [`Scaling`](@ref)). `T` is any concrete
type — `Number`, `SArray`, etc. — making `Constant` the right carrier for
boundary literals such as `x + 1`, `v + SVector(1)` where no multiplicative
identity of `T` is needed (or even defined).
"""
struct Constant{T} <: AbstractScalar{T}
    val::T
    Constant{T}(val) where {T} = (_assert_concrete(:Constant, T); new{T}(convert(T, val)))
end
Constant(val::T) where {T} = Constant{T}(val)

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
    Unity{T}()

Type-level multiplicative identity / structural one for [`AbstractScalar`](@ref).
Materializes to `one(T)`; requires `T` to be a *square scalar* — a type with
`one(T)` defined (`Number`, square `SMatrix{N,N,F}`, …). Construction rejects
`T` lacking `one(T)` (e.g. `SVector`, non-square `SMatrix`).

Lets the scalar `simplify` rules collapse multiplicative identities by
dispatch — structurally, with no `.val` inspection — mirroring how `Null`
collapses additive identities.
"""
struct Unity{T} <: AbstractScalar{T}
    function Unity{T}() where {T}
        _assert_concrete(:Unity, T)
        applicable(one, T) || throw(ArgumentError(
            "Unity{T} requires `one(T)` to be defined (a square-scalar shape); got T=$T"))
        new{T}()
    end
end
Unity(::Type{T}) where {T} = Unity{_unity_space(T)}()
Unity(::T) where {T}       = Unity{_unity_space(T)}()

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

# Promote a non-AbstractScalar value to a scalar leaf at the operator boundary.
# Wraps as `Constant` — a literal carrier, no `one(·)` multiplication. (The
# isotropic `Scaling(val::Number)` form was removed: a Number literal at the
# boundary is a *value*, not a "1 · identity" coefficient.)
asscalar(s::AbstractScalar) = s
asscalar(x)                 = Constant(x)
Base.convert(::Type{<:AbstractScalar}, x) = Constant(x)

# --- Operator overloads ------------------------------------------------------
# Every binary op with at least one AbstractScalar lifts into a `Scalar` tree.
# Non-AbstractScalar operands canonicalise to `Constant`. The `b` (and `a`)
# slot is unbounded — bad pairings (e.g. eltypes with no `promote_op`) error
# downstream at `Scalar` construction.

for op in (:+, :-, :*, :/, :\, :^, :min, :max)
    @eval Base.$op(a::AbstractScalar, b::AbstractScalar) = Scalar($op, (a, b))
    @eval Base.$op(a::AbstractScalar, b)                 = Scalar($op, (a, Constant(b)))
    @eval Base.$op(a,                 b::AbstractScalar) = Scalar($op, (Constant(a), b))
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
# yet). Leaves: Symbolic prints its symbol; Constant and Scaling print their
# stored `val`; Null and Unity print the `0`/`1` glyphs (type-agnostic, like
# the term-side Zero). Scalar interior nodes render infix when the op is in
# `_INFIX`, else as a call.

const _SCALAR_INFIX = (:+, :-, :*, :/, :\, :^)
_scalar_callsym(f) = nameof(f)

Base.show(io::IO, s::AbstractScalar) = _scalar_show(io, s)

_scalar_show(io::IO, ::Symbolic{S}) where {S} = print(io, S)
_scalar_show(io::IO, s::Constant)             = show(io, s.val)
_scalar_show(io::IO, s::Scaling)              = show(io, s.val)
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
