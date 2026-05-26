# Symbolic differentiation of an AbstractScalar with respect to a Symbolic.
# Scalar-side analogue of StencilCalculus/src/differentiate.jl — but with no
# spatial dimension: no per-offset contributions, no Stencil. The result is a
# single AbstractScalar (`Null{T}()` when `s` does not depend on the variable;
# otherwise the chain-rule expression, simplified).

# --- @scalar_rule macro -------------------------------------------------------

"""
    @scalar_rule f(x) = expr
    @scalar_rule f(x, y) = (expr1, expr2)

Concise syntax for registering a symbolic derivative rule for a primitive `f`,
mirroring `ChainRules.@scalar_rule`.

The single-argument form defines `∂f/∂x`. The two-argument form defines both
partials `(∂f/∂x, ∂f/∂y)` — the tuple maps positionally to `Val{1}` and
`Val{2}` respectively.

Inside `expr`, the argument names `x`, `y` refer to the corresponding
`AbstractScalar` nodes (not numeric values). Use the scalar arithmetic
operators and `Constant(v)` for numeric literals.

**Examples:**

```julia
@scalar_rule sin(x)      = cos(x)
@scalar_rule exp(x)      = exp(x)
@scalar_rule +(x, y)     = (one(x), one(y))    # shorthand; actual table uses _unity
@scalar_rule log(x)      = inv(x)
```

Each call expands to one or more `derivative(::typeof(f), ::Val{i}, ...) = ...`
methods. Existing hand-written methods are equivalent and interoperate with this
macro.
"""
macro scalar_rule(call, rhs)
    # Parse `f(args...)` from the LHS call expression.
    call isa Expr && call.head === :call ||
        throw(ArgumentError("@scalar_rule: expected a function call on the left, got `$call`"))
    f    = call.args[1]
    args = call.args[2:end]
    nargs = length(args)

    # RHS: either a single expr (unary) or a tuple (one element per argument).
    exprs = if rhs isa Expr && rhs.head === :tuple
        rhs.args
    else
        [rhs]
    end
    length(exprs) == nargs || throw(ArgumentError(
        "@scalar_rule: $nargs argument(s) in `$call` but $(length(exprs)) " *
        "partial expression(s) on the right"))

    # arg declarations for the method signature: each is `name::AbstractScalar`.
    sig = [:($(a)::AbstractScalar) for a in args]

    # Generate one `derivative` method per partial.
    methods = map(enumerate(exprs)) do (i, expr)
        quote
            StencilCore.derivative(::typeof($(esc(f))), ::Val{$i}, $(esc.(sig)...)) = $(esc(expr))
        end
    end
    Expr(:block, methods...)
end

# --- Derivative table (frule-shape: ∂f/∂(arg i)) ---

"""
    derivative(f, ::Val{i}, args...) -> AbstractScalar

The symbolic partial derivative `∂f/∂(argᵢ)` of a primitive `f` applied to
scalar `args` (ChainRules frule style). The `0`/`1` cases use the structural
[`Null`](@ref) / [`Unity`](@ref) carriers via `_unity` (which routes through
`_unity_space` so a Number leaf stays Number and an SVector leaf lands in the
square SMatrix Jacobian space).
"""
function derivative end

# Joined element type across scalar args (used by the derivative table for
# Number-compatible operand pairs; under Q1=(c) cross-shape mixes don't reach
# these call-sites because `_diff_scalar` short-circuits via `J`).
_joined_eltype(args) = mapreduce(eltype, promote_type, args)

# Multiplicative identity of the `T → T` linear-map space — the "1" in the
# chain rule. Delegates to `Unity(T)`'s outer ctor, which routes through
# `_unity_space` to land Number → Number, SVector{N,F} → SMatrix{N,N,F}.
_unity(::Type{T}) where {T} = Unity(T)

# Jacobian eltype `J = Jac(Tout, Tin)` for the top-level derivative. Under
# Q1=(c) only Number/Number and matching-N SVector/SVector pairs are allowed.
_jacobian_type(::Type{Tout}, ::Type{Tin}) where {Tout<:Number, Tin<:Number} =
    promote_type(Tout, Tin)
# `similar_type` fills in the trailing `L = N*N`, giving the concrete form
# `SMatrix{N, N, F, N*N}` that `_assert_concrete` accepts.
_jacobian_type(::Type{SVector{N, F1}}, ::Type{SVector{N, F2}}) where {N, F1, F2} =
    similar_type(SMatrix{N, N, F1}, promote_type(F1, F2))
# Same-type fallback: the self-Jacobian of `T` is `T` itself (e.g. SMatrix is
# its own linear-map space; user-defined types must satisfy `one(T)` ↦ identity).
_jacobian_type(::Type{T}, ::Type{T}) where {T} = T
_jacobian_type(::Type{T1}, ::Type{T2}) where {T1, T2} = throw(ArgumentError(
    "differentiate: unsupported shape pair eltype(s)=$T1 vs eltype(v)=$T2. " *
    "Only (Number, Number) and matching-N (SVector{N}, SVector{N}) pairs are " *
    "supported by default. To enable additional combinations, add a method:\n" *
    "    StencilCore._jacobian_type(::Type{$T1}, ::Type{$T2}) = <Jacobian type>"))

# Dispatch every entry on Vararg{AbstractScalar} so StencilCalculus can add its
# disjoint Vararg{AbstractPointwise} methods to the same generic.
derivative(::typeof(+), ::Val,    args::Vararg{AbstractScalar}) = _unity(_joined_eltype(args))
derivative(::typeof(-), ::Val{1}, x::AbstractScalar)            = Scalar(-, (_unity(eltype(x)),))
derivative(::typeof(-), ::Val{1}, x::AbstractScalar, y::AbstractScalar) = _unity(_joined_eltype((x, y)))
derivative(::typeof(-), ::Val{2}, x::AbstractScalar, y::AbstractScalar) = Scalar(-, (_unity(_joined_eltype((x, y))),))
derivative(::typeof(*), ::Val{1}, x::AbstractScalar, y::AbstractScalar) = y
derivative(::typeof(*), ::Val{2}, x::AbstractScalar, y::AbstractScalar) = x
derivative(::typeof(/), ::Val{1}, x::AbstractScalar, y::AbstractScalar) = Scalar(/, (_unity(_joined_eltype((x, y))), y))
derivative(::typeof(/), ::Val{2}, x::AbstractScalar, y::AbstractScalar) = Scalar(-, (Scalar(/, (x, Scalar(*, (y, y)))),))
derivative(::typeof(^), ::Val{1}, x::AbstractScalar, n::AbstractScalar) =
    Scalar(*, (n, Scalar(^, (x, Scalar(-, (n, _unity(_joined_eltype((n,))) ))))))
derivative(::typeof(sin),  ::Val{1}, x::AbstractScalar) = Scalar(cos, (x,))
derivative(::typeof(cos),  ::Val{1}, x::AbstractScalar) = Scalar(-, (Scalar(sin, (x,)),))
derivative(::typeof(exp),  ::Val{1}, x::AbstractScalar) = Scalar(exp, (x,))
derivative(::typeof(log),  ::Val{1}, x::AbstractScalar) = Scalar(/, (_unity(_joined_eltype((x,))), x))
derivative(::typeof(sqrt), ::Val{1}, x::AbstractScalar) =
    Scalar(/, (_unity(_joined_eltype((x,))), Scalar(*, (Constant(2), Scalar(sqrt, (x,))))))
derivative(::typeof(tan),  ::Val{1}, x::AbstractScalar) =
    # ∂tan(x)/∂x = 1 + tan²(x); avoids introducing sec.
    _unity(_joined_eltype((x,))) + Scalar(*, (Scalar(tan, (x,)), Scalar(tan, (x,))))
derivative(::typeof(abs),  ::Val{1}, x::AbstractScalar) =
    # ∂|x|/∂x = sign(x); undefined at x = 0 (caller's responsibility).
    Scalar(sign, (x,))
derivative(f, ::Val, args::Vararg{AbstractScalar}) =
    throw(ArgumentError("no scalar derivative rule for $(f)"))

# --- Leaf rules ------------------------------------------------------------

# A scalar leaf either matches the variable (derivative = structural Unity in
# `J`-shape) or it does not (derivative = `Null{J}`). Under Q1=(c), all
# leaf-level Jacobians collapse to the *outer* Jacobian eltype `J` threaded
# through the recursion.
_diff_scalar(::Constant, v::Var, J::Type) = Null{_to_bool_shape(J)}()
_diff_scalar(::Unity,    v::Var, J::Type) = Null{_to_bool_shape(J)}()
_diff_scalar(::Null,     v::Var, J::Type) = Null{_to_bool_shape(J)}()
_diff_scalar(s::Var{S2}, v::Var{S}, J::Type) where {S2, S} =
    S2 === S ? _unity(eltype(v)) : Null{_to_bool_shape(J)}()

# --- Chain rule on Scalar interior nodes -----------------------------------

# Walk children; for each non-Null sub-derivative, compose with the primitive's
# `derivative` and accumulate (sum). Null contributions short-circuit so we
# never call `derivative` for a branch that contributes nothing.
#
# Q3=(A): for `*` with arg-index 1, swap to `sub * dfn` (left-multiply) — the
# correct order for non-commutative multiplication. Commutative cases
# (Number×Number, Number×SArray) are unaffected.
function _diff_scalar(s::Scalar, v::Var, J::Type)
    out = nothing
    for (i, arg) in enumerate(s.args)
        sub = _diff_scalar(arg, v, J)
        sub isa Null && continue
        dfn     = derivative(s.fn, Val(i), s.args...)
        contrib = if s.fn === (*) && i == 1
            simplify(Scalar(*, (sub, dfn)))
        else
            simplify(Scalar(*, (dfn, sub)))
        end
        out = (out === nothing) ? contrib : simplify(Scalar(+, (out, contrib)))
    end
    return out === nothing ? Null{_to_bool_shape(J)}() : out
end

# --- Public entry point ----------------------------------------------------

"""
    differentiate(s::AbstractScalar, v::Var{S}) -> AbstractScalar

Differentiate `s` with respect to the named scalar parameter `v` (matched on
the symbol only). The result is an `AbstractScalar` whose `eltype` is the
Jacobian element type `J = _jacobian_type(eltype(s), eltype(v))`:
`promote_type(Tout, Tin)` for Number/Number, `SMatrix{N, N, promote_type(F1,
F2)}` for matching-N `SVector{N, F1}` / `SVector{N, F2}`. Mixed shape-classes
throw. Returns `Null{J}()` when `s` does not depend on `v`; otherwise the
chain-rule expression, simplified. Scalar-side analogue of
[`StencilCalculus.differentiate`](@ref).
"""
function differentiate(s::AbstractScalar, v::Var)
    J = _jacobian_type(eltype(s), eltype(v))
    _diff_scalar(simplify(s), v, J)
end
