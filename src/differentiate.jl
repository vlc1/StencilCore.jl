# Symbolic differentiation of an AbstractScalar with respect to a Symbolic.
# Scalar-side analogue of StencilCalculus/src/differentiate.jl — but with no
# spatial dimension: no per-offset contributions, no Stencil. The result is a
# single AbstractScalar (`Null{T}()` when `s` does not depend on the variable;
# otherwise the chain-rule expression, simplified).

# --- Derivative table (frule-shape: ∂f/∂(arg i)) ---

"""
    derivative(f, ::Val{i}, args...) -> AbstractScalar

The symbolic partial derivative `∂f/∂(argᵢ)` of a primitive `f` applied to
scalar `args` (ChainRules frule style). The `0`/`1` cases use the type-level
`Null` / value-level `Scaling(one(...))`.
"""
function derivative end

_se(args) = mapreduce(eltype, promote_type, args)   # promoted scalar eltype
_unity(T) = Scaling{T}(one(eltype(T)))              # Scaling-form of "1 of type T"

# Dispatch every entry on Vararg{AbstractScalar} so StencilCalculus can add its
# disjoint Vararg{AbstractTerm} methods to the same generic.
derivative(::typeof(+), ::Val,    args::Vararg{AbstractScalar}) = _unity(_se(args))
derivative(::typeof(-), ::Val{1}, x::AbstractScalar)            = Scalar(-, (_unity(eltype(x)),))
derivative(::typeof(-), ::Val{1}, x::AbstractScalar, y::AbstractScalar) = _unity(_se((x, y)))
derivative(::typeof(-), ::Val{2}, x::AbstractScalar, y::AbstractScalar) = Scalar(-, (_unity(_se((x, y))),))
derivative(::typeof(*), ::Val{1}, x::AbstractScalar, y::AbstractScalar) = y
derivative(::typeof(*), ::Val{2}, x::AbstractScalar, y::AbstractScalar) = x
derivative(::typeof(/), ::Val{1}, x::AbstractScalar, y::AbstractScalar) = Scalar(/, (_unity(_se((x, y))), y))
derivative(::typeof(/), ::Val{2}, x::AbstractScalar, y::AbstractScalar) = Scalar(-, (Scalar(/, (x, Scalar(*, (y, y)))),))
derivative(::typeof(^), ::Val{1}, x::AbstractScalar, n::AbstractScalar) =
    Scalar(*, (n, Scalar(^, (x, Scalar(-, (n, _unity(_se((n,))) ))))))
derivative(::typeof(sin),  ::Val{1}, x::AbstractScalar) = Scalar(cos, (x,))
derivative(::typeof(cos),  ::Val{1}, x::AbstractScalar) = Scalar(-, (Scalar(sin, (x,)),))
derivative(::typeof(exp),  ::Val{1}, x::AbstractScalar) = Scalar(exp, (x,))
derivative(::typeof(log),  ::Val{1}, x::AbstractScalar) = Scalar(/, (_unity(_se((x,))), x))
derivative(::typeof(sqrt), ::Val{1}, x::AbstractScalar) =
    Scalar(/, (_unity(_se((x,))), Scalar(*, (Scaling(2), Scalar(sqrt, (x,))))))
derivative(f, ::Val, args::Vararg{AbstractScalar}) =
    throw(ArgumentError("no scalar derivative rule for $(f)"))

# --- Leaf rules ------------------------------------------------------------

# A scalar leaf either matches the variable (derivative = unit Scaling) or it
# does not (derivative = Null). Scaling/Null are constants w.r.t. anything.
# Element type is promoted from both operands (`_se`), so the result type is
# consistent with downstream Scalar promotion.
_sdiff(c::Scaling, v::Symbolic) = Null{_se((c, v))}()
_sdiff(n::Null,    v::Symbolic) = Null{_se((n, v))}()
_sdiff(s::Symbolic{S2, T}, v::Symbolic{S}) where {S2, T, S} =
    S2 === S ? _unity(_se((s, v))) : Null{_se((s, v))}()

# --- Chain rule on Scalar interior nodes -----------------------------------

# Walk children; for each non-Null sub-derivative, multiply by the primitive's
# `derivative` and accumulate (sum). Null contributions short-circuit so we
# never call `derivative` for a branch that contributes nothing.
function _sdiff(s::Scalar, v::Symbolic)
    out = nothing
    for (i, arg) in enumerate(s.args)
        sub = _sdiff(arg, v)
        sub isa Null && continue
        dfn     = derivative(s.fn, Val(i), s.args...)
        contrib = simplify(Scalar(*, (dfn, sub)))
        out = (out === nothing) ? contrib : simplify(Scalar(+, (out, contrib)))
    end
    return out === nothing ? Null{_se((s, v))}() : out
end

# --- Public entry point ----------------------------------------------------

"""
    differentiate(s::AbstractScalar, v::Symbolic{S}) -> AbstractScalar

Differentiate `s` with respect to the named scalar parameter `v` (matched on
the symbol only). Returns `Null{T}()` (with `T` promoted from both operands)
when `s` does not depend on `v`; otherwise the chain-rule expression,
simplified. Scalar-side analogue of [`StencilCalculus.differentiate`](@ref).
"""
function differentiate(s::AbstractScalar, v::Symbolic)
    _sdiff(simplify(s), v)
end
