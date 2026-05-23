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
`Null`/`Unity`.
"""
function derivative end

_se(args) = mapreduce(eltype, promote_type, args)   # promoted scalar eltype

derivative(::typeof(+), ::Val,    args...) = Unity{_se(args)}()
derivative(::typeof(-), ::Val{1}, x)       = Scalar(-, (Unity{eltype(x)}(),))
derivative(::typeof(-), ::Val{1}, x, y)    = Unity{_se((x, y))}()
derivative(::typeof(-), ::Val{2}, x, y)    = Scalar(-, (Unity{_se((x, y))}(),))
derivative(::typeof(*), ::Val{1}, x, y)    = y
derivative(::typeof(*), ::Val{2}, x, y)    = x
derivative(::typeof(/), ::Val{1}, x, y)    = Scalar(/, (Unity{_se((x, y))}(), y))
derivative(::typeof(/), ::Val{2}, x, y)    = Scalar(-, (Scalar(/, (x, Scalar(*, (y, y)))),))
derivative(::typeof(^), ::Val{1}, x, n)    = Scalar(*, (n, Scalar(^, (x, Scalar(-, (n, Unity{_se((n,))}()))))))
derivative(::typeof(sin),  ::Val{1}, x)    = Scalar(cos, (x,))
derivative(::typeof(cos),  ::Val{1}, x)    = Scalar(-, (Scalar(sin, (x,)),))
derivative(::typeof(exp),  ::Val{1}, x)    = Scalar(exp, (x,))
derivative(::typeof(log),  ::Val{1}, x)    = Scalar(/, (Unity{_se((x,))}(), x))
derivative(::typeof(sqrt), ::Val{1}, x)    = Scalar(/, (Unity{_se((x,))}(), Scalar(*, (Const(2), Scalar(sqrt, (x,))))))
derivative(f, ::Val, args...) =
    throw(ArgumentError("no scalar derivative rule for $(f)"))

# --- Leaf rules ------------------------------------------------------------

# A scalar leaf either matches the variable (derivative = Unity) or it does
# not (derivative = Null). Const/Null/Unity are constants w.r.t. anything.
_sdiff(c::Const, ::Symbolic) = Null{eltype(c)}()
_sdiff(n::Null,  ::Symbolic) = n
_sdiff(u::Unity, ::Symbolic) = Null{eltype(u)}()
_sdiff(::Symbolic{S2, T}, ::Symbolic{S}) where {S2, T, S} =
    S2 === S ? Unity{T}() : Null{T}()

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
    return out === nothing ? Null{eltype(s)}() : out
end

# --- Public entry point ----------------------------------------------------

"""
    differentiate(s::AbstractScalar, v::Symbolic{S}) -> AbstractScalar

Differentiate `s` with respect to the named scalar parameter `v` (matched on
the symbol only). Returns `Null{eltype(s)}()` when `s` does not depend on
`v`; otherwise the chain-rule expression, simplified. Scalar-side analogue
of [`StencilCalculus.differentiate`](@ref).
"""
function differentiate(s::AbstractScalar, v::Symbolic)
    _sdiff(simplify(s), v)
end
