# Symbolic differentiation of an AbstractScalar with respect to a Symbolic.
# Scalar-side analogue of StencilCalculus/src/differentiate.jl — but with no
# spatial dimension: no per-offset contributions, no Stencil. The result is a
# single AbstractScalar (`Null{T}()` when `s` does not depend on the variable;
# otherwise the chain-rule expression, simplified).

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
# these call-sites because `_sdiff` short-circuits via `J`).
_se(args) = mapreduce(eltype, promote_type, args)

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
_jacobian_type(::Type{T1}, ::Type{T2}) where {T1, T2} = throw(ArgumentError(
    "differentiate: shape-class mismatch eltype(s)=$T1 vs eltype(v)=$T2; only " *
    "(Number, Number) and matching-N (SVector{N}, SVector{N}) are supported"))

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
    Scalar(/, (_unity(_se((x,))), Scalar(*, (Constant(2), Scalar(sqrt, (x,))))))
derivative(f, ::Val, args::Vararg{AbstractScalar}) =
    throw(ArgumentError("no scalar derivative rule for $(f)"))

# --- Leaf rules ------------------------------------------------------------

# A scalar leaf either matches the variable (derivative = structural Unity in
# `J`-shape) or it does not (derivative = `Null{J}`). Under Q1=(c), all
# leaf-level Jacobians collapse to the *outer* Jacobian eltype `J` threaded
# through the recursion.
_sdiff(::Constant, v::Symbolic, J::Type) = Null{J}()
_sdiff(::Unity,    v::Symbolic, J::Type) = Null{J}()
_sdiff(::Null,     v::Symbolic, J::Type) = Null{J}()
_sdiff(s::Symbolic{S2}, v::Symbolic{S}, J::Type) where {S2, S} =
    S2 === S ? _unity(eltype(v)) : Null{J}()

# --- Chain rule on Scalar interior nodes -----------------------------------

# Walk children; for each non-Null sub-derivative, compose with the primitive's
# `derivative` and accumulate (sum). Null contributions short-circuit so we
# never call `derivative` for a branch that contributes nothing.
#
# Q3=(A): for `*` with arg-index 1, swap to `sub * dfn` (left-multiply) — the
# correct order for non-commutative multiplication. Commutative cases
# (Number×Number, Number×SArray) are unaffected.
function _sdiff(s::Scalar, v::Symbolic, J::Type)
    out = nothing
    for (i, arg) in enumerate(s.args)
        sub = _sdiff(arg, v, J)
        sub isa Null && continue
        dfn     = derivative(s.fn, Val(i), s.args...)
        contrib = if s.fn === (*) && i == 1
            simplify(Scalar(*, (sub, dfn)))
        else
            simplify(Scalar(*, (dfn, sub)))
        end
        out = (out === nothing) ? contrib : simplify(Scalar(+, (out, contrib)))
    end
    return out === nothing ? Null{J}() : out
end

# --- Public entry point ----------------------------------------------------

"""
    differentiate(s::AbstractScalar, v::Symbolic{S}) -> AbstractScalar

Differentiate `s` with respect to the named scalar parameter `v` (matched on
the symbol only). The result is an `AbstractScalar` whose `eltype` is the
Jacobian element type `J = _jacobian_type(eltype(s), eltype(v))`:
`promote_type(Tout, Tin)` for Number/Number, `SMatrix{N, N, promote_type(F1,
F2)}` for matching-N `SVector{N, F1}` / `SVector{N, F2}`. Mixed shape-classes
throw. Returns `Null{J}()` when `s` does not depend on `v`; otherwise the
chain-rule expression, simplified. Scalar-side analogue of
[`StencilCalculus.differentiate`](@ref).
"""
function differentiate(s::AbstractScalar, v::Symbolic)
    J = _jacobian_type(eltype(s), eltype(v))
    _sdiff(simplify(s), v, J)
end
