# Reduction of an AbstractScalar tree to a single value (`materialize`) or to
# a Julia `Expr` evaluating it in a host kernel (`_scalar_body_expr`).
#
# Unlike StencilCalculus's `materialize`, which produces a `LazyArray` over a
# grid box, this returns one scalar value: an AbstractScalar has no axes. The
# `_scalar_body_expr` form is used by Calculus's per-cell codegen when it
# descends into a `Fill{<:AbstractScalar}` — the scalar is evaluated once per
# cell (no `idx`-dependence).

"""
    materialize(s::AbstractScalar, pairs::NamedTuple = (;))

Substitute the [`Symbolic`](@ref) leaves named in `pairs` into `s` and reduce
to a single value of type `eltype(s)`. The scalar-side analogue of
[`StencilCalculus.materialize`](@ref).
"""
materialize(::Symbolic{S}, pairs::NamedTuple) where {S}             = pairs[S]
materialize(::Null{T}, ::NamedTuple = (;)) where {T}                = zero(T)
materialize(s::Scaling{V, T}, ::NamedTuple = (;)) where {V, T}      = s.val * one(T)
materialize(s::Scalar, pairs::NamedTuple = (;))                     =
    s.fn(map(a -> materialize(a, pairs), s.args)...)

"""
    _scalar_body_expr(s::AbstractScalar) -> Expr or literal

Lower a scalar tree to an `Expr` that evaluates it in the StencilCalculus
codegen `args::NamedTuple` context (no per-cell indices — scalars are
position-independent). Used by `_body_expr(::Fill{<:AbstractScalar}, …)` on
the term side. Returns an `Expr` suitable for embedding in a larger AST.
"""
_scalar_body_expr(::Symbolic{S}) where {S}       = Expr(:., :args, QuoteNode(S))
_scalar_body_expr(::Null{T}) where {T}           = Expr(:call, :zero, T)
_scalar_body_expr(s::Scaling{V, T}) where {V, T} = Expr(:call, :*, s.val, Expr(:call, :one, T))
_scalar_body_expr(s::Scalar) =
    Expr(:call, nameof(s.fn), (_scalar_body_expr(a) for a in s.args)...)
