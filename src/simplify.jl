# Rule rewriter for AbstractScalar. Mirrors StencilCalculus/src/simplify.jl on
# the scalar side: post-walks a scalar tree (children first), applies the first
# matching rule per node, and repeats to a fixed point. Equality for the
# fixed-point check is `===` (structural — every node bottoms out in egal data).

# --- Default rules ---------------------------------------------------------

# 1. Identity / annihilator. Purely *structural*: `Null` and `Unity` are
#    matched by type (dispatch), never by `.val`. Numerical zeros / ones
#    sitting in a `Constant` or `Scaling`'s `.val` are not collapsed by this
#    rule — they belong to a future static-encoding pass (e.g. `StaticInt`).
#    The eltype-preservation gate on `Unity` guards against shape-changing
#    multiplications (e.g. `Unity{SMatrix} * Constant{Int}` must stay a
#    `Scalar` because returning the Int operand would silently drop the
#    matrix eltype).

rule_identity_scalar(::AbstractScalar) = nothing
function rule_identity_scalar(s::Scalar)
    f, a = s.fn, s.args
    if f === (+) && length(a) == 2
        a[1] isa Null && return a[2]
        a[2] isa Null && return a[1]
    elseif f === (-) && length(a) == 2
        a[2] isa Null && return a[1]
        a[1] isa Null && return Scalar(-, (a[2],))            # 0 - b = -b
    elseif f === (*) && length(a) == 2
        (a[1] isa Null || a[2] isa Null) && return Null{eltype(s)}()
        (a[1] isa Unity && eltype(s) === eltype(a[2])) && return a[2]
        (a[2] isa Unity && eltype(s) === eltype(a[1])) && return a[1]
    elseif f === (/) && length(a) == 2
        (a[2] isa Unity && eltype(s) === eltype(a[1])) && return a[1]
        a[1] isa Null  && return Null{eltype(s)}()
    elseif f === (-) && length(a) == 1                         # double negation
        inner = a[1]
        inner isa Scalar && inner.fn === (-) && length(inner.args) == 1 &&
            return inner.args[1]
    end
    return nothing
end

# 2. Folding. Two paths.
#    Path 1 — *coefficient fold*: every arg is shape-decomposable into a Number
#    coefficient (`_coef`). Apply `s.fn` to the coefficients and emit
#    `Constant{eltype(s)}` (Number eltype) or `Scaling{eltype(s)}` (non-Number
#    eltype). Captures the differentiation pipeline result, e.g.
#    `Constant(2) * Unity{SMatrix}() → Scaling{SMatrix}(2)`.
#    Path 2 — *direct fold*: every arg is a `Constant` (possibly carrying a
#    non-Number value like `SVector`). Apply `s.fn` to the `.val`s directly
#    and emit `Constant{eltype(s)}`.

# Number coefficient for shape-decomposable carriers. `nothing` ⇒ not
# coefficient-foldable (the carrier holds a full non-Number value).
#
# Unity / Null contribute the structural multiplicative / additive identity:
# `Bool(true)` / `Bool(false)`. Bool is the *universal* scalar identity —
# `x * true === x`, `x + false === x` for any Number `x`, with no type
# widening. Returning `one(eltype(T))` / `zero(eltype(T))` would force a
# spurious promotion via `eltype(T)`'s type (e.g. `Int(2) * one(Float64)
# === 2.0::Float64`), changing the stored `V` for no semantic reason.
_coef(c::Constant{T}) where {T <: Number} = c.val
_coef(::Constant)                         = nothing
_coef(s::Scaling)                         = s.val
_coef(::Unity)                            = true
_coef(::Null)                             = false
_coef(::AbstractScalar)                   = nothing

const _SCALAR_FOLDABLE = (+, -, *, /, \, ^, min, max)
rule_fold_scalar(::AbstractScalar) = nothing
function rule_fold_scalar(s::Scalar)
    any(==(s.fn), _SCALAR_FOLDABLE) || return nothing

    coefs = map(_coef, s.args)
    if all(c -> c !== nothing, coefs)
        folded = s.fn(coefs...)
        return eltype(s) <: Number ? Constant{eltype(s)}(folded) :
                                     Scaling{eltype(s)}(folded)
    end

    if all(a -> a isa Constant, s.args)
        return Constant{eltype(s)}(s.fn(map(a -> a.val, s.args)...))
    end

    nothing
end

# 3. Canonicalise a `Scaling` coefficient inside a `*` / `/` node to its
#    equivalent `Constant` when doing so preserves the parent's eltype. This
#    closes the algebraic equivalence
#
#        Scaling{S}(c) * y  ≡  Constant(c) * y    when one(S) acts as identity
#                                                  on `y`'s value space
#
#    that neither the structural identity rule (no `Null`/`Unity` in the tree)
#    nor the value fold rule (a non-foldable sibling like `Symbolic` blocks
#    Path 1) catches. The eltype-preservation gate keeps the rule conservative:
#    `Scaling{SMatrix}(c) * Constant{Int}` does NOT collapse — that would
#    silently demote the matrix coefficient.
rule_collapse_scaling(::AbstractScalar) = nothing
function rule_collapse_scaling(s::Scalar)
    (s.fn === (*) || s.fn === (/)) && length(s.args) == 2 || return nothing
    a1, a2 = s.args
    new_a1 = a1 isa Scaling ? Constant(a1.val) : a1
    new_a2 = a2 isa Scaling ? Constant(a2.val) : a2
    (new_a1 === a1 && new_a2 === a2) && return nothing
    Base.promote_op(s.fn, eltype(new_a1), eltype(new_a2)) === eltype(s) || return nothing
    Scalar(s.fn, (new_a1, new_a2))
end

const SCALAR_DEFAULT_RULES = (
    rule_identity_scalar,
    rule_fold_scalar,
    rule_collapse_scaling,
)

# --- Rewriter --------------------------------------------------------------

# Apply the first matching rule to `s` (else return `s` unchanged).
function _scalar_apply(s::AbstractScalar, rules)
    for r in rules
        u = r(s)
        u === nothing || return u
    end
    return s
end

# Rebuild a node with simplified children, reusing the node when unchanged
# (so `===` detects quiescence).
_scalar_rebuild(s::AbstractScalar, rules) = s                  # leaves
function _scalar_rebuild(s::Scalar, rules)
    newargs = map(a -> _scalar_rewrite(a, rules), s.args)
    newargs === s.args ? s : Scalar(s.fn, newargs)
end

_scalar_rewrite(s::AbstractScalar, rules) =
    _scalar_apply(_scalar_rebuild(s, rules), rules)

"""
    simplify(s::AbstractScalar, rules = SCALAR_DEFAULT_RULES; maxsteps = 64)

Rewrite a scalar tree to a normal form by post-walking and applying `rules` to
a fixed point. The default rules are *purely structural*: identities and
annihilators dispatch on [`Null`](@ref) (additive zero) and [`Unity`](@ref)
(multiplicative one, with an eltype-preservation gate), never on `.val`.
Folding combines numerical coefficients across [`Constant`](@ref) /
[`Scaling`](@ref) / `Unity` / `Null` args (Path 1) or direct values across
all-`Constant` args (Path 2). The scalar-side analogue of
[`StencilCalculus.simplify`](@ref).
"""
function simplify(s::AbstractScalar, rules = SCALAR_DEFAULT_RULES; maxsteps::Int = 64)
    for _ in 1:maxsteps
        s′ = _scalar_rewrite(s, rules)
        s′ === s && return s
        s = s′
    end
    @warn "StencilCore.simplify hit the step budget; returning current form" maxsteps
    return s
end
