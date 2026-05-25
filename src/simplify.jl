# Rule rewriter for AbstractScalar. Mirrors StencilCalculus/src/simplify.jl on
# the scalar side: post-walks a scalar tree (children first), applies the first
# matching rule per node, and repeats to a fixed point. Equality for the
# fixed-point check is `===` (structural — every node bottoms out in egal data).

# --- Default rules ---------------------------------------------------------

# 1. Identity / annihilator. `Null` is matched by *type* (structural zero);
#    multiplicative identity is matched on `Scaling` by *value* (`isone(.val)`).
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
        (a[1] isa Scaling && isone(a[1].val)) && return a[2]
        (a[2] isa Scaling && isone(a[2].val)) && return a[1]
    elseif f === (/) && length(a) == 2
        (a[2] isa Scaling && isone(a[2].val)) && return a[1]
        a[1] isa Null  && return Null{eltype(s)}()
    elseif f === (-) && length(a) == 1                         # double negation
        inner = a[1]
        inner isa Scalar && inner.fn === (-) && length(inner.args) == 1 &&
            return inner.args[1]
    end
    return nothing
end

# 2. Constant folding over an allow-listed pure operator. Produces a `Scaling`
#    constructed from the folded value (Scaling is the literal leaf in this
#    algebra; the pre-simplified-input assumption means we never need to
#    canonicalise to `Null` here).
const _SCALAR_FOLDABLE = (+, -, *, /, \, ^, min, max)
rule_fold_scalar(::AbstractScalar) = nothing
function rule_fold_scalar(s::Scalar)
    (all(a -> a isa Scaling, s.args) && any(==(s.fn), _SCALAR_FOLDABLE)) || return nothing
    Scaling(s.fn(map(a -> a.val, s.args)...))
end

const SCALAR_DEFAULT_RULES = (
    rule_identity_scalar,
    rule_fold_scalar,
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
a fixed point. The default rules apply additive/multiplicative identities
(`Null` by type, `Scaling(1)` by value) and fold all-`Scaling` arguments. The
scalar-side analogue of [`StencilCalculus.simplify`](@ref).
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
