# Rule rewriter for AbstractScalar. Mirrors StencilCalculus/src/simplify.jl on
# the scalar side: post-walks a scalar tree (children first), applies the first
# matching rule per node, and repeats to a fixed point. Equality for the
# fixed-point check is `===` (structural — every node bottoms out in egal data).

# --- Default rules ---------------------------------------------------------

# 1. Identity / annihilator. Purely *structural*: `Null` and `Unity` are
#    matched by type (dispatch), never by `.val`. Numerical zeros / ones
#    sitting in a `Constant`'s `.val` are not collapsed by this rule — they
#    belong to a future static-encoding pass (e.g. `StaticInt`).
#
#    Shape-stability gates on `Null` and `Unity`:
#    - `Null` additive rules carry `eltype(s) === eltype(arg)` to prevent
#      shape-changing broadcast identities (e.g. `zero(SVector) + scalar`
#      should not collapse to the scalar — broadcasting gives a vector).
#    - `Unity` multiplicative rules use `_right_unity_shape` /
#      `_left_unity_shape` to additionally allow cross-precision same-size
#      matrix identities (e.g. `SMatrix{Int} * Unity{SMatrix{Float}} →
#      SMatrix{Int}`) while blocking shape-changing ones (e.g. `Int *
#      Unity{SMatrix}` must stay a `Scalar`).

# Shape-compatibility helpers for the Unity identity rules.
# These cover the *cross-precision* cases that the `eltype(s) === eltype(arg)`
# gate does not: when `one(J) * v == v` (or `v * one(J) == v`) holds in value
# but `eltype(s)` is wider than `eltype(v)`.  Returning `v` (narrower type) is
# deliberate — the convention is to preserve the more-specific stored type.

# `v::T * one(J) == v`: safe when both are square matrices of the same size N.
_right_unity_shape(::Type, ::Type) = false
_right_unity_shape(::Type{SMatrix{N,N,Tx,L}}, ::Type{SMatrix{N,N,Tf,L}}) where {N,Tx,Tf,L} = true

# `one(J) * v::T == v`: same-size square matrices, plus identity × column vector.
_left_unity_shape(::Type, ::Type) = false
_left_unity_shape(::Type{SMatrix{N,N,Tx,L}}, ::Type{SMatrix{N,N,Tf,L}}) where {N,Tx,Tf,L} = true
_left_unity_shape(::Type{SVector{N,Tx}}, ::Type{SMatrix{N,N,Tf,L}}) where {N,Tx,Tf,L} = true

# 1. Additive identity. Purely *structural*: `Null` is matched by type
#    (dispatch), never by `.val`. Shape-stability gate: `eltype(s) ===
#    eltype(arg)` prevents broadcast identities that would change the
#    expression shape (e.g. `zero(SVector) + scalar` must not collapse to
#    the scalar — broadcasting gives a vector, not a scalar).
rule_additive_identity(::AbstractScalar) = nothing
function rule_additive_identity(s::Scalar)
    f, a = s.fn, s.args
    if f === (+) && length(a) == 2
        (a[1] isa Null && eltype(s) === eltype(a[2])) && return a[2]
        (a[2] isa Null && eltype(s) === eltype(a[1])) && return a[1]
    elseif f === (-) && length(a) == 2
        (a[2] isa Null && eltype(s) === eltype(a[1])) && return a[1]
        # 0 - b = -b, only when unary minus preserves the expression eltype.
        (a[1] isa Null && Base.promote_op(-, eltype(a[2])) === eltype(s)) &&
            return Scalar(-, (a[2],))
    end
    return nothing
end

# 2. Multiplicative identity / annihilator. `Null` is the annihilator for `*`
#    and `0/x`; `Unity` is the identity for `*`, `/`, and `x/x`.
#    `_right_unity_shape` / `_left_unity_shape` additionally allow
#    cross-precision same-size matrix identities while blocking shape-changing
#    ones. Returning the narrower stored type is deliberate.
rule_multiplicative_identity(::AbstractScalar) = nothing
function rule_multiplicative_identity(s::Scalar)
    f, a = s.fn, s.args
    if f === (*) && length(a) == 2
        (a[1] isa Null || a[2] isa Null) && return Null{eltype(s)}()
        if a[1] isa Unity
            a[2] isa Unity && return Unity{eltype(s)}()        # I * I = I
            (eltype(s) === eltype(a[2]) || _left_unity_shape(eltype(a[2]), eltype(a[1]))) &&
                return a[2]
        elseif a[2] isa Unity
            (eltype(s) === eltype(a[1]) || _right_unity_shape(eltype(a[1]), eltype(a[2]))) &&
                return a[1]
        end
    elseif f === (/) && length(a) == 2
        if a[2] isa Unity
            a[1] isa Unity && return Unity{eltype(s)}()        # I / I = I
            (eltype(s) === eltype(a[1]) || _right_unity_shape(eltype(a[1]), eltype(a[2]))) &&
                return a[1]
        end
        a[1] isa Null && return Null{eltype(s)}()
    end
    return nothing
end

# 3. Double negation: -(-x) → x.
rule_double_negation(::AbstractScalar) = nothing
function rule_double_negation(s::Scalar)
    s.fn === (-) && length(s.args) == 1 || return nothing
    inner = s.args[1]
    inner isa Scalar && inner.fn === (-) && length(inner.args) == 1 &&
        return inner.args[1]
    return nothing
end

# 2. Folding. Two paths.
#    Path 1 — *coefficient fold*: every arg is Number-coefficient-decomposable
#    via `_coef`. Restricted to Number-eltype results: emits
#    `Constant{eltype(s)}(folded)`. Non-Number eltypes (e.g. SMatrix Jacobians)
#    are not folded here — those trees remain as `Scalar` nodes until
#    `materialize` evaluates them.
#    Path 2 — *direct fold*: every arg is a `Constant` (possibly carrying a
#    non-Number value like `SVector`). Apply `s.fn` to the `.val`s directly
#    and emit `Constant{eltype(s)}`.

# Number coefficient for coefficient-foldable carriers. `nothing` ⇒ not
# coefficient-foldable (the carrier holds a full non-Number value).
#
# Unity / Null contribute the structural multiplicative / additive identity:
# `Bool(true)` / `Bool(false)`. Bool is the *universal* scalar identity —
# `x * true === x`, `x + false === x` for any Number `x`, with no type
# widening.
_coef(c::Constant{T}) where {T <: Number} = c.val
_coef(::Constant)                         = nothing
_coef(::Unity)                            = true
_coef(::Null)                             = false
_coef(::AbstractScalar)                   = nothing

const _SCALAR_FOLDABLE = Set{Any}((+, -, *, /, \, ^, min, max))

# 2a. Coefficient fold (Path 1): every arg is Number-coefficient-decomposable
#     via `_coef`. Restricted to Number-eltype results: emits
#     `Constant{eltype(s)}(folded)`. Non-Number eltypes (e.g. SMatrix Jacobians)
#     are not folded here — those trees remain as `Scalar` nodes until
#     `materialize` evaluates them.
rule_coefficient_fold(::AbstractScalar) = nothing
function rule_coefficient_fold(s::Scalar)
    s.fn in _SCALAR_FOLDABLE || return nothing
    eltype(s) <: Number       || return nothing
    coefs = map(_coef, s.args)
    all(c -> c !== nothing, coefs) || return nothing
    Constant{eltype(s)}(s.fn(coefs...))
end

# 2b. Constant fold (Path 2): all args are `Constant` — apply `s.fn` to the
#     `.val`s directly and emit `Constant{eltype(s)}`. Handles non-Number values
#     such as `Constant(SVector(1,0)) + Constant(SVector(0,1))`.
rule_constant_fold(::AbstractScalar) = nothing
function rule_constant_fold(s::Scalar)
    s.fn in _SCALAR_FOLDABLE               || return nothing
    all(a -> a isa Constant, s.args)        || return nothing
    Constant{eltype(s)}(s.fn(map(a -> a.val, s.args)...))
end

"""
    SCALAR_DEFAULT_RULES

Default simplification rule set used by [`simplify`](@ref). Each element is a
callable with the signature:

    rule(s::AbstractScalar) -> Union{AbstractScalar, Nothing}

A rule returns a replacement node when applicable, or `nothing` to pass to
the next rule. Rules are tried **in order**; the first non-`nothing` result
wins for a given node. The post-walk in `_scalar_rebuild` guarantees that
children are already in normal form when a rule fires on their parent.

Built-in rules (in application order):

| Rule                          | Action                                 |
|:------------------------------|:---------------------------------------|
| `rule_additive_identity`      | `0+x→x`, `x+0→x`, `0-x→-x`, `x-0→x` |
| `rule_multiplicative_identity`| `0*x→0`, `1*x→x`, `x/1→x`, `0/x→0`   |
| `rule_double_negation`        | `-(-x) → x`                            |
| `rule_coefficient_fold`       | Number-valued `Constant`/`Unity`/`Null` args → single `Constant` |
| `rule_constant_fold`          | All-`Constant` args → direct `Constant` |

To extend, compose a new tuple that includes the new rule(s) and pass it as
the `rules` keyword argument to `simplify`:

```julia
my_rules = (my_rule, StencilCore.SCALAR_DEFAULT_RULES...)
simplify(expr; rules = my_rules)
```
"""
const SCALAR_DEFAULT_RULES = (
    rule_additive_identity,
    rule_multiplicative_identity,
    rule_double_negation,
    rule_coefficient_fold,
    rule_constant_fold,
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
Folding combines numerical coefficients across [`Constant`](@ref) / `Unity` /
`Null` args (Path 1, Number-eltype results only) or direct values across
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
