# AGENTS.md

Canonical record of the **stencil + term-like type vocabulary** shared by
`StencilAssembly` (CSC assembly) and `StencilCalculus` (symbolic CAS).
StencilCore owns the *types*; it has no assembly and depends only on
`StaticArrays`. CSC assembly invariants live in
[`StencilAssembly/AGENTS.md`](../StencilAssembly/AGENTS.md); the CAS
design lives in [`docs/cas.md`](../StencilCalculus/docs/cas.md).

## Sticky decisions

1. **Access-style trait.** `AccessStyle` (abstract) with singletons
   `ColumnAccess` (CSC) and `RowAccess` (CSR, reserved).
   `AbstractStencil{S<:AccessStyle}` is the supertype; every stencil
   carries `S` as its **last** type parameter. The accessor
   `AccessStyle(st)` / `AccessStyle(::Type)` lives once at the supertype.

2. **`AbstractTerm{T}` — array-like, dimension-/size-less.** `T` is the
   materialized element type (concrete or abstract): a term "behaves like
   an array whose `eltype` is `T`", with grid rank `N` unknown until it is
   substituted/materialized. `eltype(::Type{<:AbstractTerm{T}}) = T`.
   Concrete subtypes live in `StencilCalculus`; StencilCore owns only the
   abstract type so coefficients can be named without depending on the CAS.

3. **`ArrayOrTermLike{T} = Union{AbstractArray{T}, AbstractTerm{T}}`** is
   the coefficient type of every stencil. A stencil is **assemblable** when
   its coefficient is a concrete `AbstractArray`, **symbolic** when it is an
   `AbstractTerm` (must be `materialize`d first).

4. **Type-level offsets (`StaticPair` / `StaticShift`).** Shifts enter via
   compile-time-known DSL operators, so they are encoded in the type system
   (same footing as `LinearStencil`'s `O`/`L`).
   `StaticPair{D, O}` (alias `SPair`) is one offset `O` along axis `D`.
   `StaticShift{P<:Tuple{Vararg{StaticPair}}}` (alias `SShift`) is a
   **normalized** collection — pairs sorted ascending by `D`, no duplicate
   `D` (same-`D` summed), no zero `O` (dropped); the empty shift is the
   identity. Type-level algebra: `+`, unary/binary `-`, `*Int`. Constants
   `ô` (zero shift) and `ê₁ … ê₉` (unit per-axis); `show` renders e.g.
   `3ê₁ + ê₂`. Accessors `dim` / `offset` (on `StaticPair`).

5. **Reverse-lexicographic offset order.** Offsets are ordered reverse-lex
   on the offset vector (axis `N` most significant — `CartesianIndex`
   order). This single convention is shared by `Stencil`, `StarStencil`,
   and `LinearStencil` so narrowing is a verbatim coefficient copy.

6. **`LinearStencil{D, O, L, E<:SVector{L}, A<:ArrayOrTermLike{E}, S}`.**
   Contiguous offsets along mesh dim `D`; `offsets::SUnitRange{O, L}`,
   `term::A`. Coefficient element `E` is the per-cell `SVector{L}` of all
   `L` per-offset coefficients in **ascending offset order** (`term[c][1]`
   ↦ offset `O`); scalar = `eltype(E)`. Offsets are **diagonal indices**
   `δ = col − row`. **`N` is not a parameter** — it is `ndims(A)` for a
   concrete coefficient (checked `D ≤ N`), unknown for a symbolic one.

7. **`StarStencil{L, N, M, E<:SVector{M}, A<:ArrayOrTermLike{E}, S}` —
   interlaced.** A single `term::A` whose per-cell element is the
   `SVector{M}` of the **whole** star, `M = 2NL + 1`, in reverse-lex offset
   order with the **diagonal as the explicit middle slot** `(M+1)/2`. This
   replaces a per-axis decomposition: the diagonal is a free coefficient
   (Helmholtz `k²`, parabolic `∂ₜ`), not a sum of per-axis centers. `N` is
   kept and validated to equal `(M−1)/(2L)` (and `ndims(A)` when concrete).
   Per-axis offset `δ` along axis `d` lands at row coord `c_d − δ`.

8. **`Stencil{M, C<:NTuple{M, StaticShift}, E<:SVector{M}, A<:ArrayOrTermLike{E}, S}`
   — the general lingua franca.** An explicit reverse-lex `shifts::C` plus a
   matching `SVector{M}`-valued `term::A`; the form `differentiate` emits.
   **Not assembled directly** — narrowed via [`as_linear`](@ref) /
   [`as_star`](@ref) (future `as_planar`).

9. **Narrowing = pattern match + verbatim `term` copy.** Because all three
   stencils share the reverse-lex layout, `as_linear` (offsets single-axis,
   same `D`, contiguous) and `as_star` (canonical reverse-lex star pattern)
   build the optimized stencil reusing `term` unchanged; mismatches raise
   `ArgumentError`.

10. **Constructors use a positional `Type` tag for `S`** (default
    `ColumnAccess`), matching across all stencils:
    `LinearStencil{D}(offsets, term)` ≡
    `LinearStencil{D}(ColumnAccess, offsets, term)`;
    `StarStencil{L}(RowAccess, term)`; `Stencil(RowAccess, shifts, term)`.
    `Stencil{S}(…)` would bind the leading `M`, so the tag is positional.
    Inner ctors validate (`D ≥ 1`/`D ≤ N`; `M = 2NL+1`; `SVector` length
    matched at the type); friendly outer ctors raise `ArgumentError`.

11. **AccessStyle = anchoring + emission, *not* transposition.** Offsets
    (`δ = col − row`, reverse-lex) are **invariant** under `S`. `S` selects
    (i) the coefficient anchor — `RowAccess` stores the row value `g_σ`,
    `ColumnAccess` stores `Shifted(−σ, g_σ)` (identical for constant
    coefficients) — and (ii) emission direction (CSC descending / CSR
    ascending). The adjoint `Aᵀ` (which would negate offsets) is a separate
    explicit operation.

12. **Scalar algebra — `AbstractScalar{T}`.** Sibling of, *not* subtype of,
    `AbstractTerm`. A scalar materialises to **one value** of type `T` (no
    axes). Five concrete leaves plus one interior node:

    - **`Symbolic{S, T}`** — named substitution leaf; `S` is a `Symbol`,
      `T` the materialised eltype.
    - **`Constant{T}`** — literal value carrier; `val::T` stored as-is and
      materialised to `val`. `T` is any concrete type (including non-`Number`
      like `SVector` — this is the carrier numeric literals canonicalise to
      at the operator boundary).
    - **`Scaling{T, V <: Number}`** — `val::V` stored, materialises to
      `val * one(T)`. `T` is the *materialised container type*; the one-curly
      inner ctor `Scaling{Traw}(val)` canonicalises `Traw` and promotes
      eltype-vs-`V` (`Scaling{Float32}(1.0)` lands at `T = Float64`). Value-
      space outer ctors `Scaling(T)` / `Scaling(T, val)` route `T` through
      `_unity_space` so `Scaling(SVector{N, F})` lands in the square Jacobian
      space `SMatrix{N, N, F}`.
    - **`Null{T}`** — structural additive zero, dispatch-matched.
    - **`Unity{T}`** — structural multiplicative one, dispatch-matched.
      Construction requires `one(T)` defined (`Number`, square `SMatrix`,
      …). Outer ctor `Unity(T)` routes through `_unity_space` so
      `Unity(SVector{N, F}) === Unity{SMatrix{N, N, F}}()`.
    - **`Scalar{F, A<:Tuple{Vararg{AbstractScalar}}, T}`** — interior node
      `fn(args…)`; `T = Base.promote_op(fn, eltype.(args)…)` computed at
      construction (a `Union{}` result throws).

    All concrete `T`s are enforced via `_assert_concrete`. `Λ` is an alias
    of `Scaling`. The `@symbolic name [T]` macro binds `name = Symbolic{:name, T}()`
    (default `T = Float64`).

13. **Operator boundary canonicalises to `Constant`.** Every binary op
    `(::AbstractScalar, x)` with `x` not an `AbstractScalar` lifts as
    `Scalar(op, (·, Constant(x)))` (and symmetrically for the left slot).
    `asscalar(x) = Constant(x)` for non-scalars; `Base.convert(::Type{<:AbstractScalar}, x) = Constant(x)`.
    The isotropic `Scaling(val::Number)` form is **not** provided — a numeric
    literal is data, not a coefficient of `one(·)`.

14. **`simplify` is purely structural.** Identity / annihilator rules
    match by *type* — `Null` (additive zero), `Unity` (multiplicative one,
    with an eltype-preservation gate so `Unity{SMatrix} * Constant{Int}` does
    not silently change eltype) — never by `.val`. Numerical zeros / ones
    sitting in `Constant.val` or `Scaling.val` are *not* collapsed to
    `Null` / `Unity`; that step waits for a future static-value encoding
    (`StaticFloat64`, `StaticInt`).

    Folding has two paths: **coefficient fold** combines Number coefficients
    from any of `Constant{<:Number}`, `Scaling`, `Unity` (coef `1`), `Null`
    (coef `0`) — emit `Constant{eltype(s)}` (Number parent) or
    `Scaling{eltype(s)}` (non-Number parent); **direct fold** applies the
    operator to `.val`s of all-`Constant` args. Folding combines values; it
    does not inspect them to decide.

15. **`differentiate` (scalar-side).** Top-level `differentiate(s, v)`
    requires `eltype(s)` and `eltype(v)` to share *shape-class* — Number or
    matching-`N` `SVector{N}` — else `ArgumentError`. The result's eltype is
    the **Jacobian** `J = _jacobian_type(eltype(s), eltype(v))`:
    `promote_type` for Number/Number, square `SMatrix{N, N, promote_type(F1,
    F2)}` for `SVector{N, F1}` / `SVector{N, F2}`. `J` is threaded through
    `_sdiff` so every `Null` and self-leaf `Unity` is typed by it. Product-
    rule composition is position-aware on `*` (Q3-A): for `i = 1`, contrib
    is `Scalar(*, (sub, dfn))` — left-multiplication, correct under non-
    commuting `*`. The derivative table is otherwise mechanical.

## Public surface

Exports: `AccessStyle`, `ColumnAccess`, `RowAccess`, `AbstractStencil`,
`AbstractTerm`, `ArrayOrTermLike`, `StaticPair`, `SPair`, `StaticShift`,
`SShift`, `dim`, `offset`, `ô`, `ê₁ … ê₉`, `LinearStencil`, `StarStencil`,
`Stencil`, `as_linear`, `as_star`,
`AbstractScalar`, `Symbolic`, `Constant`, `Scaling`, `Λ`, `Null`, `Unity`,
`Scalar`, `@symbolic`, `simplify`, `materialize`, `differentiate`,
`derivative`.

Files: `access.jl` (trait + supertype), `term.jl` (`AbstractTerm` +
`ArrayOrTermLike` + `_assert_concrete`), `staticshift.jl` (`StaticPair` /
`StaticShift` + algebra + `show`), `scalars.jl` (`AbstractScalar` family +
`@symbolic` + operator overloads + `_unity_space`), `trees.jl`
(`AbstractTrees` plumbing for scalars), `simplify.jl` (`simplify` + identity
+ fold rules), `materialize.jl` (`materialize` + `_scalar_body_expr`),
`differentiate.jl` (`differentiate` + `derivative` table + `_jacobian_type`
+ `_unity`), `structured.jl` (`LinearStencil` / `StarStencil`), `general.jl`
(`Stencil` + narrowing).

Tests: `julia --project=. -e 'using Pkg; Pkg.test()'`.

## Scope

Implemented: the type vocabulary above + `as_linear` / `as_star` narrowing
+ the scalar CAS (`simplify`, `materialize`, `differentiate`). **No
assembly** — CSC `assemble` / `update!` / `build` live in `StencilAssembly`
and dispatch on a concrete-array coefficient with `S = ColumnAccess`.

Deferred: `PlanarStencil` + `as_planar`; a direct CSC kernel for the general
`Stencil` (narrowing-only for now); CSR (`RowAccess`) assembly; non-square
SArray Jacobians (cross-shape `differentiate`); true SVector×SVector /
SMatrix×SMatrix chain rules in the scalar tree; static-value encoding for
`Constant` / `Scaling.val` (would re-enable value-based identity rules).
