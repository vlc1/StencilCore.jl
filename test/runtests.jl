using StencilCore
using Test
using AbstractTrees
using StaticArrays: SUnitRange, SVector, SMatrix

# Structs must be defined at top level (not inside @testset scopes).
struct _DummyStencil{S} <: AbstractStencil{S} end
struct _DummyTerm{T} <: AbstractPointwise{T} end

# Symbolic half of the SoA→AoS coefficient combiner (the CAS provides the real
# one): stub it for _DummyTerm so symbolic narrowing can be exercised here.
StencilCore._interlace(::NTuple{M, _DummyTerm}) where {M} = _DummyTerm{SVector{M, Float64}}()

@testset "StencilCore" begin

    @testset "AccessStyle trait" begin
        @test AccessStyle(_DummyStencil{ColumnAccess}()) === ColumnAccess()
        @test AccessStyle(_DummyStencil{RowAccess}())    === RowAccess()
        @test AccessStyle(_DummyStencil{ColumnAccess})   === ColumnAccess()
        @test ColumnAccess <: AccessStyle
        @test RowAccess    <: AccessStyle
    end

    @testset "AbstractPointwise / ArrayOrTermLike" begin
        @test eltype(_DummyTerm{Float64}) === Float64
        @test eltype(_DummyTerm{Float64}()) === Float64
        @test _DummyTerm{Float64}() isa ArrayOrTermLike{Float64}
        @test [1.0, 2.0] isa ArrayOrTermLike{Float64}
        @test !(_DummyTerm{Float64}() isa ArrayOrTermLike{Int})
    end

    @testset "StaticPair accessors" begin
        p = StaticPair{2, 3}()
        @test dim(p) == 2 && offset(p) == 3
        @test dim(StaticPair{2, 3}) == 2 && offset(StaticPair{2, 3}) == 3
        @test SPair === StaticPair && SShift === StaticShift
    end

    @testset "StaticShift normalization" begin
        # User's canonical example.
        s = SShift((SPair{1, 1}(),)) + SShift((SPair{2, 1}(),)) + SShift((SPair{1, 2}(),))
        @test s isa StaticShift{Tuple{StaticPair{1, 3}, StaticPair{2, 1}}}

        # Sorted ascending by dimension regardless of input order.
        @test SShift(SPair{3, 1}(), SPair{1, 1}()) isa
              StaticShift{Tuple{StaticPair{1, 1}, StaticPair{3, 1}}}

        # Same-dim summing to zero drops the pair.
        @test SShift(SPair{1, 2}(), SPair{1, -2}()) isa StaticShift{Tuple{}}

        # Explicit zero offset is dropped at construction.
        @test SShift(SPair{1, 0}()) isa StaticShift{Tuple{}}
        @test SShift() isa StaticShift{Tuple{}}
    end

    @testset "StaticShift algebra" begin
        @test (3ê₁ + ê₂) isa StaticShift{Tuple{StaticPair{1, 3}, StaticPair{2, 1}}}
        @test -ê₁ isa StaticShift{Tuple{StaticPair{1, -1}}}
        @test (ê₁ - ê₁) isa StaticShift{Tuple{}}
        @test (2 * ê₃) isa StaticShift{Tuple{StaticPair{3, 2}}}
        @test (ê₃ * 2) isa StaticShift{Tuple{StaticPair{3, 2}}}
        @test (0 * ê₁) isa StaticShift{Tuple{}}
        @test -(3ê₁ + ê₂) isa StaticShift{Tuple{StaticPair{1, -3}, StaticPair{2, -1}}}
        @test iszero(SShift())
        @test !iszero(ê₁)
    end

    @testset "StaticShift display" begin
        @test repr(3ê₁ + ê₂)    == "3ê₁ + ê₂"
        @test repr(ê₁)          == "ê₁"
        @test repr(-ê₁)         == "-ê₁"
        @test repr(-ê₁ + 2ê₃)   == "-ê₁ + 2ê₃"
        @test repr(ê₁ - ê₂)     == "ê₁ - ê₂"
        @test repr(SShift())    == "𝟎"
        # Multi-digit dimension subscripts (D > 9).
        @test repr(SShift(SPair{10, 1}())) == "ê₁₀"
    end

    @testset "LinearStencil construction" begin
        # Concrete-array coefficient (array-of-structs).
        arr = fill(SVector(-1.0, 1.0), 5)
        st = LinearStencil{1}(SUnitRange(0, 1), arr)
        @test st isa LinearStencil{1, 0, 2, SVector{2, Float64}, typeof(arr), ColumnAccess}
        @test st isa AbstractStencil{ColumnAccess}
        @test AccessStyle(st) === ColumnAccess()
        @test st.term === arr

        # Explicit RowAccess.
        st_row = LinearStencil{1}(RowAccess, SUnitRange(0, 1), arr)
        @test AccessStyle(st_row) === RowAccess()

        # Symbolic-term coefficient (no grid rank, no D ≤ N check).
        sym = _DummyTerm{SVector{2, Float64}}()
        sst = LinearStencil{2}(SUnitRange(0, 1), sym)
        @test sst isa LinearStencil{2, 0, 2, SVector{2, Float64}, typeof(sym), ColumnAccess}
        @test sst.term === sym

        # D ≤ N enforced for concrete arrays (2-D stencil dim on a 1-D coef array).
        @test_throws ArgumentError LinearStencil{2}(SUnitRange(0, 1), arr)
        # SVector length must match L.
        @test_throws ArgumentError LinearStencil{1}(SUnitRange(0, 2), arr)  # L=3 vs SVector{2}
        # Non-SUnitRange offsets.
        @test_throws ArgumentError LinearStencil{1}(0:1, arr)
    end

    @testset "StarStencil construction (interlaced)" begin
        # 2-D L=1: M = 2NL+1 = 5; single SVector{5} per cell, diagonal mid-slot.
        arr = fill(SVector(-1.0, -1.0, 4.0, -1.0, -1.0), 5, 4)
        st = StarStencil{1}(arr)
        @test st isa StarStencil{1, 2, 5, SVector{5, Float64}, typeof(arr), ColumnAccess}
        @test AccessStyle(st) === ColumnAccess()
        @test st.term === arr

        # 3-D L=2: M = 2*3*2+1 = 13.
        arr3 = fill(SVector(ntuple(_ -> 1.0, 13)...), 4, 5, 6)
        st3 = StarStencil{2}(arr3)
        @test st3 isa StarStencil{2, 3, 13, SVector{13, Float64}, typeof(arr3), ColumnAccess}

        # Symbolic coefficient: grid rank derived from (L, M).
        sym = _DummyTerm{SVector{5, Float64}}()
        sst = StarStencil{1}(sym)
        @test sst isa StarStencil{1, 2, 5, SVector{5, Float64}, typeof(sym), ColumnAccess}

        # ndims must equal (M-1)/(2L): SVector{5} (N=2) on a 3-D array.
        @test_throws ArgumentError StarStencil{1}(fill(SVector(-1.0, -1.0, 4.0, -1.0, -1.0), 3, 4, 5))
        # M not of the form 2NL+1: SVector{4}, L=1 ⇒ (4-1)%2 ≠ 0.
        @test_throws ArgumentError StarStencil{1}(fill(SVector(1.0, 2.0, 3.0, 4.0), 5, 4))
    end

    @testset "Stencil construction + narrowing (SoA)" begin
        @test ô isa StaticShift{Tuple{}}

        # Linear pattern: single axis, contiguous offsets -2:0. Structure-of-
        # arrays: one scalar coefficient (array) per offset.
        shifts = (-2ê₁, -ê₁, ô)
        terms = (fill(1.0, 5), fill(-4.0, 5), fill(3.0, 5))
        st = Stencil(RowAccess, shifts, terms)
        @test st isa Stencil{3, typeof(shifts), typeof(terms), RowAccess}
        @test AccessStyle(st) === RowAccess()
        @test st.terms === terms
        ln = as_linear(st)
        @test ln isa LinearStencil{1, -2, 3, SVector{3, Float64}, <:Any, RowAccess}
        @test ln.offsets == SUnitRange(-2, 0)
        # narrowing interlaces SoA → AoS: an SVector{3} per cell
        @test ln.term == fill(SVector(1.0, -4.0, 3.0), 5)

        # Symbolic coefficients narrow too (via the _interlace extension point).
        syms = ntuple(_ -> _DummyTerm{Float64}(), 3)
        lns = as_linear(Stencil(ColumnAccess, shifts, syms))
        @test lns isa LinearStencil{1, -2, 3, SVector{3, Float64}, _DummyTerm{SVector{3, Float64}}, ColumnAccess}

        # Star pattern: 2-D L=1, reverse-lex (-ê₂, -ê₁, ô, ê₁, ê₂), M=5.
        sshifts = (-ê₂, -ê₁, ô, ê₁, ê₂)
        sterms = ntuple(i -> fill(Float64(i), 5, 4), 5)
        sst = Stencil(ColumnAccess, sshifts, sterms)
        ss = as_star(sst)
        @test ss isa StarStencil{1, 2, 5, SVector{5, Float64}, <:Any, ColumnAccess}
        @test ss.term == fill(SVector(1.0, 2.0, 3.0, 4.0, 5.0), 5, 4)

        # Rejections.
        @test_throws ArgumentError as_linear(sst)                       # spans axes ⇒ not linear
        @test_throws ArgumentError as_star(st)                          # 3 offsets ≠ 2NL+1
        # Non-contiguous single-axis ⇒ not linear.
        gap = Stencil(RowAccess, (-2ê₁, ô, 2ê₁), (fill(1.0, 5), fill(2.0, 5), fill(3.0, 5)))
        @test_throws ArgumentError as_linear(gap)
        # Shift count ≠ terms count ⇒ friendly ctor error.
        @test_throws ArgumentError Stencil(RowAccess, (-ê₁, ô), (fill(1.0, 5), fill(2.0, 5), fill(3.0, 5)))
    end

    @testset "AbstractScalar leaves + eltype" begin
        τ = Var{:τ, Float64}()
        @test τ isa AbstractScalar{Float64}
        @test eltype(τ) === Float64
        @test Var{:τ}() isa Var{:τ, Float64}                      # default T = Float64
        @test eltype(Var{:τ, Float32}()) === Float32

        # Constant: any concrete T; stores `val` as-is.
        @test Constant(2.0) isa Constant{Float64}
        @test Constant(3).val === 3
        @test eltype(Constant(SVector(1, 2))) === SVector{2, Int}
        @test Constant(SVector(1, 2)).val === SVector(1, 2)
        @test_throws ArgumentError Constant{Real}(1)              # T must be concrete

        # Unity: structural multiplicative one; inner ctor's `T` must admit
        # `one(T)` AND be Bool-shaped; outer ctors route through `_unity_space`
        # so a value-space type (Number, SVector) lands in its Bool-shaped
        # identity space.
        @test Unity{Bool}() isa AbstractScalar{Bool}
        @test eltype(Unity{Bool}()) === Bool
        @test Unity(Float64) === Unity{Bool}()
        @test Unity(3.14) === Unity{Bool}()
        @test Unity{SMatrix{2, 2, Bool, 4}}() isa Unity                    # square SMatrix OK
        # Outer ctor on SVector remaps to the square SMatrix Bool identity space.
        @test Unity(SVector{2, Float64}) === Unity{SMatrix{2, 2, Bool, 4}}()
        @test Unity(SVector(1.0, 2.0))   === Unity{SMatrix{2, 2, Bool, 4}}()
        # Inner ctor on SVector still rejects (no `one(SVector)`).
        @test_throws ArgumentError Unity{SVector{2, Float64}}()
        @test_throws ArgumentError Unity{Integer}()                        # T must be concrete
        @test_throws ArgumentError Unity{Float64}()                        # violates Bool invariant

        # Null: type and value ctors always produce Bool-shaped result.
        @test eltype(Null{Bool}()) === Bool
        @test Null(Float64) === Null{Bool}()
        @test Null(3.14) === Null{Bool}()
        @test Null(7) === Null{Bool}()

        # T must be concrete; non-Bool T is rejected.
        @test_throws ArgumentError Var{:s, Real}()
        @test_throws ArgumentError Null{Integer}()
        @test_throws ArgumentError Null{Float64}()                         # violates Bool invariant
    end

    @testset "@var macro" begin
        @var τ
        @var dt Float32
        @test τ === Var{:τ, Float64}()
        @test dt === Var{:dt, Float32}()
    end

    @testset "Scalar tree node + operator overloads" begin
        τ = Var{:τ, Float64}(); α = Constant(2)
        # Binary op among AbstractScalars builds a Scalar node.
        s = τ * α
        @test s isa Scalar{typeof(*)}
        @test eltype(s) === Float64                                          # promotes Float64 ↔ Int
        @test s.args === (τ, α)
        # Numeric literal canonicalises to Constant at the operator boundary.
        s2 = τ + 3
        @test s2 isa Scalar{typeof(+)}
        @test s2.args[2] === Constant(3)
        s3 = 4 - τ
        @test s3.args[1] === Constant(4)
        # Non-Number literal at the boundary: the bug-motivating SVector case.
        v = Var{:v, SVector{1, Int}}()
        sv = v + SVector(1)
        @test sv isa Scalar{typeof(+)}
        @test sv.args[2] === Constant(SVector(1))
        # Unary.
        @test (-τ) isa Scalar{typeof(-)}
        @test sin(τ) isa Scalar{typeof(sin)}

        # Union{}-result Scalars throw at construction.
        @test_throws ArgumentError Scalar(+, (Var{:s, String}(), Var{:n, Float64}()))
    end

    @testset "AbstractScalar show" begin
        τ = Var{:τ, Float64}()
        @test repr(τ) == "τ"
        @test repr(Constant(2.0)) == "2.0"
        @test repr(Constant(3)) == "3"
        @test repr(Null{Bool}()) == "0"             # type-agnostic glyph
        @test repr(Unity{Bool}()) == "U"            # type-agnostic glyph
        let J = SMatrix{2, 2, Bool, 4}
            @test repr(Constant(2) * Unity{J}()) == "2U"      # numeric juxtaposition
            @test repr(τ * Unity{J}()) == "τ * U"             # symbolic: explicit *, no parens
            @test repr(Unity{J}() * τ) == "U * τ"             # Unity on left
        end
        @test repr(τ * Constant(2.0)) == "(τ * 2.0)"   # infix
        @test repr(-τ) == "-τ"
        @test repr(Scalar(exp, (τ,))) == "exp(τ)"      # call form
    end

    @testset "AbstractScalar simplify" begin
        τ = Var{:τ, Float64}()
        N = Null{Bool}()
        U = Unity{Bool}()                   # the structural multiplicative one
        simp = StencilCore.simplify

        # Leaves are already normal form.
        @test simp(τ) === τ
        @test simp(Constant(2.0)) === Constant(2.0)
        @test simp(N) === N
        @test simp(U) === U

        # Identity / annihilator: purely structural — Null and Unity by type.
        @test simp(N + τ) === τ
        @test simp(τ + N) === τ
        @test simp(N - τ) === Scalar(-, (τ,))          # 0 - b = -b
        @test simp(τ - N) === τ
        @test simp(τ * N) === N
        @test simp(N * τ) === N
        @test simp(τ * U) === τ
        @test simp(U * τ) === τ
        @test simp(τ / U) === τ
        @test simp(N / τ) === N
        @test simp(-(-τ)) === τ                         # double negation

        # Numerical zeros / ones in Constant .val are NOT structural
        # identities: simplify leaves them as Scalar nodes (until/unless the
        # value is statically encoded).
        @test simp(τ * Constant(1.0)) isa Scalar
        @test simp(τ * Constant(0.0)) isa Scalar

        # Fold rule, Path 1 — coefficient fold. Number-only args fold to a
        # `Constant`.
        @test simp(Constant(2.0) + Constant(3.0)) === Constant(5.0)
        @test simp(Constant(2.0) * Constant(0.0)) === Constant(0.0)
        @test !(simp(Constant(2.0) * Constant(0.0)) isa Null)
        @test simp(Constant(6.0) / Constant(2.0)) === Constant(3.0)
        @test simp(Constant(2)^Constant(3)) === Constant(8)

        # Mixed: identity collapses, then fold collapses.
        @test simp((τ + N) * (Constant(2.0) + Constant(3.0))) == τ * Constant(5.0)
    end

    @testset "AbstractScalar materialize" begin
        τ = Var{:τ, Float64}()
        mat = StencilCore.materialize

        # Leaves.
        @test mat(Constant(2.5)) === 2.5
        @test mat(Constant(3)) === 3
        @test mat(Constant(SVector(1, 2))) === SVector(1, 2)
        @test mat(τ, (τ = 7.0,)) === 7.0
        @test mat(Null{Bool}()) === false
        @test mat(Unity{Bool}()) === true
        @test mat(Unity{SMatrix{2, 2, Bool, 4}}()) === one(SMatrix{2, 2, Bool, 4})

        # Scalar tree.
        @test mat(τ * Constant(3.0), (τ = 4.0,)) === 12.0
        @test mat(τ + Constant(1.0), (τ = 2.0,)) === 3.0
        @test mat(τ * τ + τ, (τ = 5.0,)) === 30.0
        # _scalar_body_expr round-trips a representative tree.
        e = StencilCore._scalar_body_expr(τ * Constant(3.0))
        @test e isa Expr
        let args = (τ = 4.0,)
            @test Core.eval(@__MODULE__, :(let args = $(args); $e end)) === 12.0
        end
    end

    @testset "AbstractScalar differentiate" begin
        τ = Var{:τ, Float64}(); α = Constant(2.0)
        η = Var{:η, Float64}()
        diff = StencilCore.differentiate
        U = Unity{Bool}()

        # Leaves.
        @test diff(τ, τ) === U
        @test diff(Constant(2.0), τ) === Null{Bool}()
        @test diff(η, τ) === Null{Bool}()
        @test diff(Null{Bool}(), τ) === Null{Bool}()
        @test diff(Unity{Bool}(), τ) === Null{Bool}()

        # Sum rule: ∂(τ + α)/∂τ = 1.
        @test diff(τ + α, τ) === U
        # Product rule: ∂(α * τ)/∂τ = α.
        @test diff(α * τ, τ) === α
        @test diff(τ * α, τ) === α
        # ∂(τ²)/∂τ = τ + τ  (no like-term folding).
        @test diff(τ * τ, τ) == Scalar(+, (τ, τ))

        # Chain rule via primitive: ∂sin(τ)/∂τ = cos(τ).
        @test diff(sin(τ), τ) == Scalar(cos, (τ,))
        # ∂exp(τ)/∂τ = exp(τ).
        @test diff(exp(τ), τ) == Scalar(exp, (τ,))

        # No dependence ⇒ Null{Bool} (Bool-shaped, precision resolved by promotion).
        @test diff(α + η, τ) === Null{Bool}()

        # Mixed-eltype: promote across both operands — result is still Null{Bool}.
        @test diff(Constant(2), τ) === Null{Bool}()          # Int leaf, Float64 variable
        @test diff(Null{Bool}(), τ) === Null{Bool}()

        # tan is now supported: ∂tan(τ)/∂τ = 1 + tan(τ)².
        @test diff(sin(τ), τ) == Scalar(cos, (τ,))              # already tested above; confirm consistency
        let d = diff(tan(τ), τ)
            @test StencilCore.materialize(d, (τ = π/4,)) ≈ 2.0  # sec²(π/4) = 2
        end

        # No rule for an arbitrary primitive ⇒ throws when the chain rule fires.
        struct _NoRuleFn end
        (::_NoRuleFn)(x) = x
        norf = _NoRuleFn()
        @test_throws ArgumentError diff(Scalar(norf, (τ,)), τ)
    end

    @testset "AbstractScalar differentiate (SVector / Jacobian)" begin
        diff = StencilCore.differentiate
        mat  = StencilCore.materialize
        x = Var{:x, SVector{2, Float64}}()
        y = Var{:y, SVector{2, Float64}}()
        τ = Var{:τ, Float64}()

        # ∂(2x)/∂x: the product rule returns Constant(2) * Unity{J}(). The
        # fold does not fire for non-Number eltypes, so the result stays as a
        # Scalar tree. Materialization equals 2I (as SMatrix{Int}).
        let J = SMatrix{2, 2, Bool, 4}
            d = diff(2x, x)
            @test d == Scalar(*, (Constant(2), Unity{J}()))
            @test mat(d) === SMatrix{2, 2, Int, 4}(2, 0, 0, 2)
        end

        # Self-derivative of an SVector symbol is the structural Unity in the
        # Jacobian (square SMatrix Bool) shape.
        let J = SMatrix{3, 3, Bool, 9}
            x3 = Var{:x3, SVector{3, Float32}}()
            @test diff(x3, x3) === Unity{J}()
            @test mat(diff(x3, x3)) === one(J)
        end

        # Sum rule across vector terms: ∂(2x + 3x)/∂x materializes to 5I (Int).
        let J = SMatrix{2, 2, Int, 4}
            @test mat(diff(2x + 3x, x)) === J(5, 0, 0, 5)
        end

        # Independence: ∂y/∂x = Null{Bool-J}.
        let J = SMatrix{2, 2, Bool, 4}
            @test diff(y, x) === Null{J}()
            @test diff(Constant(SVector(1.0, 0.0)), x) === Null{J}()
        end

        # Top-level shape-class mismatch ⇒ ArgumentError.
        @test_throws ArgumentError diff(τ, x)
        @test_throws ArgumentError diff(x, τ)
        # Same-class but mismatched N ⇒ also rejected.
        let x4 = Var{:x4, SVector{4, Float64}}()
            @test_throws ArgumentError diff(x, x4)
        end
    end

    @testset "AbstractScalar simplify (shape-aware)" begin
        simp = StencilCore.simplify
        x = Var{:x, SVector{2, Float64}}()
        J  = SMatrix{2, 2, Bool, 4}

        # Fold Path 1 restricted to Number eltypes: `Constant(Number) *
        # Unity{SMatrix}()` does NOT fold (non-Number result) — stays as Scalar.
        @test simp(Constant(2.0) * Unity{J}()) isa Scalar
        @test simp(Unity{J}() * Constant(2.0)) isa Scalar

        # But Number × Number still folds (Unity{Bool} acts as true):
        @test simp(Constant(2.0) + Unity{Bool}()) === Constant{Float64}(3.0)

        # Fold Path 2 — direct fold over non-Number Constants (the user's
        # `v + SVector(1)` bug case).
        let s = simp(Constant(SVector(1, 0)) + Constant(SVector(0, 1)))
            @test s === Constant{SVector{2, Int}}(SVector(1, 1))
        end

        # Identity rule eltype-preservation gate: `Unity{SMatrix} * x`
        # where `x::AbstractScalar{SVector}` has parent eltype SVector and
        # `eltype(x)` SVector — they match, so the rule fires and returns x.
        @test simp(Unity{J}() * x) === x

        # `Unity{SMatrix} * Constant{Int}`: eltype gate fails (SMatrix ≠ Int),
        # identity rule does not fire. Path 1 fold also does not fire (non-Number
        # result). Stays as a Scalar tree.
        @test simp(Unity{J}() * Constant(3)) isa Scalar

        # `2 * x` → simplifies to Scalar(*, (Constant(2), x)) (no fold for
        # non-Number SVector eltype).
        let T = SVector{2, Float64}, x2 = Var{:x2, T}()
            collapsed = Scalar(*, (Constant(2), x2))
            @test simp(2 * x2) === collapsed
        end

        # Cross-precision SMatrix × Unity: same-size square matrices with
        # different element types — `one(J::Float64)` is still the identity for
        # `A::SMatrix{Int}`, so the rule fires and preserves the narrower type.
        let A = 2 * one(SMatrix{2, 2, Int, 4})
            @test simp(Constant(A) * Unity{J}()) === Constant(A)   # right-multiply
            @test simp(Unity{J}() * Constant(A)) === Constant(A)   # left-multiply
            @test simp(Constant(A) / Unity{J}()) === Constant(A)   # right-divide
        end

        # Scalar Int * Unity{SMatrix}: shape mismatch — Int ≠ 2×2 matrix —
        # the identity rule does not fire.  (Distinct from the existing Float64
        # test above; both must stay as Scalar.)
        @test simp(Constant(2) * Unity{J}()) isa Scalar
    end

    @testset "AbstractScalar AbstractTrees plumbing" begin
        τ = Var{:τ, Float64}()
        @test AbstractTrees.nodevalue(τ) === (:τ, Float64)
        @test AbstractTrees.children(τ) === ()
        @test AbstractTrees.nodevalue(Constant(2.5)) === 2.5
        @test AbstractTrees.children(Constant(2.5)) === ()
        @test AbstractTrees.nodevalue(Null{Bool}()) === false
        @test AbstractTrees.nodevalue(Unity{Bool}()) === true
        @test AbstractTrees.children(Unity{Bool}()) === ()
        s = τ * Constant(2.0)
        @test AbstractTrees.nodevalue(s) === *
        @test AbstractTrees.children(s) === s.args
    end

    @testset "AbstractScalar getindex" begin
        @var x SVector{2, Float64}

        # 1-D: IndexLinear and IndexCartesian (N=1) both produce the same node;
        # Julia prefers Cartesian (more specific first arg) — result is identical.
        s = x[1]
        @test s isa Scalar{typeof(getindex)}
        @test eltype(s) === Float64
        @test repr(s) == "x[1]"
        @test materialize(s, (x = SVector(3.0, 4.0),)) === 3.0
        @test materialize(x[2], (x = SVector(3.0, 4.0),)) === 4.0

        # Symbolic index (AbstractScalar{Int}): Constant(i) == i (identity)
        @var i Int
        @test Constant(i) === i                          # idempotent lift
        si = x[i]
        @test si isa Scalar{typeof(getindex)}
        @test eltype(si) === Float64
        @test repr(si) == "x[i]"
        @test materialize(si, (x = SVector(3.0, 4.0), i = 2)) === 4.0

        # Matrix-valued scalar — IndexLinear (single Int, flat column-major)
        @var M SMatrix{2, 3, Float64, 6}
        sl = M[3]
        @test sl isa Scalar{typeof(getindex)}
        @test eltype(sl) === Float64
        @test repr(sl) == "M[3]"
        Mval = SMatrix{2,3}(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        @test materialize(sl, (M = Mval,)) === 3.0   # column-major: [1,2,3,4,5,6]

        # Matrix-valued scalar — IndexCartesian (two Ints)
        sc = M[1, 2]
        @test sc isa Scalar{typeof(getindex)}
        @test eltype(sc) === Float64
        @test repr(sc) == "M[1, 2]"
        @test materialize(sc, (M = Mval,)) === 3.0   # row 1, col 2 = index 3

        # Matrix-valued scalar — IndexCartesian with symbolic index
        sc_sym = M[1, i]
        @test sc_sym isa Scalar{typeof(getindex)}
        @test eltype(sc_sym) === Float64
        @test repr(sc_sym) == "M[1, i]"
        @test materialize(sc_sym, (M = Mval, i = 2)) === 3.0

        # Scalar-valued var does NOT support getindex (correct MethodError)
        @var τ Float64
        @test_throws MethodError τ[1]
    end

end
