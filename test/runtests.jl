using StencilCore
using Test
using AbstractTrees
using StaticArrays: SUnitRange, SVector, SMatrix

# Structs must be defined at top level (not inside @testset scopes).
struct _DummyStencil{S} <: AbstractStencil{S} end
struct _DummyTerm{T} <: AbstractTerm{T} end

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

    @testset "AbstractTerm / ArrayOrTermLike" begin
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
        τ = Symbolic{:τ, Float64}()
        @test τ isa AbstractScalar{Float64}
        @test eltype(τ) === Float64
        @test Symbolic{:τ}() isa Symbolic{:τ, Float64}                      # default T = Float64
        @test eltype(Symbolic{:τ, Float32}()) === Float32

        # Scaling{V,T}: V<:Number; T = Traw when V<:eltype(Traw), else widens.
        @test Scaling{Int}(7).val === 7
        @test Λ === Scaling

        # V wider than eltype(T_raw) ⇒ T promotes (no throw).
        @test Scaling{Float64}(1)   isa Scaling{Float64, Int}
        @test Scaling{Float32}(1.0) isa Scaling{Float64, Float64}
        @test eltype(Scaling{Float32}(1.0)) === Float64

        # SArray shape: Scaling{SMatrix{2,2,Float64}}(2). The raw `T` (missing
        # the `L=4` parameter) is canonicalised into the fully-resolved
        # `SMatrix{2,2,Float64,4}`.
        let T2 = SMatrix{2, 2, Float64, 4}
            sc = Scaling{SMatrix{2, 2, Float64}}(2)
            @test sc isa Scaling{T2, Int}
            @test eltype(sc) === T2
        end

        # Value-space outer ctors: Scaling(T) and Scaling(T, val). `T` is
        # routed through `_unity_space` so an SVector lands in its Jacobian
        # (square SMatrix) space. `val` is stored as-is (V = typeof(val)).
        @test Scaling(Float64)      === Scaling{Float64, Float64}(1.0)
        @test Scaling(Float64, 2)   === Scaling{Float64, Int}(2)
        @test Scaling(Float64, 2.0) === Scaling{Float64, Float64}(2.0)
        let M3 = SMatrix{3, 3, Float64, 9}
            @test Scaling(SVector{3, Float64})    === Scaling{M3, Float64}(1.0)
            @test Scaling(SVector{3, Float64}, 2) === Scaling{M3, Int}(2)
        end

        # Symbol-anchored ctors delegate to the value-space form.
        let τT = Symbolic{:τ, Float64}
            @test Scaling(τT)      === Scaling(Float64)
            @test Scaling(τT, 2.0) === Scaling(Float64, 2.0)
        end
        let xT = Symbolic{:x, SVector{2, Float64}}, M2 = SMatrix{2, 2, Float64, 4}
            @test Scaling(xT)    === Scaling{M2, Float64}(1.0)
            @test Scaling(xT, 3) === Scaling{M2, Int}(3)
        end

        # Constant: any concrete T; stores `val` as-is.
        @test Constant(2.0) isa Constant{Float64}
        @test Constant(3).val === 3
        @test eltype(Constant(SVector(1, 2))) === SVector{2, Int}
        @test Constant(SVector(1, 2)).val === SVector(1, 2)
        @test_throws ArgumentError Constant{Real}(1)              # T must be concrete

        # Unity: structural multiplicative one; inner ctor's `T` must admit
        # `one(T)`; outer ctors route through `_unity_space` so a value-space
        # type (Number, SVector) lands in its identity space.
        @test Unity{Float64}() isa AbstractScalar{Float64}
        @test eltype(Unity{Float64}()) === Float64
        @test Unity(Float64) === Unity{Float64}()
        @test Unity(3.14) === Unity{Float64}()
        @test Unity{SMatrix{2, 2, Float64, 4}}() isa Unity                 # square SMatrix OK
        # Outer ctor on SVector remaps to the square SMatrix identity space.
        @test Unity(SVector{2, Float64}) === Unity{SMatrix{2, 2, Float64, 4}}()
        @test Unity(SVector(1.0, 2.0))   === Unity{SMatrix{2, 2, Float64, 4}}()
        # Inner ctor on SVector still rejects (no `one(SVector)`).
        @test_throws ArgumentError Unity{SVector{2, Float64}}()
        @test_throws ArgumentError Unity{Integer}()                        # T must be concrete

        # Null: type and value ctors.
        @test eltype(Null{Float64}()) === Float64
        @test Null(Float64) === Null{Float64}()
        @test Null(3.14) === Null{Float64}()
        @test Null(7) === Null{Int}()

        # T must be concrete.
        @test_throws ArgumentError Symbolic{:s, Real}()
        @test_throws ArgumentError Scaling{Number}(1)
        @test_throws ArgumentError Null{Integer}()
    end

    @testset "@symbolic macro" begin
        @symbolic τ
        @symbolic dt Float32
        @test τ === Symbolic{:τ, Float64}()
        @test dt === Symbolic{:dt, Float32}()
    end

    @testset "Scalar tree node + operator overloads" begin
        τ = Symbolic{:τ, Float64}(); α = Constant(2)
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
        v = Symbolic{:v, SVector{1, Int}}()
        sv = v + SVector(1)
        @test sv isa Scalar{typeof(+)}
        @test sv.args[2] === Constant(SVector(1))
        # Unary.
        @test (-τ) isa Scalar{typeof(-)}
        @test sin(τ) isa Scalar{typeof(sin)}

        # Union{}-result Scalars throw at construction.
        @test_throws ArgumentError Scalar(+, (Symbolic{:s, String}(), Symbolic{:n, Float64}()))
    end

    @testset "AbstractScalar show" begin
        τ = Symbolic{:τ, Float64}()
        @test repr(τ) == "τ"
        @test repr(Constant(2.0)) == "2.0"
        @test repr(Constant(3)) == "3"
        @test repr(Null{Float64}()) == "0"             # type-agnostic glyph
        @test repr(Unity{Float64}()) == "1"            # type-agnostic glyph
        @test repr(Scaling{Float64}(1.0)) == "1.0"     # Scaling prints its stored val
        @test repr(τ * Constant(2.0)) == "(τ * 2.0)"   # infix
        @test repr(-τ) == "-τ"
        @test repr(Scalar(exp, (τ,))) == "exp(τ)"      # call form
    end

    @testset "AbstractScalar simplify" begin
        τ = Symbolic{:τ, Float64}()
        N = Null{Float64}()
        U = Unity{Float64}()                   # the structural multiplicative one
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

        # Numerical zeros / ones in Constant or Scaling .val are NOT structural
        # identities: simplify leaves them as Scalar nodes (until/unless the
        # value is statically encoded).
        @test simp(τ * Constant(1.0)) isa Scalar
        @test simp(τ * Scaling{Float64}(1.0)) isa Scalar
        @test simp(τ * Constant(0.0)) isa Scalar

        # Fold rule, Path 1 — coefficient fold. Number-only args fold to a
        # `Constant`; mixed Number-coefficient args fold to a `Scaling` when
        # the parent eltype is non-Number.
        @test simp(Constant(2.0) + Constant(3.0)) === Constant(5.0)
        @test simp(Constant(2.0) * Constant(0.0)) === Constant(0.0)
        @test !(simp(Constant(2.0) * Constant(0.0)) isa Null)
        @test simp(Constant(6.0) / Constant(2.0)) === Constant(3.0)
        @test simp(Constant(2)^Constant(3)) === Constant(8)

        # Mixed: identity collapses, then fold collapses.
        @test simp((τ + N) * (Constant(2.0) + Constant(3.0))) == τ * Constant(5.0)
    end

    @testset "AbstractScalar materialize" begin
        τ = Symbolic{:τ, Float64}()
        mat = StencilCore.materialize

        # Leaves.
        @test mat(Constant(2.5)) === 2.5
        @test mat(Constant(3)) === 3
        @test mat(Constant(SVector(1, 2))) === SVector(1, 2)
        @test mat(τ, (τ = 7.0,)) === 7.0
        @test mat(Null{Float64}()) === 0.0
        @test mat(Unity{Float64}()) === 1.0
        @test mat(Unity{SMatrix{2, 2, Float64, 4}}()) === SMatrix{2, 2, Float64}(1, 0, 0, 1)
        @test mat(Scaling{Float64}(1.0)) === 1.0
        # SMatrix-shaped Scaling materializes as `val * I`.
        @test mat(Scaling{SMatrix{2, 2, Float64}}(2)) === SMatrix{2, 2, Float64}(2, 0, 0, 2)

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
        τ = Symbolic{:τ, Float64}(); α = Constant(2.0)
        η = Symbolic{:η, Float64}()
        diff = StencilCore.differentiate
        U = Unity{Float64}()

        # Leaves.
        @test diff(τ, τ) === U
        @test diff(Constant(2.0), τ) === Null{Float64}()
        @test diff(η, τ) === Null{Float64}()
        @test diff(Null{Float64}(), τ) === Null{Float64}()
        @test diff(Unity{Float64}(), τ) === Null{Float64}()

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

        # No dependence ⇒ Null with the right eltype.
        @test diff(α + η, τ) === Null{Float64}()

        # Mixed-eltype: promote across both operands.
        @test diff(Constant(2), τ) === Null{Float64}()          # Int leaf, Float64 variable
        @test diff(Null{Int}(), τ) === Null{Float64}()

        # No rule for an arbitrary primitive ⇒ throws when the chain rule fires.
        @test_throws ArgumentError diff(Scalar(tan, (τ,)), τ)
    end

    @testset "AbstractScalar differentiate (SVector / Jacobian)" begin
        diff = StencilCore.differentiate
        mat  = StencilCore.materialize
        x = Symbolic{:x, SVector{2, Float64}}()
        y = Symbolic{:y, SVector{2, Float64}}()
        τ = Symbolic{:τ, Float64}()

        # User's motivating example: ∂(2x)/∂x = 2I. The `2` is stored as Int
        # (its input type), not widened to Float64 — Unity's coefficient is
        # the Bool identity `true`, so `2 * true === 2::Int` preserves V.
        let J = SMatrix{2, 2, Float64, 4}
            @test diff(2x, x) === Scaling{J}(2)
            @test diff(2x, x) isa Scaling{J, Int}
            @test mat(diff(2x, x)) === J(2.0, 0.0, 0.0, 2.0)
        end

        # Self-derivative of an SVector symbol is the structural Unity in the
        # Jacobian (square SMatrix) shape.
        let J = SMatrix{3, 3, Float32, 9}
            x3 = Symbolic{:x3, SVector{3, Float32}}()
            @test diff(x3, x3) === Unity{J}()
            @test mat(diff(x3, x3)) === J(1, 0, 0, 0, 1, 0, 0, 0, 1)
        end

        # Sum rule across vector terms: ∂(2x + 3x)/∂x = 5I. V stays Int.
        let J = SMatrix{2, 2, Float64, 4}
            @test diff(2x + 3x, x) === Scaling{J}(5)
        end

        # Independence: ∂y/∂x = Null{J}.
        let J = SMatrix{2, 2, Float64, 4}
            @test diff(y, x) === Null{J}()
            @test diff(Scaling{SVector{2, Float64}}(0.0), x) === Null{J}()
        end

        # Top-level shape-class mismatch ⇒ ArgumentError.
        @test_throws ArgumentError diff(τ, x)
        @test_throws ArgumentError diff(x, τ)
        # Same-class but mismatched N ⇒ also rejected.
        let x4 = Symbolic{:x4, SVector{4, Float64}}()
            @test_throws ArgumentError diff(x, x4)
        end
    end

    @testset "AbstractScalar simplify (shape-aware)" begin
        simp = StencilCore.simplify
        x = Symbolic{:x, SVector{2, Float64}}()
        J  = SMatrix{2, 2, Float64, 4}

        # Fold Path 1 — coefficient fold preserves matrix shape.
        @test simp(Scaling{J}(2.0) + Scaling{J}(3.0)) === Scaling{J}(5.0)

        # Fold Path 1 mixed: `Constant(Number) * Unity{SMatrix}()` compacts to
        # `Scaling{SMatrix}(num)` — the differentiation pipeline's clean form.
        @test simp(Constant(2.0) * Unity{J}()) === Scaling{J}(2.0)
        @test simp(Unity{J}() * Constant(2.0)) === Scaling{J}(2.0)
        @test simp(Constant(2.0) + Unity{Float64}()) === Constant{Float64}(3.0)

        # Fold Path 2 — direct fold over non-Number Constants (the user's
        # `v + SVector(1)` bug case).
        let s = simp(Constant(SVector(1, 0)) + Constant(SVector(0, 1)))
            @test s === Constant{SVector{2, Int}}(SVector(1, 1))
        end

        # Identity rule has an eltype-preservation gate: `Unity{SMatrix} * x`
        # where `x::AbstractScalar{SVector}` has parent eltype SVector and
        # `eltype(x)` SVector — they match, so the rule fires and returns x.
        @test simp(Unity{J}() * x) === x

        # But `Unity{SMatrix} * Constant{Int}` does NOT collapse (parent
        # eltype = SMatrix, other operand eltype = Int): the rule must avoid
        # silently changing eltype. Falls through to the fold instead, with
        # the Int preserved (V = Int, not widened to Float64).
        @test simp(Unity{J}() * Constant(3)) === Scaling{J}(3)
        @test simp(Unity{J}() * Constant(3)) isa Scaling{J, Int}

        # Scaling/Constant canonicalisation: `Scaling{matrix-shape}(c) * y`
        # collapses to `Constant(c) * y` when the substitution preserves
        # eltype. Both `Scaling(T, 2) * x` and `2x` reach the same canonical
        # form.
        let T = SVector{2, Float64}, x2 = Symbolic{:x2, T}()
            collapsed = Scalar(*, (Constant(2), x2))
            @test simp(Scaling(T, 2) * x2) === collapsed
            @test simp(2 * x2)             === collapsed
            @test simp(Scaling(T, 2) * x2) === simp(2 * x2)
        end

        # Number-shape Scaling collapses too when eltype-safe.
        let τ = Symbolic{:τ, Float64}()
            @test simp(Scaling{Float64}(2.0) * τ) === Scalar(*, (Constant(2.0), τ))
            @test simp(Scaling{Float64}(2.0) / τ) === Scalar(/, (Constant(2.0), τ))
        end

        # Eltype-preservation gate: `Scaling{SMatrix} * Constant{Int}` does
        # NOT collapse via this rule (Int times Int loses the matrix eltype).
        # Path 1 fold handles it instead, preserving the matrix shape.
        @test simp(Scaling{J}(2) * Constant(3)) === Scaling{J}(6)
    end

    @testset "AbstractScalar AbstractTrees plumbing" begin
        τ = Symbolic{:τ, Float64}()
        @test AbstractTrees.nodevalue(τ) === (:τ, Float64)
        @test AbstractTrees.children(τ) === ()
        @test AbstractTrees.nodevalue(Constant(2.5)) === 2.5
        @test AbstractTrees.children(Constant(2.5)) === ()
        @test AbstractTrees.nodevalue(Null{Float64}()) === 0.0
        @test AbstractTrees.nodevalue(Unity{Float64}()) === 1.0
        @test AbstractTrees.children(Unity{Float64}()) === ()
        @test AbstractTrees.nodevalue(Scaling{Int}(1)) === 1
        s = τ * Constant(2.0)
        @test AbstractTrees.nodevalue(s) === *
        @test AbstractTrees.children(s) === s.args
    end

end
