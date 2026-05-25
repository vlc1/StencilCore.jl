using StencilCore
using Test
using AbstractTrees
using StaticArrays: SUnitRange, SVector

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

        # Scaling: V<:eltype(T), Λ alias, Scaling(T)→one(T).
        @test eltype(Scaling(2.0)) === Float64
        @test Scaling(3).val === 3
        @test Scaling{Int}(7).val === 7
        @test Λ === Scaling
        @test Scaling() === Scaling{Float64}(1.0)
        @test Scaling(Float64) === Scaling{Float64}(1.0)
        @test Scaling(Float32).val === 1.0f0

        # V <: eltype(T) constraint.
        @test_throws ArgumentError Scaling{Float64}(1)        # Int ⊄ Float64
        @test_throws ArgumentError Scaling{Float32}(1.0)      # Float64 ⊄ Float32

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
        τ = Symbolic{:τ, Float64}(); α = Scaling(2)
        # Binary op among AbstractScalars builds a Scalar node.
        s = τ * α
        @test s isa Scalar{typeof(*)}
        @test eltype(s) === Float64                                          # promotes Float64 ↔ Int
        @test s.args === (τ, α)
        # Numeric literal canonicalises to Scaling at the operator boundary.
        s2 = τ + 3
        @test s2 isa Scalar{typeof(+)}
        @test s2.args[2] === Scaling(3)
        s3 = 4 - τ
        @test s3.args[1] === Scaling(4)
        # Unary.
        @test (-τ) isa Scalar{typeof(-)}
        @test sin(τ) isa Scalar{typeof(sin)}

        # Union{}-result Scalars throw at construction.
        @test_throws ArgumentError Scalar(+, (Symbolic{:s, String}(), Symbolic{:n, Float64}()))
    end

    @testset "AbstractScalar show" begin
        τ = Symbolic{:τ, Float64}()
        @test repr(τ) == "τ"
        @test repr(Scaling(2.0)) == "2.0"
        @test repr(Scaling(3)) == "3"
        @test repr(Null{Float64}()) == "0"             # type-agnostic glyph
        @test repr(Scaling{Float64}(1.0)) == "1.0"     # the new "unit Scaling" prints its val
        @test repr(τ * Scaling(2.0)) == "(τ * 2.0)"    # infix
        @test repr(-τ) == "-τ"
        @test repr(Scalar(exp, (τ,))) == "exp(τ)"      # call form
    end

    @testset "AbstractScalar simplify" begin
        τ = Symbolic{:τ, Float64}()
        N = Null{Float64}()
        U = Scaling{Float64}(1.0)              # the new multiplicative identity (by value)
        simp = StencilCore.simplify

        # Leaves are already normal form.
        @test simp(τ) === τ
        @test simp(Scaling(2.0)) === Scaling(2.0)
        @test simp(N) === N
        @test simp(U) === U

        # Identity / annihilator: Null by type, Scaling(1) by value.
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

        # Fold rule: all-Scaling args fold to a Scaling carrying the result value.
        @test simp(Scaling(2.0) + Scaling(3.0)) === Scaling(5.0)
        @test simp(Scaling(2.0) * Scaling(0.0)) === Scaling(0.0)
        @test !(simp(Scaling(2.0) * Scaling(0.0)) isa Null)
        @test simp(Scaling(6.0) / Scaling(2.0)) === Scaling(3.0)
        @test simp(Scaling(2)^Scaling(3)) === Scaling(8)

        # Mixed: identity collapses, then fold collapses.
        @test simp((τ + N) * (Scaling(2.0) + Scaling(3.0))) == τ * Scaling(5.0)
    end

    @testset "AbstractScalar materialize" begin
        τ = Symbolic{:τ, Float64}()
        mat = StencilCore.materialize

        # Leaves.
        @test mat(Scaling(2.5)) === 2.5                 # V=T=Float64: 2.5 * one(Float64)
        @test mat(Scaling(3)) === 3                     # V=T=Int:     3   * one(Int)
        @test mat(Scaling(Float64)) === 1.0             # val=one(T)
        @test mat(τ, (τ = 7.0,)) === 7.0
        @test mat(Null{Float64}()) === 0.0
        @test mat(Scaling{Float64}(1.0)) === 1.0

        # Scalar tree.
        @test mat(τ * Scaling(3.0), (τ = 4.0,)) === 12.0
        @test mat(τ + Scaling(1.0), (τ = 2.0,)) === 3.0
        @test mat(τ * τ + τ, (τ = 5.0,)) === 30.0
        # _scalar_body_expr round-trips a representative tree.
        e = StencilCore._scalar_body_expr(τ * Scaling(3.0))
        @test e isa Expr
        let args = (τ = 4.0,)
            @test Core.eval(@__MODULE__, :(let args = $(args); $e end)) === 12.0
        end
    end

    @testset "AbstractScalar differentiate" begin
        τ = Symbolic{:τ, Float64}(); α = Scaling(2.0)
        η = Symbolic{:η, Float64}()
        diff = StencilCore.differentiate
        U = Scaling{Float64}(1.0)

        # Leaves.
        @test diff(τ, τ) === U
        @test diff(Scaling(2.0), τ) === Null{Float64}()
        @test diff(η, τ) === Null{Float64}()
        @test diff(Null{Float64}(), τ) === Null{Float64}()

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
        @test diff(Scaling(2), τ) === Null{Float64}()           # V=Int promoted with Float64
        @test diff(Null{Int}(), τ) === Null{Float64}()

        # No rule for an arbitrary primitive ⇒ throws when the chain rule fires.
        @test_throws ArgumentError diff(Scalar(tan, (τ,)), τ)
    end

    @testset "AbstractScalar AbstractTrees plumbing" begin
        τ = Symbolic{:τ, Float64}()
        @test AbstractTrees.nodevalue(τ) === (:τ, Float64)
        @test AbstractTrees.children(τ) === ()
        @test AbstractTrees.nodevalue(Scaling(2.5)) === 2.5
        @test AbstractTrees.children(Scaling(2.5)) === ()
        @test AbstractTrees.nodevalue(Null{Float64}()) === 0.0
        @test AbstractTrees.nodevalue(Scaling{Int}(1)) === 1
        s = τ * Scaling(2.0)
        @test AbstractTrees.nodevalue(s) === *
        @test AbstractTrees.children(s) === s.args
    end

end
