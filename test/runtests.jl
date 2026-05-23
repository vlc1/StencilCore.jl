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

        @test eltype(Const(2.0)) === Float64
        @test Const(3).val === 3
        @test Const{Int}(7).val === 7

        @test eltype(Null{Float64}()) === Float64
        @test eltype(Unity{Int}()) === Int

        # T must be concrete.
        @test_throws ArgumentError Symbolic{:s, Real}()
        @test_throws ArgumentError Const{Number}(1)
        @test_throws ArgumentError Null{Integer}()
        @test_throws ArgumentError Unity{Number}()
    end

    @testset "@symbolic / @const macros" begin
        @symbolic τ
        @symbolic dt Float32
        @const α 1
        @const β 2.5
        @test τ === Symbolic{:τ, Float64}()
        @test dt === Symbolic{:dt, Float32}()
        @test α === Const(1)
        @test β === Const(2.5)
    end

    @testset "Scalar tree node + operator overloads" begin
        τ = Symbolic{:τ, Float64}(); α = Const(2)
        # Binary op among AbstractScalars builds a Scalar node.
        s = τ * α
        @test s isa Scalar{typeof(*)}
        @test eltype(s) === Float64                                          # promotes Float64 ↔ Int
        @test s.args === (τ, α)
        # Numeric literal canonicalises to Const at the operator boundary.
        s2 = τ + 3
        @test s2 isa Scalar{typeof(+)}
        @test s2.args[2] === Const(3)
        s3 = 4 - τ
        @test s3.args[1] === Const(4)
        # Unary.
        @test (-τ) isa Scalar{typeof(-)}
        @test sin(τ) isa Scalar{typeof(sin)}

        # Union{}-result Scalars throw at construction.
        @test_throws ArgumentError Scalar(+, (Symbolic{:s, String}(), Symbolic{:n, Float64}()))
    end

    @testset "AbstractScalar shift-invariance" begin
        τ = Symbolic{:τ, Float64}(); α = Const(2.0); n = Null{Float64}(); u = Unity{Float64}()
        @test τ[] === τ
        @test τ[ê₁] === τ
        @test α[3ê₁ + ê₂] === α
        @test n[ê₁] === n
        @test u[ô] === u
        # Scalar tree node is also a scalar; it is shift-invariant via the
        # AbstractScalar method.
        s = τ * α
        @test s[ê₁] === s
    end

    @testset "AbstractScalar show" begin
        τ = Symbolic{:τ, Float64}()
        @test repr(τ) == "τ"
        @test repr(Const(2.0)) == "2.0"
        @test repr(Const(3)) == "3"
        @test repr(Null{Float64}()) == "0"            # type-agnostic glyph
        @test repr(Unity{Float64}()) == "1"
        @test repr(τ * Const(2.0)) == "(τ * 2.0)"     # infix
        @test repr(-τ) == "-τ"
        @test repr(Scalar(exp, (τ,))) == "exp(τ)"     # call form
    end

    @testset "AbstractScalar AbstractTrees plumbing" begin
        τ = Symbolic{:τ, Float64}()
        @test AbstractTrees.nodevalue(τ) === :τ
        @test AbstractTrees.children(τ) === ()
        @test AbstractTrees.nodevalue(Const(2.5)) === 2.5
        @test AbstractTrees.children(Const(2.5)) === ()
        @test AbstractTrees.nodevalue(Null{Float64}()) === 0.0
        @test AbstractTrees.nodevalue(Unity{Int}()) === 1
        s = τ * Const(2.0)
        @test AbstractTrees.nodevalue(s) === *
        @test AbstractTrees.children(s) === s.args
    end

end
