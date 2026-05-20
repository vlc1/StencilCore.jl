using StencilCore
using Test
using StaticArrays: SUnitRange, SVector

# Structs must be defined at top level (not inside @testset scopes).
struct _DummyStencil{S} <: AbstractStencil{S} end
struct _DummyTerm{T} <: AbstractTerm{T} end

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

    @testset "StarStencil construction" begin
        arrs = (fill(SVector(-1.0, 2.0, -1.0), 5, 4), fill(SVector(-1.0, 2.0, -1.0), 5, 4))
        st = StarStencil{1}(arrs)
        @test st isa StarStencil{1, 2, 3, SVector{3, Float64}, typeof(arrs), ColumnAccess}
        @test AccessStyle(st) === ColumnAccess()

        # Symbolic per-axis coefficients.
        syms = (_DummyTerm{SVector{3, Float64}}(), _DummyTerm{SVector{3, Float64}}())
        sst = StarStencil{1}(syms)
        @test sst isa StarStencil{1, 2, 3, SVector{3, Float64}, typeof(syms), ColumnAccess}

        # M = 2L+1 mismatch (SVector{2} with L=1 expects SVector{3}).
        bad = (fill(SVector(-1.0, 1.0), 5, 4), fill(SVector(-1.0, 1.0), 5, 4))
        @test_throws ArgumentError StarStencil{1}(bad)
    end

end
