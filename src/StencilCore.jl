module StencilCore

using AbstractTrees
using StaticArrays: SUnitRange, SVector

include("access.jl")
include("term.jl")
include("staticshift.jl")
include("scalars.jl")
include("scalar_trees.jl")
include("scalar_simplify.jl")
include("stencils.jl")
include("general.jl")

# Access-style trait + abstract stencil supertype.
export AccessStyle, ColumnAccess, RowAccess, AbstractStencil

# Term-like supertype shared by arrays and symbolic terms.
export AbstractTerm, ArrayOrTermLike

# Scalar algebra: abstract supertype + concrete leaves and tree node.
export AbstractScalar, Symbolic, Const, Null, Unity, Scalar
export @symbolic, var"@const"

# Type-level offsets.
export StaticPair, SPair, StaticShift, SShift, dim, offset
export ô, ê₁, ê₂, ê₃, ê₄, ê₅, ê₆, ê₇, ê₈, ê₉

# Stencil types (relaxed coefficient; assembly lives in StencilAssembly).
export LinearStencil, StarStencil, Stencil

# Narrowing (Stencil → assemblable LinearStencil / StarStencil).
export as_linear, as_star

end # module StencilCore
