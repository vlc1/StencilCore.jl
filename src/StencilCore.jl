module StencilCore

using StaticArrays: SUnitRange, SVector

include("access.jl")
include("term.jl")
include("staticshift.jl")
include("stencils.jl")

# Access-style trait + abstract stencil supertype.
export AccessStyle, ColumnAccess, RowAccess, AbstractStencil

# Term-like supertype shared by arrays and symbolic terms.
export AbstractTerm, ArrayOrTermLike

# Type-level offsets.
export StaticPair, SPair, StaticShift, SShift, dim, offset
export ê₁, ê₂, ê₃, ê₄, ê₅, ê₆, ê₇, ê₈, ê₉

# Stencil types (relaxed coefficient; assembly lives in CartesianOperators).
export LinearStencil, StarStencil

end # module StencilCore
