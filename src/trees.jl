# AbstractTrees interface for AbstractScalar — parallel to the term-side
# implementation in StencilCalculus/src/trees.jl. Internal `Scalar` nodes
# expose their operator as the node value and operand scalars as children;
# leaves are childless.

AbstractTrees.nodevalue(::Symbolic{S, T}) where {S, T} = (S, T)
AbstractTrees.children(::Symbolic)                     = ()

AbstractTrees.nodevalue(s::Scaling) = s.val
AbstractTrees.children(::Scaling)   = ()

AbstractTrees.nodevalue(::Null{T}) where {T} = zero(T)
AbstractTrees.children(::Null)               = ()

AbstractTrees.nodevalue(s::Scalar) = s.fn
AbstractTrees.children(s::Scalar)  = s.args
