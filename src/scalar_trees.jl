# AbstractTrees interface for AbstractScalar — parallel to the term-side
# implementation in StencilCalculus/src/trees.jl. Internal `Scalar` nodes
# expose their operator as the node value and operand scalars as children;
# leaves are childless.

AbstractTrees.nodevalue(::Symbolic{S}) where {S} = S
AbstractTrees.children(::Symbolic)               = ()

AbstractTrees.nodevalue(c::Const) = c.val
AbstractTrees.children(::Const)   = ()

AbstractTrees.nodevalue(::Null{T}) where {T} = zero(T)
AbstractTrees.children(::Null)               = ()

AbstractTrees.nodevalue(::Unity{T}) where {T} = one(T)
AbstractTrees.children(::Unity)               = ()

AbstractTrees.nodevalue(s::Scalar) = s.fn
AbstractTrees.children(s::Scalar)  = s.args
