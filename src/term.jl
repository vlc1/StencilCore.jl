# The abstract term-like supertype shared by arrays and symbolic terms.
#
# `AbstractTerm{T}` is "a dimension- and size-less array-like object
# whose `eltype` is `T`" (when `T` is concrete) or "looks like `T`"
# (when `T` is abstract). Concrete subtypes (`Slot`, `Const`, `Term`,
# `Shifted`) live in the StencilCalculus package; this package only owns the
# supertype so that stencil coefficient types can be expressed as
# `ArrayOrTermLike{T}` without depending on the CAS.

"""
    AbstractTerm{T}

Supertype of every symbolic grid expression. An `AbstractTerm{T}`
behaves like a dimension- and size-less array whose `eltype` is `T`:
its grid rank `N` is unknown until it is materialized against concrete
arrays, but its element type `T` (the value each cell will hold once
materialized) is fixed at construction.

`eltype(::AbstractTerm{T}) === T`. Concrete subtypes are provided by the
StencilCalculus package.
"""
abstract type AbstractTerm{T} end

Base.eltype(::Type{<:AbstractTerm{T}}) where {T} = T
Base.eltype(t::AbstractTerm) = eltype(typeof(t))

# Shared concrete-eltype guard, used by every leaf type whose `T` it
# materializes / assembles into (Slot, Symbolic, Const, Null, Unity, …).
@inline function _assert_concrete(name::Symbol, ::Type{T}) where {T}
    isconcretetype(T) || throw(ArgumentError(
        "$(name) needs a concrete element type; got $(T). Use e.g. Float64 " *
        "(the default), Float32, or a concrete SVector."))
    return nothing
end

"""
    ArrayOrTermLike{T} = Union{AbstractArray{T}, AbstractTerm{T}}

A coefficient that is either a concrete array (assemblable) or a
symbolic term (must be `materialize`d first), both with element type
`T`. Stencil types parameterise their coefficient field over this
union.
"""
const ArrayOrTermLike{T} = Union{AbstractArray{T}, AbstractTerm{T}}
