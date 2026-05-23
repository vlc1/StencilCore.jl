using Documenter
using StencilCore

makedocs(;
    sitename = "StencilCore.jl",
    modules = [StencilCore],
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "API reference" => "api.md",
    ],
    checkdocs = :none,
    # Some docstrings (Symbolic, Null, Unity, Scalar, @symbolic, simplify,
    # materialize, differentiate) name their term-side analogues — which live
    # in StencilCalculus, so @ref cannot resolve them here.
    warnonly = [:cross_references],
)

deploydocs(;
    repo = "github.com/vlc1/StencilCore.jl",
    devbranch = "main",
)
