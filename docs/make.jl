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
)

deploydocs(;
    repo = "github.com/vlc1/StencilCore.jl",
    devbranch = "main",
)
