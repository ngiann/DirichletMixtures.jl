module DirichletMixtures


    using Distributions, StatsFuns, Printf

    include("fitdirichlet.jl")
    include("fitdirichletmixture.jl")

    export fitdirichlet, fitdirichletmixture


end
