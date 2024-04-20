function fitdirichletmixture(X; M = 2, iterations = 10, pr = ones(M) / M, 𝛂 = [rand(size(X, 1))*3 .+ 0.1 for _ in 1:M], reportlowerbound=false)

    K, N = size(X)

    @assert(M == length(pr) == length(𝛂))

    @assert(all(K .== length.(𝛂)))
    
    
    # return N × M matrix
    loglikelihood(𝛂) = reduce(hcat, [[logpdf(Dirichlet(𝛂ₖ), x) for x in eachcol(X)] for 𝛂ₖ in 𝛂])
    
    
    function responsibilities(logl, pr)

        @assert(size(logl, 2) == length(pr) == M)
        
        local logresp = [logl[n, m] + log(pr[m]) for n in 1:N, m in 1:M]
        
        local resp = exp.(logresp .- logsumexp(logresp, dims = 2))
        
        @assert(size(logresp) == size(resp) == size(logl) == (N, M))
     
        return resp  # return N × M matrix
        
    end
    

    # iterate between two steps: calculate responsibilities and adapt components

    for iter in 1:iterations

        # calculate log-likelihoods

        local logl = loglikelihood(𝛂) 

        # calculate responsibilities

        local resp = responsibilities(logl, pr)

        # update prior

        pr = vec(sum(resp, dims=1)) / N

        # update dirichlet parameters

        for m in 1:M

            dₘ = Distributions.fit_mle(Dirichlet, X, vec(resp[:, m]), maxiter = 100_000)

            𝛂[m] = params(dₘ)[1]

        end

        if reportlowerbound

            lowerbound = sum(resp .* logl) + sum([entropy(resp[:, m]) for m in 1:M])
            
            @printf("(%d) Lower bound reads: %f\n", iter, lowerbound)
            
        end

    end

    return pr, 𝛂

end