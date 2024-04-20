function fitdirichlet(X; iterations = 10)

    Distributions.fit_mle(Dirichlet, X, maxiter = iterations)

end