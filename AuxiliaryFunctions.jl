

function MatrixToVector(M)
    "Concatenates Matrix into a vector by putting the columns underneath eachother"
    collength = length(M[1,:])
    vec = []
    for i = 1:collength
        addvec = M[:,i]
        vec = [vec;addvec];
    end
    return vec
end

##
function MvNormalSample(μ,Σ)
    """Sample from a MV normal distribution using spectral decomposition"""
    n = length(μ)
    F = eigen(Σ)
    U = F.vectors
    Λ = Diagonal(F.values)
    sqrtΛ = Diagonal(sqrt.(Complex.(F.values)))
    A = U*sqrtΛ
    z = rand(Normal(),n,1)
    x = real(μ + A*z)
    return x
end

## MV pdfs
function MvNormalpdf(μ,Σ,y)
    """Also works for non positive definite Σ (MvNormal does not!). Returns the logarithm"""
    N = length(μ)
    Det = det(Σ)

    if Det == 0.0
        factor = 1
        while Det == 0.0
            factor += 1
            Det = det(Σ*factor)
        end
        global value = N*log(factor).-(N/2)*log(2*pi).-log(sqrt(abs(det(Σ*factor)))).-0.5*(transpose(y-μ))*inv(Σ)*(y-μ)
    elseif Det == Inf
        factor = 1
        while Det == Inf
            factor += 1
            Det = det(Σ/factor)
        end
        global value = -N*log(factor)-(N/2)*log((2*pi)).-log(sqrt(abs(det(Σ*factor)))).-0.5*(transpose(y-μ))*inv(Σ)*(y-μ)
    else
        global value = -(N/2)*log(2*pi).-log(sqrt(abs(det(Σ)))).-0.5*(transpose(y-μ))*inv(Σ)*(y-μ)
    end
    return value[1]
end

function MvTDistpdf(d,μ,Σ,X)
    p = length(X)
    val = loggamma((d+p)/2).-loggamma(d/2).-(p/2)*log(d*pi).-(1/2)*logdet(Complex.(Σ)).-0.5*(d+p)*log(Complex.(1 .+(1/d)*transpose(X-μ)*inv(Σ)*(X-μ)))
    return real(val[1])
end
