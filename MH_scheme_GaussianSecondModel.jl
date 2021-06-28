using Random
using Distributions
include("CreateCovarianceMatrices.jl")
include("AuxiliaryFunctions.jl")



function log_posterior(Y,ν2,a,γ,Mask,E,n,func_γ)

    γ_mat = zeros(n,n)

    for i = 1:n
        γ_mat[i,:] = γ[(i-1)*n+1:i*n]
    end
    γ = γ_mat
    return log_Y(Y,ν2,a,E,Mask)+log_a(a,γ,n,func_γ)+log_π_ν2(ν2)+log_π_γ(γ)
end

function log_Y(Y,ν2,a,E,Mask)
    N = length(Y)
    log_Y = MvNormalpdf(Mask*E*a,ν2*Matrix(I,N,N),Y)
    return log_Y
end

function log_a(a,γ,n,func_γ)
    func_given_γ(l,m,l2,m2) = func_γ(l,m,l2,m2,γ)
    C_γ = CovarianceMatrixCoefficients2D(n,func_given_γ)
    N = length(a)
    return MvNormalpdf(zeros(N),C_γ,a)
end

function log_π_ν2(ν2)
    return log(pdf(Normal(5,1),ν2))
end

function log_π_γ(γ)
    N = size(γ)[1]
    val = 1
    for i = 1:N, j = 1:N
        val = val*pdf(Exponential(1/(i*j)),γ[i,j])
    end
    return log(val)
end
