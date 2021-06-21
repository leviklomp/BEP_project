using Random
using Statistics
using SpecialFunctions
using Distributions
using StatsBase
using Plots
using LinearAlgebra
using JLD

# Example :transition_density(θ) = MvNormal(θ,Matrix(I,.,.))
function MetropolisHastingsAlgorithm(posterior_density,initial_θ,transition_density,transition_sample,n = 1000,burnin = 200,logvariant = true)
    acceptance = 0
    Dim = size(initial_θ)[1]
    θ_matrix = zeros(Dim,n+1)
    θ_matrix[:,1] = initial_θ
    for i = 1:n
        println(i)
        θ_i = θ_matrix[:,i]
        θ_new = transition_sample(θ_i) #potential new θ coordinate

        q_z_to_y = transition_density(θ_new,θ_i)
        q_y_to_z = transition_density(θ_i,θ_new)
        v_y = posterior_density(θ_i)
        v_z = posterior_density(θ_new)

        if logvariant
            #log version
            #this is done because this can cause errors (not clear why)


            q_ratio_log = q_z_to_y - q_y_to_z
            v_ratio_log = v_z - v_y
            #println([q_y_to_z,q_z_to_y,q_ratio_log])
            #println([v_y,v_z,v_ratio_log])
            log_α = v_ratio_log+q_ratio_log
            #println(log_α)
            u = rand(Uniform(0,1))

            if log(u) < log_α
                println("ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT")
                θ_i = θ_new
                acceptance += 1
            end
        else #normal version
            #this is done because this can cause errors (not clear why)
            if (v_y  >= 1) | (v_z  >= 1)
                v_z,v_y = NaN,NaN
                println("REJECT REJECT REJECT REJECT REJECT REJECT REJECT REJECT REJECT REJECT")
            end
            q_ratio = q_z_to_y/q_y_to_z
            v_ratio = v_z/v_y
            println([q_y_to_z,q_z_to_y,q_ratio])
            println([v_y,v_z,v_ratio])
            α = v_ratio*q_ratio
            u = rand(Uniform(0,1))

            if u < α
                println("ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT")
                θ_i = θ_new
                acceptance += 1
            end

        end
        θ_matrix[:,i+1] = θ_i
    end
    return θ_matrix[:,burnin:end],acceptance/n
end

## Needed for the transition density
function MvNormalpdf(μ,Σ,y)
    """Also works for non positive definite Σ (MvNormal does not!). Returns the logarithm"""
    N = length(μ)
    Det = det(Σ)

    #((2*pi)^(N/2)*
    #(1/((2*pi)^(N/2)*sqrt(abs(det(Σ)))))*exp(-0.5*(transpose(y-μ))*inv(Σ)*(y-μ)))
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

function MvTDistpdf(d,μ,Σ,X) #dealing with underflow using eigenvalues
    p = length(X)
    val = loggamma((d+p)/2).-loggamma(d/2).-(p/2)*log(d*pi).-(1/2)*logdet(Complex.(Σ)).-0.5*(d+p)*log(Complex.(1 .+(1/d)*transpose(X-μ)*inv(Σ)*(X-μ)))
    return real(val[1])
end
