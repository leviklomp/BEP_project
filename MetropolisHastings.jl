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
            #log version (to prevent numerical underflow)
            
            q_ratio_log = q_z_to_y - q_y_to_z
            v_ratio_log = v_z - v_y
           
            log_α = v_ratio_log+q_ratio_log
            #println(log_α)
            u = rand(Uniform(0,1))

            if log(u) < log_α #acceptance
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
           
            α = v_ratio*q_ratio
            u = rand(Uniform(0,1))
          
            if u < α #acceptance
                println("ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT ACCEPT")
                θ_i = θ_new
                acceptance += 1
            end

        end
        θ_matrix[:,i+1] = θ_i
    end
    return θ_matrix[:,burnin:end],acceptance/n
end

