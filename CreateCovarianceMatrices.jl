using LinearAlgebra
using Random
using Distributions

## FIRST MODEL

function CreateKernelMatrix1D(kernelfunction1d,xas)
    """Creates kernel for 1-dimensional line """
    N = length(xas)
    K_matrix = zeros(N,N)
    for i = 1:N
        for j = 1:N
            K_matrix[i,j] = kernelfunction1d(xas[i],xas[j])
        end
    end
    return K_matrix
end

function CreateKernelMatrix2D(kernelfunction2d,xas,yas)
    """Creates kernel for 2-dimensional grid """
    Nx = length(xas)
    Ny = length(yas)
    K_matrix = zeros(Nx*Ny,Nx*Ny)
    for x1 = 1:Nx, y1 = 1:Ny #first coordinate
        for y2 = 1:Ny, x2 = 1:Nx #second coordinate
            t_rowindex = [xas[x1];yas[y1]]
            t_colindex = [xas[x2];yas[y2]]
            rowindex = (x1-1)*Ny + y1
            colindex = (x2-1)*Ny + y2
            K_matrix[rowindex,colindex] = kernelfunction2d(t_rowindex,t_colindex)
        end
    end

    return K_matrix
end

function CreateKernelMatrixCoordset2D(kernelfunction2d,Coordset)
    """Creates kernel for 2-dimensional coordinate set (input is an array of tuples)"""
    N = length(Coordset)
    K_matrix = zeros(N,N)
    for i = 1:N
        for j = 1:N
            t1 = Coordset[i]
            t2 = Coordset[j]
            K_matrix[i,j] = kernelfunction2d(t1,t2)
        end
    end
    return K_matrix
end


## SECOND MODEL
function CovarianceMatrixCoefficients2D(n,f)
    """Creates covariance matrix given the indices of the eigenfunction. This is a 2D case and
    hence the matrix return will be n^2 X n^2"""
    C = zeros(Int(n^2),Int(n^2))
    for l2 = 1:n
        for m2 = 1:n
            for l = 1:n
                for m = 1:n
                    #calculating the value and index of coefficient a_lm against a_pq
                    rowindex = n*(l-1)+m
                    colindex = n*(l2-1)+m2
                    C[rowindex,colindex]  = f(l,m,l2,m2)
                end
            end
        end
    end
    return C
end



function GridEigenFunctionMatrix2D(xas,yas,f,n)
    """Returns eigenfunction matrix for a given grid (xas,yas) 2D-eigenfunction 'f' and the number
    n^2 distinct eigenfunctions"""
    Nx = length(xas)
    Ny = length(yas)
    CoordSet = []
    for i = 1:Nx
        for j = 1:Ny
            CoordSet = append!(CoordSet,[(xas[i],yas[j])])
        end
    end
    return CoordsetEigenFunctionMatrix2D(CoordSet,f,n)
end

function CoordsetEigenFunctionMatrix2D(Coordset,f,n)
    """Returns eigenfunction matrix for a given coordinate set (array of tuples) 2D-eigenfunction 'f' and the number
    n^2 distinct eigenfunctions"""
    N = length(Coordset)
    E = zeros(N,Int(n^2))*im
    for i = 1:N
        for l = 1:n
            for m = 1:n
                t = Coordset[i]
                col_idx = (l-1)*n+m
                E[i,col_idx] = f(t[1],t[2],l,m)
            end
        end
    end
    return real.(E)
end
