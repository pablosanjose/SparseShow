module SparseShow

using SparseArrays
export spshow

struct Spdef end
spdef = Spdef()

dist((i, j), (sx, sy)) = min(i - 1, sx - i) + min(j - 1, sy - j)
selectshown(s, n::Int) = selectshown(s, (n, n))
function selectshown(s::SparseMatrixCSC{T,TI}, (maxrows, maxcols)::Tuple) where {T,TI}
    I, J, V = findnz(s)
    srow, scol = size(s)
    dists = [dist(ij, (srow, scol)) for ij in zip(I,J)]
    perm = sortperm(dists)
    nperm = length(perm)
    IJVn = Tuple{TI,TI,T}[]
    rows = TI[]
    cols = TI[]
    nrows = 0; ncols = 0; pind = 1;
    for ind in perm
        i, j, v = I[ind], J[ind], V[ind]
        isnewrow = !(i in rows)
        isnewcol = !(j in cols)
        if nrows + Int(isnewrow) <= maxrows && ncols + Int(isnewcol) <= maxcols
            isnewrow && (nrows += 1; push!(rows, i))
            isnewcol && (ncols += 1; push!(cols, j))
        else
            continue
        end
        push!(IJVn, (i, j, v))
        nrows == maxrows && ncols == maxcols && break
    end
    sort!(rows)
    sort!(cols)
    maxrows, maxcols = length(rows), length(cols)
    matrix = Union{T,Int,Spdef}[spdef for _ in 1:maxrows+1, _ in 1:maxcols+1]
    for (i, j, v) in IJVn
        row = findfirst(isequal(i), rows)
        col = findfirst(isequal(j), cols)
        matrix[row + 1, col + 1] = v
    end
    for row in 1:maxrows
        matrix[row + 1, 1] = rows[row]
    end
    for col in 1:maxcols
        matrix[1, col + 1] = cols[col]
    end
    return matrix
end

spshow(s::SparseMatrixCSC) = spshow(stdout, s)
spshow(io::IO, s::SparseMatrixCSC{T}) where {T} = selectshown(s, displaysize(io) ./ (1,10) .- (5,2))

function Base.replace_in_print_matrix(A::Matrix{Union{T,Int,Spdef}}, i::Integer, j::Integer,s::AbstractString) where {T}
    if i == j == 1
        return Base.replace_with_centered_mark(s, c = ' ')
    elseif A[i, j] isa Spdef
        return Base.replace_with_centered_mark(s)
    else
        return s
    end
end

end # module
