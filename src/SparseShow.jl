module SparseShow

using SparseArrays
using Base: alignment
export sparseshow

function findnz_zip(S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    numnz = nnz(S)
    IJV = Vector{Tuple{Ti,Ti,Tv}}(undef, numnz)

    count = 1
    @inbounds for col = 1 : S.n, k = S.colptr[col] : (S.colptr[col+1]-1)
        IJV[count] = (S.rowval[k], col, S.nzval[k])
        count += 1
    end

    return IJV
end

sparseshow(S::SparseMatrixCSC) = sparseshow(stdout, S)
function sparseshow(io::IO, S::SparseMatrixCSC)
    paddings = (pre, sep, post, hd, vd, dd, ul, vl) = 
               (" ", "  ", "", "  \u2026  ", "\u22ee", "  \u22f1  ", "_", "|")
    
    io = IOContext(io, :compact => true)
    xnnz = nnz(S)
    print(io, S.m, "×", S.n, " ", typeof(S), " with ", xnnz, " stored ",
              xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        print(io, ":\n")
        shown = selectshown(io, S, length.(paddings))
        print_sparse_matrix(io, shown, paddings)
    end
end
    
function print_sparse_matrix(io, (matrix, rowinds, colinds, ind_align, col_aligns, rowsplit, colsplit),
                            (pre, sep, post, hd, vd, dd, ul, vl), emptymark = '⋅', hmod = 5, vmod = 5)

    print(io, pre, repeat(ul, sum(ind_align[])))
    for j in eachindex(colinds)                         # print table header
        align = alignment(io, colinds[j])
        r = repeat(ul, sum(col_aligns[j]) - sum(align) + 1)
        print(io, vl, colinds[j],  r)
    end
    print(io, "\n")

    for i in eachindex(rowinds)
        align = alignment(io, rowinds[i])
        l = repeat(" ", sum(ind_align[]) - sum(align))  # print row header
        print(io, pre, l, rowinds[i], vl)
        for j in eachindex(colinds)
            if !isassigned(matrix, Int(i), Int(j))      # isassigned accepts only `Int` indices
                align = Base.undef_ref_alignment
                sx = Base.undef_ref_str
            elseif ismissing(matrix[i,j])
                align = alignment(io, Text(emptymark))  # replace missings with emptymark
                sx = sprint(show, Text(emptymark), context = io, sizehint = 0)
            else
                x = matrix[i,j]
                align = alignment(io, x)
                sx = sprint(show, x, context = io, sizehint = 0)
            end
            l = repeat(" ", col_aligns[j][1]-align[1])  # pad on left/right and right as needed
            r = repeat(" ", col_aligns[j][2]-align[2])
            print(io, l, sx, r, sep)
        end
        print(io, "\n")
        if !ismissing(rowsplit) && i == rowsplit
            align = alignment(io, Text(vd))
            l = repeat(" ", sum(ind_align[]) - sum(align))
            print(io, pre, l, vd, vl)   
            for j in eachindex(colinds)
                if mod(j, hmod) == 0
                    l = repeat(" ", col_aligns[j][1] - align[1])
                    r = repeat(" ", col_aligns[j][2] - align[2])
                    print(io, l, Text(vd), r, sep)  
                else
                    l = repeat(" ", sum(col_aligns[j]))
                    print(io, l, sep)
                end
            end
            print(io, "\n")
        end
    end
end

function selectshown(io, s::SparseMatrixCSC{Tv,Ti}, padlengths) where {Tv,Ti}
    sz = size(s)
    IJV = sort!(findnz_zip(s), by = ijv -> _cornerdist(ijv, sz))
    IJVshown = Tuple{Ti,Ti,Tv}[]
    rowinds = Ti[]
    colinds = Ti[]
    ind_align = Ref((1,0)) # alignment of row index column
    col_aligns = Tuple{Int,Int}[]  # first col is for row indices
    screensize = _screensize(io)
    for (i, j, v) in IJV
        isnewrow = !(i in rowinds)
        isnewcol = !(j in colinds)
        isnewrow && push!(rowinds, i)
        isnewcol && push!(colinds, j)
        push!(IJVshown, (i, j, v))
        _updatealignments!(ind_align, col_aligns, io, IJVshown, rowinds, colinds)
        printsize = (_totalheight(rowinds, padlengths), _totalwidth(ind_align, col_aligns, padlengths))
        if any(printsize .> screensize)  # rewind
            pop!(IJVshown)
            isnewrow && pop!(rowinds)
            isnewcol && (pop!(colinds); pop!(col_aligns))
        end
        all(printsize .> screensize) && break
    end

    numrows, numcols = length(rowinds), length(colinds)
    matrix = Union{Tv, Missing}[missing for _ in 1:numrows + 1, _ in 1:numcols + 1]
  
    sort!(rowinds)
    colperms = sortperm(colinds)
    col_aligns = col_aligns[colperms]
    sort!(colinds)  
    
    rowsplit = _findsplit(1, rowinds, sz, IJV, IJVshown)
    colsplit = _findsplit(2, colinds, sz, IJV, IJVshown)
    
    for (i, j, v) in IJVshown
        row = findfirst(isequal(i), rowinds)
        col = findfirst(isequal(j), colinds)
        matrix[row, col] = v
    end

    return matrix, rowinds, colinds, ind_align, col_aligns, rowsplit, colsplit
end

_cornerdist((i, j, v), (nrows, ncols)) = min(i - 1, nrows - i) + min(j - 1, ncols - j)

function _updatealignments!(ind_align, col_aligns, io, IJVshown, rowinds, colinds)
    length(colinds) > length(col_aligns) && push!(col_aligns, (1,0))
    @assert length(colinds) == length(col_aligns)
    (_, j, v) = last(IJVshown)
    col = findfirst(isequal(j), colinds)  # should never return `nothing`
    # widen alignments of row-index and value columns, including the length of the column index in the latter
    col_aligns[col] = max.(col_aligns[col], alignment(io, v), (1, sum(alignment(io, colinds[col]))))
    ind_align[] = max.(ind_align[], alignment(io, last(rowinds)))
    return nothing
end
_totalwidth(ind_align, col_aligns, (pre, sep, post, hd, vd, dd, ul, vl)) =
    pre + sum(ind_align[]) + vl + sum(a -> sep + 1 + sum(a), col_aligns) - sep + hd + post
_totalheight(rowinds, (pre, sep, post, hd, vd, dd, ul, vl)) = 1 + vd + length(rowinds)
#_screensize(io) = !get(io, :limit, false) ? typemax(Int) : displaysize(io) .- (4, 0)
_screensize(io) = displaysize(io) .- (4, 0)
function _findsplit(axis, inds, sz, IJV, IJVshown) 
    (length(IJV) == length(IJVshown) || isempty(inds)) && return missing
    middle = sz[axis] ÷ 2
    split = findfirst(i -> i > middle, inds)
    if split === nothing
        checkinterval = last(inds)+1:sz[axis]
        split = length(inds)
    elseif split == 1
        checkinterval = 1:first(inds)-1
        split = 0
    else
        checkinterval = inds[split-1]+1:inds[split]-1
        split -= 1
    end
    hiddenrange = length(IJVshown)+1:length(IJV)
    emptyinterval = !any(i -> IJV[i][axis] in checkinterval, hiddenrange)
    return emptyinterval ? missing : split
end

# function print_new_matrix(io::IO, X::AbstractVecOrMat,
#                              pre::AbstractString = " ",  # pre-matrix string
#                              sep::AbstractString = "  ", # separator between elements
#                              post::AbstractString = "",  # post-matrix string
#                              hdots::AbstractString = "  \u2026  ",
#                              vdots::AbstractString = "\u22ee",
#                              ddots::AbstractString = "  \u22f1  ",
#                              hmod::Integer = 5, vmod::Integer = 5)

#     if !get(io, :limit, false)
#         screenheight = screenwidth = typemax(Int)
#     else
#         sz = displaysize(io)
#         screenheight, screenwidth = sz[1] - 4, sz[2]
#     end
#     screenwidth -= length(pre) + length(post)
#     presp = repeat(" ", length(pre))  # indent each row to match pre string
#     postsp = ""
#     @assert textwidth(hdots) == textwidth(ddots)
#     sepsize = length(sep)
#     rowsA, colsA = UnitRange(axes(X,1)), UnitRange(axes(X,2))
#     m, n = length(rowsA), length(colsA)
#     # To figure out alignments, only need to look at as many rows as could
#     # fit down screen. If screen has at least as many rows as A, look at A.
#     # If not, then we only need to look at the first and last chunks of A,
#     # each half a screen height in size.
#     halfheight = div(screenheight,2)
#     if m > screenheight
#         rowsA = [rowsA[(0:halfheight-1) .+ firstindex(rowsA)]; rowsA[(end-div(screenheight-1,2)+1):end]]
#     end
#     # Similarly for columns, only necessary to get alignments for as many
#     # columns as could conceivably fit across the screen
#     maxpossiblecols = div(screenwidth, 1+sepsize)
#     if n > maxpossiblecols
#         colsA = [colsA[(0:maxpossiblecols-1) .+ firstindex(colsA)]; colsA[(end-maxpossiblecols+1):end]]
#     end
#     A = alignment(io, X, rowsA, colsA, screenwidth, screenwidth, sepsize)
#     # Nine-slicing is accomplished using print_matrix_row repeatedly
#     if m <= screenheight # rows fit vertically on screen
#         if n <= length(A) # rows and cols fit so just print whole matrix in one piece
#             for i in rowsA
#                 print(io, i == first(rowsA) ? pre : presp)
#                 print_matrix_row(io, X,A,i,colsA,sep)
#                 print(io, i == last(rowsA) ? post : postsp)
#                 if i != last(rowsA); println(io); end
#             end
#         else # rows fit down screen but cols don't, so need horizontal ellipsis
#             c = div(screenwidth-length(hdots)+1,2)+1  # what goes to right of ellipsis
#             Ralign = reverse(alignment(io, X, rowsA, reverse(colsA), c, c, sepsize)) # alignments for right
#             c = screenwidth - sum(map(sum,Ralign)) - (length(Ralign)-1)*sepsize - length(hdots)
#             Lalign = alignment(io, X, rowsA, colsA, c, c, sepsize) # alignments for left of ellipsis
#             for i in rowsA
#                 print(io, i == first(rowsA) ? pre : presp)
#                 print_matrix_row(io, X,Lalign,i,colsA[1:length(Lalign)],sep)
#                 print(io, (i - first(rowsA)) % hmod == 0 ? hdots : repeat(" ", length(hdots)))
#                 print_matrix_row(io, X, Ralign, i, (n - length(Ralign)) .+ colsA, sep)
#                 print(io, i == last(rowsA) ? post : postsp)
#                 if i != last(rowsA); println(io); end
#             end
#         end
#     else # rows don't fit so will need vertical ellipsis
#         if n <= length(A) # rows don't fit, cols do, so only vertical ellipsis
#             for i in rowsA
#                 print(io, i == first(rowsA) ? pre : presp)
#                 print_matrix_row(io, X,A,i,colsA,sep)
#                 print(io, i == last(rowsA) ? post : postsp)
#                 if i != rowsA[end] || i == rowsA[halfheight]; println(io); end
#                 if i == rowsA[halfheight]
#                     print(io, i == first(rowsA) ? pre : presp)
#                     print_matrix_vdots(io, vdots,A,sep,vmod,1)
#                     print(io, i == last(rowsA) ? post : postsp * '\n')
#                 end
#             end
#         else # neither rows nor cols fit, so use all 3 kinds of dots
#             c = div(screenwidth-length(hdots)+1,2)+1
#             Ralign = reverse(alignment(io, X, rowsA, reverse(colsA), c, c, sepsize))
#             c = screenwidth - sum(map(sum,Ralign)) - (length(Ralign)-1)*sepsize - length(hdots)
#             Lalign = alignment(io, X, rowsA, colsA, c, c, sepsize)
#             r = mod((length(Ralign)-n+1),vmod) # where to put dots on right half
#             for i in rowsA
#                 print(io, i == first(rowsA) ? pre : presp)
#                 print_matrix_row(io, X,Lalign,i,colsA[1:length(Lalign)],sep)
#                 print(io, (i - first(rowsA)) % hmod == 0 ? hdots : repeat(" ", length(hdots)))
#                 print_matrix_row(io, X,Ralign,i,(n-length(Ralign)).+colsA,sep)
#                 print(io, i == last(rowsA) ? post : postsp)
#                 if i != rowsA[end] || i == rowsA[halfheight]; println(io); end
#                 if i == rowsA[halfheight]
#                     print(io, i == first(rowsA) ? pre : presp)
#                     print_matrix_vdots(io, vdots,Lalign,sep,vmod,1)
#                     print(io, ddots)
#                     print_matrix_vdots(io, vdots,Ralign,sep,vmod,r)
#                     print(io, i == last(rowsA) ? post : postsp * '\n')
#                 end
#             end
#         end
#         if isempty(rowsA)
#             print(io, pre)
#             print(io, vdots)
#             length(colsA) > 1 && print(io, "    ", ddots)
#             print(io, post)
#         end
#     end
# end

# spshow(s::SparseMatrixCSC) = spshow(stdout, s)
# spshow(io::IO, s::SparseMatrixCSC{T}) where {T} = selectshown(s, displaysize(io) ./ (1,10) .- (5,2))

# function Base.replace_in_print_matrix(A::Matrix{Union{T,Missing}}, i::Integer, j::Integer,s::AbstractString) where {T}
#     if i == j == 1
#         return Base.replace_with_centered_mark(s, c = ' ')
#     elseif A[i, j] isa Spdef
#         return Base.replace_with_centered_mark(s)
#     else
#         return s
#     end
# end

end # module
