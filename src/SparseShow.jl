module SparseShow

using SparseArrays
using Base: alignment, undef_ref_str, undef_ref_alignment
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
    paddings = (ul, vl, pre, sep, post, hd, vd, dd) = 
               ("_", "|", " ", " ", "", "  \u2026  ", "\u22ee", "  \u22f1  ")
    
    io = IOContext(io, :compact => true)
    xnnz = nnz(S)
    print(io, S.m, "×", S.n, " ", typeof(S), " with ", xnnz, " stored ",
              xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        print(io, ":\n")
        shown = selectshown(io, S, length.(paddings))
        matrix, rowinds, colinds, aligncol0, aligncols, rowsplit, colsplit = selectshown(io, S, length.(paddings))
        print_table(io, matrix; headertop = colinds, headerleft = rowinds, 
                                aligncols = aligncols, aligncol0 = aligncol0[],
                                rowsplit = rowsplit, colsplit = colsplit, 
                                sepheader = vl, padheader = ul, pre = pre, sepbody = sep, post = post,
                                hdots = hd, vdots = vd, ddots = dd)
    end
end

function print_table(io, table; headertop = missing, headerleft = missing, 
                                aligncols = missing, aligncol0 = missing,
                                rowsplit = missing, colsplit = missing, 
                                sepheader = "|", padheader = "_", pre = " ", sepbody = " ", post = "", 
                                hdots = "  \u2026  ", vdots = "\u22ee", ddots = "  \u22f1  ", 
                                hmod = 5, vmod = 5)    
    if !ismissing(headertop)
        for j in 0:size(table, 2)
            j == 0 ? printcell(io, aligncol0, Text(""); pre = pre, pad = padheader) :
                     printcell(io, aligncols[j], headertop[j]; pre = sepheader, pad = padheader, post = padheader, justify = :left)
            !ismissing(colsplit) && colsplit == j &&
                     printcell(io, alignment(io, Text(hdots)), Text(hdots))
        end
        print(io, "\n")
    end
    
    for i in 0:size(table, 1)
        if !ismissing(rowsplit) && rowsplit == i
            for j in 0:size(table, 2)
                printcell(io, j == 0 ? aligncol0 : aligncols[j], mod(j, hmod) == 0 ? Text(vdots) : Text(""),
                          pre = j <= 1 ? pre : sepbody, 
                          post = j == 0 ? Text("") : sepbody, 
                          justify = j == 0 ? :right : :dot)
                if !ismissing(colsplit) && colsplit == j
                    printcell(io, alignment(io, Text(ddots)), Text(ddots))
                end
            end
            print(io, "\n")
        end 
        i == 0 && continue
        for j in 0:size(table, 2)
            if j == 0 && !ismissing(headerleft)
                 printcell(io, aligncol0, headerleft[i]; pre = pre, justify = :right)
            else
                body = _getformatted(table, i, j)
                printcell(io, aligncols[j], body, pre = j == 1 ? sepheader : sepbody, post = sepbody)
            end
            if !ismissing(colsplit) && colsplit == j
                printcell(io, alignment(io, Text(hdots)), mod(i, vmod) == 0 ? Text(hdots) : Text(""))
            end
        end
        print(io, "\n")
    end
end

function printcell(io, cellalign, body; pre = "", pad = " ", post = "", justify = :dot)
    align = body == undef_ref_str ? undef_ref_alignment : alignment(io, body)
    if justify == :left
        align = (0, sum(align))
        cellalign = (0, sum(cellalign))
    elseif justify == :right
        align = (sum(align), 0)
        cellalign = (sum(cellalign), 0)
    end
    lefpad = repeat(pad, cellalign[1] - align[1]) 
    rightpad = repeat(pad, cellalign[2] - align[2])
    print(io, pre, lefpad, body, rightpad, post)
end

_getformatted(table, i, j, missingchar = '⋅') =
    isassigned(table, Int(i), Int(j)) ? (x = table[i,j]; x === missing ? Text(missingchar) : x) : undef_ref_str

function selectshown(io, s::SparseMatrixCSC{Tv,Ti}, padlengths) where {Tv,Ti}
    sz = size(s)
    IJV = sort!(findnz_zip(s), by = ijv -> _cornerdist(ijv, sz))
    IJVshown = Tuple{Ti,Ti,Tv}[]
    rowinds = Ti[]
    colinds = Ti[]
    aligncol0 = Ref((1,0)) # alignment of row index column
    aligncols = Tuple{Int,Int}[]  # first col is for row indices
    screensize = _screensize(io)
    for (i, j, v) in IJV
        isnewrow = !(i in rowinds)
        isnewcol = !(j in colinds)
        isnewrow && push!(rowinds, i)
        isnewcol && push!(colinds, j)
        push!(IJVshown, (i, j, v))
        _updatealignments!(aligncol0, aligncols, io, IJVshown, rowinds, colinds)
        printsize = (_totalheight(rowinds, padlengths), _totalwidth(aligncol0, aligncols, padlengths))
        if any(printsize .> screensize)  # rewind
            pop!(IJVshown)
            isnewrow && pop!(rowinds)
            isnewcol && (pop!(colinds); pop!(aligncols))
        end
        all(printsize .> screensize) && break
    end

    numrows, numcols = length(rowinds), length(colinds)
    matrix = Union{Tv, Missing}[missing for _ in 1:numrows, _ in 1:numcols]
  
    sort!(rowinds)
    colperms = sortperm(colinds)
    aligncols = aligncols[colperms]
    sort!(colinds)  
    
    rowsplit = _findsplit(1, rowinds, sz, IJV, IJVshown)
    colsplit = _findsplit(2, colinds, sz, IJV, IJVshown)
    
    for (i, j, v) in IJVshown
        row = findfirst(isequal(i), rowinds)
        col = findfirst(isequal(j), colinds)
        matrix[row, col] = v
    end

    return matrix, rowinds, colinds, aligncol0, aligncols, rowsplit, colsplit
end

_cornerdist((i, j, v), (nrows, ncols)) = min(i - 1, nrows - i) + min(j - 1, ncols - j)

function _updatealignments!(aligncol0, aligncols, io, IJVshown, rowinds, colinds)
    length(colinds) > length(aligncols) && push!(aligncols, (1,0))
    @assert length(colinds) == length(aligncols)
    (_, j, v) = last(IJVshown)
    col = findfirst(isequal(j), colinds)  # should never return `nothing`
    # widen alignments of row-index and value columns, including the length of the column index in the latter
    aligncols[col] = max.(aligncols[col], alignment(io, v), (1, sum(alignment(io, colinds[col]))))
    aligncol0[] = max.(aligncol0[], alignment(io, last(rowinds)))
    return nothing
end
_totalwidth(aligncol0, aligncols, (ul, vl, pre, sep, post, hd, vd, dd)) =
    pre + sum(aligncol0[]) + vl + sum(a -> sep + 1 + sum(a), aligncols) - sep + hd + post
_totalheight(rowinds, (ul, vl, pre, sep, post, hd, vd, dd)) = 1 + vd + length(rowinds)
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
    end
    hiddenrange = length(IJVshown)+1:length(IJV)
    emptyinterval = !any(i -> IJV[i][axis] in checkinterval, hiddenrange)
    return emptyinterval ? missing : split
end

end # module