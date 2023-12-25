
#check 




function _implies(a::SubClause, b::SubClause)::Bool
    if feature(a) == feature(b)
        if direction(a) == :L
            if direction(b) == :L
                return splitval(a) ≤ splitval(b)
            else
                return false
            end
        else
            if direction(b) == :R
                return splitval(a) ≥ splitval(b)
            else
                return false
            end
        end
    else
        return false
    end
end

