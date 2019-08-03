#' @return The shannon entropy for distribution p.
#' x is the probability distribution
#' base is the base of the logarithm
shannon.entropy = function(p, base = 2)
{
    # remove zeros and normalize, just in case
    p = p[p > 0] / sum(p)
    
    H = sum(p*log(1/p,base))

    # return the value of H
    return(H)
}

#' @return The jensen-shannon entropy for distributions p and q.
#' x is the probability distribution
#' base is the base of the logarithm
JSD = function(p, q, base = 2)
{
    m = (p+q)/2
    J = shannon.entropy(m,base) - 
       (shannon.entropy(p,base) + 
        shannon.entropy(q,base))/2
    
    # return the value of J
    return(J)
}