# Group-Wise Lasso Using GMD Algorithm
# Translated from Yang & Zou's Fortran code
# Date: August 31, 2015
# Author: Paul Stey

using DataFrames
using Debug


type LassoPath
    lambdas::Array{Float64, 1}
    betas::Array{Float64, 2}
    beta0s::Array{Float64, 1}
    npass::Int64
end


function count_groups(groups::Array{Int64, 1})
    grp_hash = xtabs(groups)
    p = length(grp_hash)
    cnt = zeros(Int64, p)
    grps = [x for x in keys(grp_hash)]
    grps = sort(grps)
    for i = 1:p
        cnt[i] = grp_hash[grps[i]]
    end
    return cnt
end


function lamfix(lam::Array{Float64, 1})
    llam = Array(Float64, 3)
    for i = 1:3
        llam[i] = log(lam[i])
    end
    lam[1] = exp(2 * llam[2] - llam[3])
    return lam[1]
end


function not_converged(b::Array{Float64, 1}, oldbeta::Array{Float64, 1}, max_gam::Float64, eps::Float64)
    result = true
    n = length(b)
    for i = 1:n
        differ = (max_gam*(b[i] - oldbeta[i])/(1 + abs(b[i])))^2
        if differ > eps
            result = false
            break
        end
    end
    return result
end


function found_nonzero(v::Array{Float64, 1})
    result = false
    n = length(v)

    for i = 1:n
        if v[i] != 0.0
            result = true
            break
        end
    end
    return result
end


function update_jxx!(jxx::Array{Int64, 1}, ga::Array{Float64, 1}, pf::Array{Float64, 1}, al::Float64, al0::Float64, bn::Int64)
    tlam = 2*al - al0
    for i = 1:bn                       
        if jxx[i] == 1 
            continue
        end
        if ga[i] > pf[i]*tlam 
            jxx[i] = 1
        end
    end
end


function compute_u(x::Array{Float64, 2}, r::Array{Float64, 1}, b::Array{Float64, 1}, start::Int64, ending::Int64, nobs::Int64, gam_i::Float64)
    u_tmp = (x[:, start:ending]'*r)
    n = length(u_tmp)
    u = Array(Float64, n)
    for i = 1:n
        u[i] = gam_i*b[start:ending][i] + u_tmp[i]/nobs
    end
    return u
end


function compute_b(u::Array{Float64, 1}, t::Float64, gam_i::Float64, unorm::Float64)
    n = length(u)
    res = Array(Float64, n)
    tmp_term = t/(gam_i*unorm)
    for i = 1:n 
        res[i] = u[i]*tmp_term
    end
    return res 
end

        
function count_nonzero_betas(beta::Array{Float64, 2}, idx::Array{Int64, 1}, ix::Array{Int64, 1}, iy::Array{Int64, 1}, l::Int64, ni::Int64)
    me = 0
    for j = 1:ni
        g = idx[j]
        if found_nonzero(beta[ix[g]:iy[g], l])  
            me += 1
        end
    end
    return me
end


get_lamfact(n::Int64, p::Int64) = n < p ? 0.05 : 0.001


function get_upperbound(x::Array{Float64, 2}, ix::Array{Int64, 1}, iy::Array{Int64, 1}, bn::Int64)
    gam = zeros(Float64, bn)
    for i = 1:bn
        gam[i] = maximum(eig(x[:, ix[i]:iy[i]]' * x[:, ix[i]:iy[i]])[1])
    end
    return gam
end



function grp_lasso(x, y, group = nothing, loss = "ls", nlam::Int64 = 100, lambda = nothing, eps = 1.0E-08, maxit = 3.0E+08, delta = nothing, intr = true)
    if typeof(x) != Array{Float64, 2}
        x = convert(Array{Float64, 2}, x)
    end
    nobs, nvars = size(x)
    if length(y) != nobs
        error("x and y have different number of rows")
    end
    c1 = loss in ["logit", "sqsvm", "hsvm"]
    c2 = sort(unique(y)) != [-1, 1]

    if c1 && c2 
        error("Classification method requires the response y to be in {-1, 1}")
    end

    if isa(group, Void)
        group = [1:nvars]
    elseif length(group) != nvars 
        error("group length does not match the number of predictors in x")
    end
    bn = maximum(group)
    bs = count_groups(group)
    if sort(unique(group)) !=  collect(1:bn) 
        error("Groups must be consecutively numbered 1, 2, 3, ...")
    end
    ix = zeros(Int64, bn)
    iy = zeros(Int64, bn)
    j = 1

    for i = 1:bn 
        ix[i] = j
        iy[i] = j + bs[i] - 1
        j += bs[i]
    end
    pf = sqrt(bs)
    dfmax = maximum(group) + 1
    pmax = min(dfmax*1.2, maximum(group))
    pmax = round(Int64, pmax)
    gam = get_upperbound(x, ix, iy, bn)

    if isa(delta, Void)
        delta = 1
    end
    if delta < 0 
        error("delta must be non-negtive")
    end
    if length(pf) != bn 
        error("The size of group-lasso penalty factor must be same as the number of groups")
    end

    lambda_factor = get_lamfact(nobs, nvars)
    if isa(lambda, Void)
        if lambda_factor >= 1
            error("lambda_factor should be less than 1")
        end
        flmin = lambda_factor
        ulam = 1
    else
        ## flmin=1 if user define lambda
        flmin = 1
        if any(lambda .< 0)   
            error("lambdas should be non-negative")
        end
        ulam = reverse(sort(lambda))
        nlam = length(lambda)
    end
    if loss == "ls"
        gam = gam/nobs
        lasso_path = ls_f(bn, bs, ix, iy, gam, x, y, pf, dfmax, pmax, nlam, flmin, ulam, eps, maxit, intr)
    end
    return lasso_path
end



function ls_f(bn, bs, ix, iy, gam, x, y, pf, dfmax, pmax, nlam, flmin, ulam, eps, maxit, intr)  
    nobs, nvars = size(x)
    bignum = 9.9E30
    mnlam = 6
    mfl = 1.0E-6
    oldbeta = zeros(nvars+1)               # add 1 at the tail for b0
    beta = zeros(nvars, nlam)              # coefs for each lambda
    b0 = zeros(nlam) 
    b = zeros(nvars+1)                     # add 1 at the tail for b0
    nbeta = zeros(Int64, nlam)
    alam = zeros(nlam)               
    ga = zeros(bn)
    jxx = zeros(Int64, bn)
    idx = zeros(Int64, pmax)
    oidx = zeros(Int64, bn)

    ## ensure penalty factors are positive
    for i = 1:bn
        if pf[i] < 0.0
            pf[i] = 0.0
        end
    end
    ## initial set up
    dif = 9.9E30
    mnl = min(mnlam, nlam)
    al = 0.0
    r = y
    npass = 0
    ni = 0
    # --------- lambda loop ----------------------------
    if flmin < 1.0
        flmin = max(mfl, flmin)       
        alf = flmin^(1/(nlam - 1))
    end
    vl = (x'*r)/nobs
    
    for i = 1:bn
        u = vl[ix[i]:iy[i]]
        ga[i] = norm(u)       
    end

    for l = 1:nlam
        al0 = al
        if flmin >= 1.0   
            al = ulam[l]
        else
            if l > 2
                al = al*alf
            elseif l == 1 
                al = bignum
            elseif l == 2 
                al0 = 0.0              
                for i = 1:bn                    
                    if pf[i] > 0.0                       
                        al0 = max(al0, ga[i]/pf[i])
                    end
                end
                al = al0*alf
            end
        end  
        update_jxx!(jxx, ga, pf, al, al0, bn)
        # --------- outer loop ----------------------------
        persist = true
        while persist
            oldbeta[nvars+1] = b[nvars+1]
            if ni > 0
                for j = 1:ni               
                    g = idx[j]                  
                    oldbeta[ix[g]:iy[g]] = b[ix[g]:iy[g]]
                end 
            end
            # --middle loop-------------------------------------
            while true              
                npass += 1
                dif = 0.0               
                for i = 1:bn                   
                    if jxx[i] == 0                       
                        continue
                    end
                    start = ix[i]
                    ending = iy[i]                   
                    oldb = b[start:ending]
                    u = compute_u(x, r, b, start, ending, nobs, gam[i])
                    unorm = norm(u)
                    t = unorm - pf[i]*al                   
                    if t > 0.0 
                        b[start:ending] = compute_b(u, t, gam[i], unorm)
                    else
                        b[start:ending] = 0.0
                    end
                    dd = b[start:ending] - oldb                    
                    if found_nonzero(dd)                        
                        dif = max(dif, gam[i]^2*dot(dd, dd))
                        r = r - x[:, start:ending]*dd                      
                        if oidx[i] == 0                    
                            ni += 1                           
                            if ni > pmax
                                break
                            end
                            oidx[i] = ni
                            idx[ni] = i
                        end
                    end
                end                
                if intr                   
                    d = mean(r)                  
                    if d != 0.0
                        b[nvars+1] = b[nvars+1] + d                       
                        r = r - d                       
                        dif = max(dif, d^2)
                    end
                end
                if ni > pmax
                    break
                elseif dif < eps
                    break
                elseif npass > maxit
                    break
                end          
                # --inner loop----------------------
                while true                     
                    npass += 1
                    dif = 0.0      
                    for j = 1:ni
                        g = idx[j]                       
                        start = ix[g]                       
                        ending = iy[g]                       
                        oldb = b[start:ending]
                        u = compute_u(x, r, b, start, ending, nobs, gam[g])
                        unorm = norm(u)
                        t = unorm - pf[g]*al
                        if t > 0.0                           
                            b[start:ending] = compute_b(u, t, gam[g], unorm)
                        else                           
                            b[start:ending] = 0.0
                        end
                        dd = b[start:ending] - oldb
                        if found_nonzero(dd)
                            dif = max(dif, gam[g]^2*dot(dd, dd))
                            r = r - x[:, start:ending]*dd
                        end
                    end       
                    if intr 
                        d = mean(r)
                        if d != 0.0
                            b[nvars+1] = b[nvars+1] + d
                            r = r - d
                            dif = max(dif, d^2)
                        end
                    end
                    if dif < eps
                        break
                    elseif npass > maxit
                        break
                    end
                end
            end
            if ni > pmax
                break
            end
            # --- final check ------------------------
            persist = false
            max_gam = maximum(gam)
            if not_converged(b, oldbeta, max_gam, eps) 
                persist = true
            end
            if persist
                continue
            end
            vl = (x'*r)/nobs
            for i = 1:bn
                if jxx[i] == 1
                    continue
                end              
                u = vl[ix[i]:iy[i]]
                ga[i] = norm(u)
                if ga[i] > al*pf[i]
                    jxx[i] = 1
                    persist = true
                end 
            end
            if persist  
                continue
            end
            break
        end
        # -- final update variable and save results--
        if ni > pmax 
            break
        end
        if ni > 0 
            for j = 1:ni
                g = idx[j]
                beta[ix[g]:iy[g], l] = b[ix[g]:iy[g]]
            end
        end
        nbeta[l] = ni
        b0[l] = b[nvars+1]
        alam[l] = al
        nalam = l
        if l < mnl
            continue
        end
        me = count_nonzero_betas(beta, idx, ix, iy, l, ni)
        if me > dfmax
            break
        end
    end
    alam[1] = lamfix(alam)
    return LassoPath(alam, beta, b0, npass)
end
