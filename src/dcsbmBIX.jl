function EM(degrees,Ntot,B,links,maxCrs,initial)

    function logsumexp(array)
        array = vec(sortcols(array))
        for j = 1:length(array)
            array[end] - array[1] > log(10^(16.0)) ? shift!(array) : break # this cutoff must be smaller than cutoff in normalize_logprob: e.g. all elements are below cutoff.
        end
        if maximum(array) - minimum(array) > 700
            println("overflow or underflow are UNAVOIDABLE: restrict Crs to smaller values.")
        end
        array[1] + log(sum(exp(array - vec(ones(1,length(array)))*array[1])))
    end
    
    function normalize_logprob(array)
        arraylength = length(array)
        for r = 1:arraylength
            array[r] < log(10^(-8.0)) ? array[r] = log(10^(-8.0)) : continue
        end
        exp(array -logsumexp(array)*ones(1,arraylength))
    end
    
    function update_theta()
        ## MAP update
        (val, ind) = findmax(PSI,2)
        block = ind2sub((Ntot,B),vec(ind))[2]
        drmean = zeros(B,1)
        nr = zeros(B,1)
        for i = 1:Ntot
            drmean[block[i]] += degrees[i]
            nr[block[i]] += 1
        end
        drmean = drmean./nr
        theta = zeros(Ntot,1)
        for i = 1:Ntot
            theta[i] = degrees[i]/drmean[block[i]]
        end
        return theta
    end    
    
    function update_gr(PSI)
        gr = mean(PSI,1)
    end
    
    function update_h()
        thetamat = theta*ones(1,B)
        h = sum(thetamat.*PSI,1)*Crs/Ntot
    end

    function updatePSI(theta,Crs,h,gr,PSIcav,nb,PSI)
        for i = 1:Ntot
            logPSIi = logunnormalizedMessage(i,theta,Crs,h,gr,PSIcav,nb)
            PSI[i,:] = normalize_logprob(logPSIi)
        end
        return PSI
    end
        
    function logunnormalizedMessage(i,theta,Crs,h,gr,PSIcav,nb)
        logmessages = sum([log(theta[i]*theta[s]*PSIcav[(s,i)]*Crs) for s in nb[i]],1)[1]
        for r = 1:B
            if isnan(gr[r]) == true || gr[r] < 10^(-9.0)
                gr[r] = 1/Ntot
            else
                continue
            end
        end
        gr = gr/sum(gr)
        log(gr) - theta[i]*h + logmessages
    end
        
    function BP()
        conv = 0
        h = update_h()
        for k in randperm(size(links,1))
            (i,j) = tuple(links[k,:]...)
            logPSIi = logunnormalizedMessage(i,theta,Crs,h,gr,PSIcav,nb)
            PSIcav[(i,j)] = logPSIi - log(theta[j]*theta[i]*PSIcav[(j,i)]*Crs) # this is actually log(PSIcav[(i,j)])
            PSIcav[(i,j)] = normalize_logprob(PSIcav[(i,j)])

            hprev = theta[j]*PSI[j,:]*Crs/Ntot
            grprev = PSI[j,:]/Ntot
            prev = PSI[j,:]/Ntot
            logPSIj = logunnormalizedMessage(j,theta,Crs,h,gr,PSIcav,nb) # new PSI with new PSIcav
            PSI[j,:] = normalize_logprob(logPSIj)
            h += theta[j]*PSI[j,:]*Crs/Ntot - hprev
            gr += PSI[j,:]/Ntot - grprev
            conv += sum(abs(PSI[j,:]/Ntot - prev))
        end
        return (conv, PSI)
    end

    function update_Crs()
        Crsnew = zeros(B,B)
        for ij = 1:Ktot
            (i,j) = (links[ij,1],links[ij,2])
            ccrs = theta[i]*theta[j]*PSIcav[(i,j)]'*PSIcav[(j,i)].*Crs
            Crsnew += ccrs/sum(ccrs)
        end
        Crs = Crsnew ./ (Ntot*gr'*gr)
        for r = 1:B
        for s = 1:B
            if Crs[r,s] > maxCrs
                Crs[r,s] = maxCrs
                Crsfail = true
            end
        end
        end
        return (Crs, Crsfail)
    end
    
    function freeenergy()
        sumlogZi = 0
        for i = 1:Ntot
            sumlogZi += logsumexp(logunnormalizedMessage(i,theta,Crs,h,gr,PSIcav,nb))
        end
        
        Ltot = Int64(0.5*size(links,1))
        logZijs = zeros(Ltot)
        CVGPs = zeros(Ltot)
        CVGTs = zeros(Ltot)
        CVMAPs = zeros(Ltot)
        cnt = 0
        for ij = 1:Ktot
            (i,j) = (links[ij,1],links[ij,2])
            if i < j
                continue
            end
            cnt += 1
            logbiprob = Float32[]
            for s = 1:B
                for t=1:B
                    push!(logbiprob, log(PSIcav[(i,j)][s])+log(theta[i])+log(Crs[s,t])+log(theta[j])+log(PSIcav[(j,i)][t]))
                end
            end
            logZijs[cnt] = logsumexp(logbiprob') - log(Ntot)
            # average loglikelihood: 
            CVGPij = 0
            for s = 1:B
                for t=1:B
                    CVGPij += PSIcav[(i,j)][s]*(log(theta[i])+log(Crs[s,t])+log(theta[j])-log(Ntot))*PSIcav[(j,i)][t]
                end
            end
            CVGPs[cnt] = CVGPij
            # average likelihood with full prob. (training error)
            CVGTij = 0
            for s = 1:B
                for t=1:B
                    CVGTij += PSIcav[(i,j)][s]*theta[i]*Crs[s,t]*theta[j]*(log(theta[i])+log(Crs[s,t])+log(theta[j])-log(Ntot))*PSIcav[(j,i)][t]/(Ntot*exp(logZijs[cnt])) # factor N cancels at Crs and Zij
                end
            end
            CVGTs[cnt] = CVGTij
            # MAP estimate
            (MAPij, s) = findmax(PSIcav[(i,j)])
            (MAPji, t) = findmax(PSIcav[(j,i)])
            CVMAPs[cnt] = log(theta[i])+log(Crs[s,t])+log(theta[j]) - log(Ntot)
            #################
        end
        avedegree = gr*Crs*gr' # average degree is obtained by the same eqn. as the standard SBM.
        
        FE = -((sumlogZi - sum(logZijs))/Ntot + 0.5*avedegree[1] - 0.5*avedegree[1]*log(Ntot)) # - 0.5*avedegree[1]*log(Ntot) is due to the difference of the def. of Zij
        CVBayes = -sum(logZijs)/Ltot #+ 0.5*avedegree[1]
        CVGP = -sum(CVGPs)/Ltot #+ 0.5*avedegree[1]
        CVGT = -sum(CVGTs)/Ltot #+ 0.5*avedegree[1] 
        CVMAP = -sum(CVMAPs)/Ltot #+ 0.5*avedegree[1]
        varCVBayes = var(logZijs) # unbiased variance
        varCVGP = var(CVGPs)
        varCVGT = var(CVGTs)
        varCVMAP = var(CVMAPs)
        return (FE, CVBayes, CVGP, CVGT, CVMAP, varCVBayes, varCVGP, varCVGT, varCVMAP)
    end
        
    function Kmeans(V)
        itmax = 64
        threshold = 0.0001
        distprev = 10
        labels = zeros(Ntot,1)
        distance = zeros(Ntot,B)
        Vset = hcat(rand(1:B,Ntot,1),V) # (Ntot,1) random integers in 1:B & eigenvector matrix V
        for i = 1:itmax
            for k=1:B
                Vk = Vset[Vset[:,1].==k,2:B+1]
                distance[:,k] = sum((V - ones(Ntot,1)*mean(Vk,1)).^2,2)
            end
            labels = zeros(Ntot,1)
            (val,ind) = findmin(distance,2)
            for i = 1:size(ind,1)
                labels[i] = ind2sub(V,ind[i])[2]
            end
            Vset = hcat(labels,V)
            if abs(mean(distance) - distprev) < threshold
                break
            else
                distprev = mean(distance)
            end
            if i == itmax
                println("itmax is not enough...")
            end
        end
    return labels
    end

    function CrsbyKmeans()
        grkmeans = zeros(1,B)
        for k = 1:B
            length(find(labels[:,1] .== k)) == 0 ? grkmeans[1,k] = 1/Ntot : grkmeans[1,k] = length(find(labels[:,1] .== k))/Ntot
        end
        Crs = zeros(B,B)
        for ij = 1:Ktot
            (i,j) = (links[ij,1],links[ij,2])
            Crs[round(Int64,labels[i]),round(Int64,labels[j])] += 1
        end
        Crs = Crs./(Ntot*(grkmeans'*grkmeans))
        return (grkmeans, Crs)
    end
    
    
    #  initial state #######
    cnv = false
    fail = false
    Crsfail = false
    itrnum = 0
    Ktot = size(links,1)
    nb = [links[links[:,1].==lnode,2] for lnode = 1:Ntot]

    theta = ones(Ntot,1)
    #------initial Crs & gr
    if initial == "normalizedLaplacian"
        (eigenvalues,V) = Laplacian(links,Ntot,B)
        labels = Kmeans(V)
        (gr, Crs) = CrsbyKmeans()
    elseif initial == "random"
        RandMat = 0.001*rand(B,B) + 0.0001
        polarize = rand(1:B)
        polarize2 = rand(1:B)
        RandMat[polarize,polarize2] = 10
        Crs = RandMat + RandMat'
        gr = ones(B)'/B
    elseif initial == "uniformAssortative"
        Crs = 10*eye(B) + 0.01*ones(B)*ones(B)'
        gr = ones(B)'/B
    elseif initial == "uniformDisassortative"
        Crs = -9.9*eye(B) + 10*ones(B)*ones(B)'
        gr = ones(B)'/B
    end

    for r = 1:B
    for s = 1:B
        if isnan(Crs[r,s]) == true || abs(Crs[r,s]) == Inf
            println("Crs: bad init.!: $(Crs[r,s])")
            fail = true
        elseif Crs[r,s] < 10^(-6.0)
            Crs[r,s] = 10^(-6.0)
        end
    end
    end
    
    PSIcav = Dict()
    for ij = 1:size(links,1)
        PSIcav[tuple(links[ij,1],links[ij,2])] = rand(1,B) + 0.0001 # = 0.0001 to avoid too small values
        PSIcav[tuple(links[ij,1],links[ij,2])] = PSIcav[tuple(links[ij,1],links[ij,2])] / sum(PSIcav[tuple(links[ij,1],links[ij,2])])
    end
    h = zeros(1,B)
    PSI = zeros(Ntot,B)
    FE = 0
    CVBayes = 0
    CVGP = 0
    CVGT = 0
    CVMAP = 0
    varCVBayes = 0
    varCVGP = 0
    varCVGT = 0
    varCVMAP = 0
    ###################
    
    if fail == false        
        itrmax = 128
        BPconvthreshold = 0.0001#0.00000001
        PSI = updatePSI(theta,Crs,h,gr,PSIcav,nb,PSI)
        gr = update_gr(PSI)
        for itr = 1:itrmax
            (BPconv, PSI) = BP()
            (Crs, Crsfail) = update_Crs()
            theta = update_theta()
            if BPconv < BPconvthreshold
                #println("converged! ^_^: itr = $(itr)")
                cnv = true
                itrnum = itr
                break
            elseif itr == itrmax
                println("NOT converged... trying again...")
            end
        end
    
        h = update_h()
        gr = update_gr(PSI)
        (FE, CVBayes, CVGP, CVGT, CVMAP, varCVBayes, varCVGP, varCVGT, varCVMAP) = freeenergy()
    end # fail == false or not
    
    return (gr,Crs,PSI,FE,CVBayes,CVGP,CVGT,CVMAP,varCVBayes,varCVGP,varCVGT,varCVMAP,fail,Crsfail,cnv,itrnum)
end





################################################
function simplegraph(links)
    links = unique(links,1)
    links = vcat(links,hcat(links[:,2],links[:,1]))
    # remove self-loops
    boolean = trues(size(links,1))
    for i = 1:size(links,1)
        links[i,1] == links[i,2] ? boolean[i] = false : continue
    end
    links = links[boolean,:]
    
    return links
end

function degreesequence(nb,Ntot)
    degrees = zeros(Int64,Ntot,1)
    for i = 1:Ntot
        degrees[i]=length(nb[i])
    end
    return degrees
end

function Laplacian(links,Ntot,B)
    shift = 1000
    Ktot = size(links,1)
    A = sparse(links[:,1],links[:,2],ones(Ktot),Ntot,Ntot)
    degrees = sum(A,2)
    V = 1./sqrt(degrees)
    Dhalfinv = sparse(collect(1:Ntot),collect(1:Ntot),vec(V[:,1]))    
    L = sparse(eye(Ntot)) - Dhalfinv*A*Dhalfinv - sparse(shift*eye(Ntot))
    (eigenvalues,V) = eigs(L,nev=B,which=:LM)#,maxiter=1000,v0=rand(Ntot))
    eigenvalues = eigenvalues + shift
    return (eigenvalues,V)
end

function IPR(V)
    IPRmax = 0
    for r = 1:size(V,2)
        IPR = sum(V[:,r].^4)/(sum(V[:,r].^2))^2
        IPR > IPRmax ? IPRmax = IPR : continue
    end
    return IPRmax
end

function DFS(nb,root)
    visited = Int64[]
    stack = push!(Int64[],root)
    while !isempty(stack)
        node = pop!(stack)
        if node in visited
            continue
        else
            push!(visited,node)
            append!(stack,filter(x->!(x in visited), nb[node]))
            stack = unique(stack)
        end
    end
    return visited
end

function LinksConnected(links,Ntotinput,cc)
    Nblock = [Ntotinput]
    B = 1 # just 1 block: no blocks are given a priori for real data.
    # Nblock of connected component -----
    cc = sort(cc)
    ccNblock = zeros(Int64,B)
    t = 1
    defects = Int64[]
    for k = 1:B
        ndef = 0
        offset = sum(Nblock[1:k-1])
        for i = 1:Nblock[k]
            if t <= length(cc)
                if offset + i == cc[t]
                    t += 1
                    continue
                end
            end
            ndef += 1
            push!(defects,offset+i) # ndef = # of defect nodes in Nblock[k]
        end
        ccNblock[k] = Nblock[k] - ndef
    end
    #---------------------------------------------------
    
    # links of connected component ------------
    boolean = trues(size(links,1))
    for u = 1:size(links,1)
        links[u,1] in cc ? continue : boolean[u] = false
    end
    links = links[boolean,:]
    #----------------------------------------------------
    
    for u = 1:size(links,1)
        links[u,1] -= countnz(defects.<links[u,1])
        links[u,2] -= countnz(defects.<links[u,2])
    end
    
    Ntot = ccNblock[1]
    return (Ntot,links)
end
################################################












######################
# Keyboard inputs ####
doc = """

Usage:
  dcsbmBIX.jl [-h] <filename> [--Bmax=<Bmax>] [--init=partition...] [--samples=<samples>]
  dcsbmBIX.jl -h | --help
  dcsbmBIX.jl --version
  

Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --Bmax=<Bmax>             Maximum number of clusters. [default: 6]
  --init=partition...       Initial partition. [default: normalizedLaplacian]
  --samples=<samples>       Number of samples for each initial partition. [default: 10]
  
"""

using DocOpt  # import docopt function

args = docopt(doc, version=v"0.0.1")
strdataset = args["<filename>"]
Bmax = parse(Int64,args["--Bmax"])
initialconditions = args["--init"]
samples = parse(Int64,args["--samples"])

######################
######################












#strdataset = "graph_zachary.txt"
Ltotinput = countlines(open( strdataset, "r" ))
fpmeta = open("metadata.txt","w")
write(fpmeta, "dataset: $(strdataset)\n\n")
open( strdataset, "r" ) do fp
    cnt = 0
    Ntotinput = 0
    links = zeros(Int64,Ltotinput,2)
    for line in eachline( fp )
    cnt += 1
        line = rstrip(line, '\n')
        u = split(line, " ")
#        u = split(line, "\t")
        u1 = parse(Int64,u[1]) # convert string to number
        u2 = parse(Int64,u[2])
        links[cnt,1] = u1
        links[cnt,2] = u2
        Ntotinput < max(u1,u2) ? Ntotinput = Int64(max(u1,u2)) : continue
    end
    links = simplegraph(links)
    write(fpmeta, "number of vertices (input): $(Ntotinput)\n")
    write(fpmeta, "number of edges (input, converted to simple graph): $(Int64(0.5*size(links,1)))\n\n")

    Nthreshold = round(Ntotinput/2)
    IPRthreshold = 10/Ntotinput
    #Bmax = 3
    #samples = 10
    #initialconditions = ["normalizedLaplacian","random","uniformAssortative","uniformDisassortative"]
    write(fpmeta, "initial conditions: $(join(initialconditions, ", "))\n")
    write(fpmeta, "numbmer of samples for each initial condition: $(samples)\n\n")
    
    nb = [links[links[:,1].==lnode,2] for lnode = 1:Ntotinput]
    cc = DFS(nb,1) # Assume node 1 belongs to the connected component
    println("connected component identified...")
    (Ntot,links) = LinksConnected(links,Ntotinput,cc)
    println("nodes & links updated... : Ntot = $(Ntot) 2L = $(size(links,1))")
    if Ntot < Nthresholddcsbm
        println("This is not a giant component... try again.")
    end
    Ltot = Int64(0.5*size(links,1))
    assignment = zeros(Int64,Ntot,Bmax) # 1st column is the node label, but B starts from 2
    assignment[:,1] = collect(1:Ntot)
    write(fpmeta, "numbmer of vertices (giant component): $(Ntot)\n")
    write(fpmeta, "numbmer of edges (giant component): $(Ltot)\n\n")

    strFE = "FE_dcsbm.dat"
    strCVBayes = "CVBayesPrediction_dcsbm.dat"
    strCVGP = "CVGibbsPrediction_dcsbm.dat"
    strCVGT = "CVGibbsTraining_dcsbm.dat"
    strCVMAP = "CVMAP_dcsbm.dat"
    
    strPartition = "assignment.dat"
    strParameters = "hyperparameters.txt"
    
    fpFE = open(strFE,"w")
    fpCVBayes = open(strCVBayes,"w")
    fpCVGP = open(strCVGP,"w")
    fpCVGT = open(strCVGT,"w")
    fpCVMAP = open(strCVMAP,"w")
    fpPartition = open(strPartition,"w")
    fpParameters = open(strParameters,"w")

    for B = 2:Bmax
        println("B = $(B)")
        FEs = zeros(samples,length(initialconditions))
        CVBayess = zeros(samples,length(initialconditions))
        CVGPs = zeros(samples,length(initialconditions))
        CVGTs = zeros(samples,length(initialconditions))
        CVMAPs = zeros(samples,length(initialconditions))
        varCVBayess = zeros(samples,length(initialconditions))
        varCVGPs = zeros(samples,length(initialconditions))
        varCVGTs = zeros(samples,length(initialconditions))
        varCVMAPs = zeros(samples,length(initialconditions))
        FEmin = 0
        itrnumopt = 0
        PSIopt = zeros(Ntot,B)
        gropt = zeros(B,1)
        Crsopt = zeros(B,B)
        
        ## upper bound of Crs ===================
        degrees = degreesequence(nb,Ntot)
        ### If Crs is too large, overflow occurs at logsumexp.
        maxCrs = Ntot
        #println("maxCrs = $(maxCrs)")
        ##=================================
        
        for init = 1:length(initialconditions)
            println(initialconditions[init])
            sm = 0
            giveup = 0
            overflow = 0
            while sm < samples
                Crsfail = false
                (gr,Crs,PSI,FE,CVBayes,CVGP,CVGT,CVMAP,varCVBayes,varCVGP,varCVGT,varCVMAP,fail,Crsfail,cnv,itrnum) = EM(degrees,Ntot,B,links,maxCrs,initialconditions[init])
                if cnv == false
                    continue
                end
                if isnan(maximum(PSI)) == true || maximum(PSI) == Inf || abs(FE) == Inf || abs(CVGP) == Inf || abs(CVMAP) == Inf
                    overflow += 1
                    if overflow > 10
                        println("overflow occurs too often...")
                        break
                    else
                        #println("overflow...")
                        continue
                    end
                end
                if Crsfail == true # it is ok that Crs goes inf. It should be inf when there is an empty cluster.
                    #println("Crs reached its upper bound...")
                end
                if fail == true
                    giveup += 1
                    if giveup > 10
                        println("too many fails: give up!")
                        break
                    else
                        continue
                    end
                else
                    sm += 1
                end

                println("sample = $(sm)")        
                FEs[sm,init] = FE
                CVBayess[sm,init] = CVBayes
                CVGPs[sm,init] = CVGP
                CVGTs[sm,init] = CVGT
                CVMAPs[sm,init] = CVMAP
                varCVBayess[sm,init] = varCVBayes
                varCVGPs[sm,init] = varCVGP
                varCVGTs[sm,init] = varCVGT
                varCVMAPs[sm,init] = varCVMAP                
                if FE < FEmin
                    FEmin = FE
                    itrnumopt = itrnum
                    PSIopt = PSI
                    Crsopt = Crs
                    gropt = gr
                end
            end
        end # initial conditions

        # standard errors of CVs
        SECVBayes = sqrt(varCVBayess[findmin(CVBayess)[2]]/Ltot)
        SECVGP = sqrt(varCVGPs[findmin(CVGPs)[2]]/Ltot)
        SECVGT = sqrt(varCVGTs[findmin(CVGTs)[2]]/Ltot)
        SECVMAP = sqrt(varCVMAPs[findmin(CVMAPs)[2]]/Ltot)
        
        println(string(minimum(FEs)))
        println(string(minimum(CVBayess)))
        println(string(minimum(CVGPs)))
        println(string(minimum(CVGTs)))
        println(string(minimum(CVMAPs)))
#        println(string(SECVBayes))
#        println(string(SECVGP))
#        println(string(SECVGT))
#        println(string(SECVMAP))
        write(fpFE, string(B)*" "*string(minimum(FEs))*"\n")
        write(fpCVBayes, string(B)*" "*string(minimum(CVBayess))*" "*string(SECVBayes)*"\n")
        write(fpCVGP, string(B)*" "*string(minimum(CVGPs))*" "*string(SECVGP)*"\n")
        write(fpCVGT, string(B)*" "*string(minimum(CVGTs))*" "*string(SECVGT)*"\n")
        write(fpCVMAP, string(B)*" "*string(minimum(CVMAPs))*" "*string(SECVMAP)*"\n")

        (val, ind) = findmax(PSIopt,2)
        block = ind2sub((Ntot,B),vec(ind))[2]
        assignment[:,B] = block # 1st column is the node label, but B starts from 2
        actualB = length(unique(vec(block)))
        write(fpmeta, "q = $(B): actual q = $(actualB), number of iteration = $(itrnumopt)\n")
#        println("PSIopt = $(PSIopt)")
        write(fpParameters, "q = $(B): \n")
        write(fpParameters, "fractions of block size = \n")
        write(fpParameters, "$(gropt)\n")
        write(fpParameters, "rescaled affinity matrix C = \n")
        write(fpParameters, "$(Crsopt)\n")
    end # B-loop
    for i = 1:Ntot
        write(fpPartition, join(assignment[i,:], " ")*"\n")
    end

    close(fpFE)
    close(fpCVBayes)
    close(fpCVGP)
    close(fpCVGT)
    close(fpCVMAP)
    close(fpPartition)
    close(fpParameters)

end
close(fpmeta)