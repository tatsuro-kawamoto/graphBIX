using PyPlot
using Random
using SparseArrays
using Statistics
using LinearAlgebra
using Arpack

function eye(n)
    Diagonal{Float64}(I, n)
end

function Cartesian2ind(A, Cartesians)
    LI = LinearIndices(A)
    return LI[Cartesians]
end

function ind2sub((a,b), i)
    i2s = CartesianIndices(zeros(Int(a),Int(b)))
    return i2s[i]
end

function sub2ind((a,b),i,j)
    A = zeros(a,b)
    ind = LinearIndices(A)[i,j]
    return ind
end

function EM(Ntot,B,links,maxCrs,initial,degreecorrection,itrmax,priorlearning,learningrate)

    function logsumexp(array)
        array = vec(sortslices(array, dims=2))
        maxval = maximum(array)
        for k = 1:length(array)
            maxval - array[k] > 500 ? array[k] = maxval - 500 : continue
        end
        array[1] + log.(sum(exp.(array - vec(ones(1,length(array)))*array[1])))
    end
    
    function normalize_logprob(array)
        arraylength = length(array)
        for r = 1:arraylength
            array[r] < log.(10^(-8.0)) ? array[r] = log.(10^(-8.0)) : continue
        end
        exp.(array - logsumexp(array)*ones(1,arraylength))
    end
    
    function update_theta()
        ## MAP update
        (val, ind) = findmax(PSI,dims=2)
        ind = Cartesian2ind(PSI,ind)
        block = [ind2sub((Ntot,B),index)[2] for index in ind]
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
        gr = mean(PSI,dims=1)
    end
    
    function update_h()
        thetamat = theta*ones(1,B)
        h = sum(thetamat.*PSI,dims=1)*Crs/Ntot
    end

    function updatePSI(theta,Crs,h,gr,PSIcav,nb,PSI)
        for i = 1:Ntot
            logPSIi = logunnormalizedMessage(i)
            PSI[i,:] = normalize_logprob(logPSIi)
        end
        return PSI
    end
        
    function logunnormalizedMessage(i)
        logmessages = transpose(zeros(length(gr)))
        for s in nb[i]
            indsi = sub2ind((Ntot,Ntot),s,i)
            logmessages += log.(theta[i]*theta[s]*PSIcav[indsi]*Crs)
        end
        for r = 1:B
            if isnan(gr[r]) == true || gr[r] < 10^(-9.0)
                gr[r] = 1/Ntot
            else
                continue
            end
        end
        gr = gr/sum(gr)
        log.(gr) - theta[i]*h + logmessages
    end
        
    function BP()
        conv = 0
        h = update_h()
        for i in randperm(Ntot)
            logPSIi = logunnormalizedMessage(i)
            for j in nb[i]
                indij = sub2ind((Ntot,Ntot),i,j)
                indji = sub2ind((Ntot,Ntot),j,i)
                PSIcav[indij] = logPSIi - log.(theta[j]*theta[i]*PSIcav[indji]*Crs) # this is actually log.(PSIcav[indij]) 
                PSIcav[indij] = normalize_logprob(PSIcav[indij])
            end
            
            hprev = theta[i]*PSI[i,:]'*Crs/Ntot
			if priorlearning == true
            	grprev = PSI[i,:]'/Ntot
			end
            prev = PSI[i,:]'/Ntot
            logPSIi = logunnormalizedMessage(i) # new PSI with new PSIcav
            PSI[i,:] = normalize_logprob(logPSIi)
            h += theta[i]*PSI[i,:]'*Crs/Ntot - hprev
			if priorlearning == true
				gr += PSI[i,:]'/Ntot - grprev # learningrate omitted here: learning rate is much slower than Crs if learningrate is included.
			end
            conv += sum(abs.(PSI[i,:]'/Ntot - prev))
        end
        return (conv, PSI)
    end

    function update_Crs()
        Crsnew = zeros(B,B)
        for ij = 1:Ktot
            (i,j) = (links[ij,1],links[ij,2])
            indij = sub2ind((Ntot,Ntot),i,j)
            indji = sub2ind((Ntot,Ntot),j,i)
            ccrs = theta[i]*theta[j]*PSIcav[indij]'*PSIcav[indji].*Crs
            Crsnew += ccrs/sum(ccrs)
        end
        Crs = learningrate*Crsnew./(Ntot*gr'*gr) + (1-learningrate)*Crs
        for r = 1:B
        for s = 1:B
            if Crs[r,s] > maxCrs
                Crs[r,s] = maxCrs
            elseif Crs[r,s] < 0
                Crs[r,s] = 0
            end
        end
        end
        return Crs
    end
    
    function freeenergy()
        sumlogZi = 0
        for i = 1:Ntot
            sumlogZi += logsumexp(logunnormalizedMessage(i))
        end
        
        Ltot = Int64(0.5*size(links,1))
        logZijs = zeros(Ltot)
        CVGPs = zeros(Ltot)
        CVGTs = zeros(Ltot)
        CVMAPs = zeros(Ltot)
        cnt = 0
        for ij = 1:Ktot
            (i,j) = (links[ij,1],links[ij,2])
            indij = sub2ind((Ntot,Ntot),i,j)
            indji = sub2ind((Ntot,Ntot),j,i)
            if i < j
                continue
            end
            cnt += 1
            logbiprob = Float32[]
            for s = 1:B
                for t=1:B
                    push!(logbiprob, log.(PSIcav[indij][s])+log.(theta[i])+log.(Crs[s,t])+log.(theta[j])+log.(PSIcav[indji][t]))
                end
            end
            logZijs[cnt] = logsumexp(logbiprob') - log.(Ntot)
            # average loglikelihood: 
            CVGPij = 0
            for s = 1:B
                for t=1:B
                    CVGPij += PSIcav[indij][s]*(log.(theta[i])+log.(Crs[s,t])+log.(theta[j])-log.(Ntot))*PSIcav[indji][t]
                end
            end
            CVGPs[cnt] = CVGPij
            # average likelihood with full prob. (training error)
            CVGTij = 0
            for s = 1:B
                for t=1:B
                    CVGTij += PSIcav[indij][s]*theta[i]*Crs[s,t]*theta[j]*(log.(theta[i])+log.(Crs[s,t])+log.(theta[j])-log.(Ntot))*PSIcav[indji][t]/(Ntot*exp.(logZijs[cnt])) # factor N cancels at Crs and Zij
                end
            end
            CVGTs[cnt] = CVGTij
            # MAP estimate
            (MAPij, s) = findmax(PSIcav[indij])
            (MAPji, t) = findmax(PSIcav[indji])
            s = Cartesian2ind(PSIcav[indij], s)
            t = Cartesian2ind(PSIcav[indij], t)
            CVMAPs[cnt] = log.(theta[i])+log.(Crs[s,t])+log.(theta[j]) - log.(Ntot)
            #################
        end
        avedegree = gr*Crs*gr' # average degree is obtained by the same eqn. as the standard SBM.
        
        FE = -((sumlogZi - sum(logZijs))/Ntot + 0.5*avedegree[1])
        CVBayes = 1-sum(logZijs)/Ltot
        CVGP = 1-sum(CVGPs)/Ltot
        CVGT = 1-sum(CVGTs)/Ltot
        CVMAP = 1-sum(CVMAPs)/Ltot
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
                distance[:,k] = sum((V - ones(Ntot,1)*mean(Vk,dims=1)).^2,dims=2)
            end
            labels = zeros(Ntot,1)
            (val,ind) = findmin(distance,dims=2)
            ind = Cartesian2ind(distance,ind)
            for i = 1:size(ind,1)
                labels[i] = ind2sub(size(V),ind[i])[2]
            end
            Vset = hcat(labels,V)
            if abs.(mean(distance) - distprev) < threshold
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
            length(findall(labels[:,1] .== k)) == 0 ? grkmeans[1,k] = 1/Ntot : grkmeans[1,k] = length(findall(labels[:,1] .== k))/Ntot
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
    itrnum = 0
    Ktot = size(links,1)
    inds = sub2ind((Ntot,Ntot),links[:,1],links[:,2])
    A = sparse(links[:,1],links[:,2],ones(Ktot),Ntot,Ntot)
    # neighboring vertices --------
    nb = Array[]
    row = rowvals(A)
    for j = 1:Ntot
        push!(nb,Int64[row[i] for i in nzrange(A,j)])
    end
    # ----------------------------------
    degrees = sum(A,dims=2)

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
        Crs = 20*eye(B) + 0.01*ones(B)*ones(B)'
        gr = ones(B)'/B
    elseif initial == "uniformDisassortative"
        Crs = -9.9*eye(B) + 10*ones(B)*ones(B)'
        gr = ones(B)'/B
    elseif initial == "Bipartite"
        halfB = round(Int64,B/2)
        layer1 = [0 1;1 0]
        layer2 = 20*eye(halfB) + 0.01*ones(halfB)*ones(halfB)'
        Crs = kron(layer1,layer2)
        gr = ones(B)'/B
    end

    if priorlearning == false
        gr = ones(B)'/B
	end
	
    for r = 1:B
    for s = 1:B
        if isnan(Crs[r,s]) == true || abs.(Crs[r,s]) == Inf
            println("Crs: bad init.!: $(Crs[r,s])")
            fail = true
        elseif Crs[r,s] < 10^(-6.0)
            Crs[r,s] = 10^(-6.0)
        end
    end
    end

    PSIcav = Dict()
    for ind in inds
        PSIcav[ind] = rand(1,B)
        PSIcav[ind] = PSIcav[ind]/sum(PSIcav[ind])
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
        #itrmax = 128
        BPconvthreshold = 0.000001#*Ntot
        PSI = updatePSI(theta,Crs,h,gr,PSIcav,nb,PSI)
        gr = update_gr(PSI)
        for itr = 1:itrmax
            (BPconv, PSI) = BP()
            Crs = update_Crs()
            if degreecorrection == true
                theta = update_theta()
            end
            if BPconv < BPconvthreshold
                #println("converged! ^_^: itr = $(itr)")
				println(". itr = $(itr)")
                cnv = true
                itrnum = itr
                break
            elseif itr == itrmax
                if isnan(maximum(gr)) == true || isnan(maximum(Crs)) == true || maximum(PSI) == Inf
                    println("overflow : Use different initial partition or modify maxCrs.")
                else
                    println("NOT converged: residual = $(Float16(BPconv))")
                end
            end
			print(".")
        end
    
        h = update_h()
        gr = update_gr(PSI)
        (FE, CVBayes, CVGP, CVGT, CVMAP, varCVBayes, varCVGP, varCVGT, varCVMAP) = freeenergy()
    end # fail == false or not
    
    return (gr,Crs,PSI,FE,CVBayes,CVGP,CVGT,CVMAP,varCVBayes,varCVGP,varCVGT,varCVMAP,fail,cnv,itrnum)
end



###############################
# Spectral 
###############################
function Laplacian(links,Ntot,B)
    shift = 1000
    Ktot = size(links,1)
    A = sparse(links[:,1],links[:,2],ones(Ktot),Ntot,Ntot)
    degrees = sum(A,dims=2)
    V = ones(size(degrees,1))/sqrt.(degrees)
    Dhalfinv = sparse(collect(1:Ntot),collect(1:Ntot),vec(V[:,1]))    
    L = sparse(eye(Ntot)) - Dhalfinv*A*Dhalfinv - sparse(shift*eye(Ntot))
    (eigenvalues,V) = eigs(L,nev=B,which=:LM)#,maxiter=1000,v0=rand(Ntot))
    eigenvalues = eigenvalues + shift*ones(size(eigenvalues,1))
    return (eigenvalues,V)
end

function nonbacktracking(links,degrees,Ntot,Bmax)
    Ktot = size(links,1)
    A = sparse(Ntot+links[:,1],Ntot+links[:,2],ones(Ktot),2*Ntot,2*Ntot)
    NB12block = sparse(collect(1:Ntot),collect(Ntot+1:2*Ntot),degrees[:,1]-ones(Ntot),2*Ntot,2*Ntot)
    NB21block = sparse(collect(Ntot+1:2*Ntot),collect(1:Ntot),-ones(Ntot),2*Ntot,2*Ntot)
    NBT = A + NB12block + NB21block
    (eigenvalues,V) = eigs(NBT,nev=Bmax,which=:LR)
    return (eigenvalues,V)
end
###############################
# END: Spectral
###############################


###############################
# Miscellaneous Functions
###############################
function degreesequence(links,Ntot)
    Ktot = size(links,1)
    A = sparse(links[:,1],links[:,2],ones(Ktot),Ntot,Ntot)
    degrees = sum(A,dims=2)
    return degrees
end

function simplegraph(links)
    links = vcat(links,hcat(links[:,2],links[:,1]))
    links = unique(links,dims=1)
    # remove self-loops
    boolean = trues(size(links,1))
    for i = 1:size(links,1)
        links[i,1] == links[i,2] ? boolean[i] = false : continue
    end
    links = links[boolean,:]
    
    return links
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
    cc = sort(cc)
    t = 1
    defects = Int64[]
    ndef = 0
    for i = 1:Ntotinput
        if t <= length(cc)
            if i == cc[t]
                t += 1
                continue
            end
        end
        ndef += 1
        push!(defects,i) # ndef = # of defect nodes
    end
    Ntot = Ntotinput - ndef
    #---------------------------------------------------
    
    # links of connected component ------------
    boolean = trues(size(links,1))
    for u = 1:size(links,1)
        links[u,1] in cc || links[u,2] in cc ? continue : boolean[u] = false # For undirected(bidirected) case, links[u,1] in cc is enough.
    end
    links = links[boolean,:]
    #----------------------------------------------------
    
    for u = 1:size(links,1)
        links[u,1] -= count(defects.<links[u,1])
        links[u,2] -= count(defects.<links[u,2])
    end
    
    return (Ntot,links)
end

function UniformGraphError(links,Ntot,degreecorrection)
    Ktot = size(links,1)
    A = sparse(links[:,1],links[:,2],ones(Ktot),Ntot,Ntot)
    degrees = sum(A,dims=2)
	if degreecorrection == true
		nullerror = 0
		for di in degrees
			nullerror += di*log.(di)
		end
		nullerror = 1 + log.(Ktot) - 2*nullerror/Ktot 
	else
		nullerror = 1 - log.(Ktot/(Ntot*Ntot-1))
	end
	
	return nullerror
end	

function AssignmentProbs(B,nodeids,PSIopt,dataset)
    strProb = "output_$(dataset)/ProbDistribution_partition$(B)_$(dataset).txt"
    fpProb = open(strProb,"w")

    Ntot = size(PSIopt,1)
    for k = 1:Ntot
        PSIoptstr = ""
        for i = 1:B
            PSIoptstr *= " $(PSIopt[k,i])"
        end
        write(fpProb, "$(nodeids[k])$(PSIoptstr)\n")
    end
    
    close(fpProb)
end
###############################
# END: Miscellaneous Functions
###############################




###############################
# Functions for plots
###############################
function sortblock(block,blockprev)
    Bnow = maximum(block)
    Bprev = maximum(blockprev)
    if Bprev > 1
        blbl = hcat(block,blockprev)
        M = zeros(Int64,Bnow,Bprev)
        for i = 1:size(block,1)
            M[blbl[i,1],blbl[i,2]] += 1
        end
        (val,ind) = findmax(M,dims=2)
        ind = Cartesian2ind(M,ind)
        maxind = [ind2sub((Bnow,Bprev),index)[2] for index in ind]
        varM = var(M,dims=2)
        c = collect(1:Bnow)
        MM = sortslices(hcat(maxind,varM,c), dims=1, by=x->(x[1],-x[2]))
        #MM = sortrows(hcat(maxind,varM,c), by=x->(x[1],-x[2]))
        relabel = zeros(Bnow,1)
        for k = 1:Bnow
            relabel[Int64(MM[k,end])] = k
        end
        for i = 1:size(block,1)
            block[i] = relabel[block[i]]
        end
    end
    
    return block
end

# Alluvial diagram: smap file generator //////////////////////////////
function AlluvialDiagram(B,block,maxPSIval,dataset)
    significant = 0.7
    Ntot = size(maxPSIval,1)
    stralluvial = "output_$(dataset)/partition$(B)_$(dataset).smap"
    fpalluvial = open(stralluvial,"w")
    
    modules = zeros(Int64,Ntot,3) # [node num., block label, significance]
    modules[:,1] = collect(1:Ntot)
    modules[:,2] = block
    #modules = sortrows(modules, by=x->x[2])
    modules = sortslices(modules, dims=1, by=x->x[2])
    
    trueB = 0
    labelprev = 0
    moduleindices = AbstractString[]
    for k = 1:Ntot
        if modules[k,2] != labelprev
            trueB += 1
            labelprev = modules[k,2] # modules[k,2] may not be successive, e.g. 1 1 3 3 4 4 ...
            push!(moduleindices, "$(modules[k,1]),...")
        end
        modules[k,2] = trueB
        maxPSIval[modules[k,1]]>significant ? modules[k,3] = 1 : continue        
    end

    write(fpalluvial, "*Undirected\n")
    write(fpalluvial, "*Modules $(trueB)\n")
    for md = 1:trueB
        write(fpalluvial, """$(md) "$(moduleindices[md])" $(Float16(0.1/trueB)) $(Float16(0.01/trueB))\n""")
    end
    write(fpalluvial, "*Insignificants 0\n")
    write(fpalluvial, "*Nodes $(Ntot)\n")
    sublabel = 0
    labelprev = 0
    for k = 1:Ntot
        modules[k,2] == labelprev ? sublabel += 1 : sublabel = 1
        labelprev = modules[k,2]
        if modules[k,3] == 1
            write(fpalluvial, """$(modules[k,2]):$(sublabel) "Node $(modules[k,1])" $(Float16(0.1/Ntot))\n""")
        else
            write(fpalluvial, """$(modules[k,2]);$(sublabel) "Node $(modules[k,1])" $(Float16(0.1/Ntot))\n""")
        end
    end
    write(fpalluvial, "*Links $(trueB^2)\n")
    for s = 1:trueB
    for t = 1:trueB
            write(fpalluvial, "$(s) $(t) $(Float16(0.1/trueB^2))\n")
    end
    end
    
    close(fpalluvial)
end


function PlotAssessments(x,y0,y1,y1error,y2,y2error,y3,y3error,y4,y4error,dataset)
    Bmin = minimum(x)
    Bmax = maximum(x)
    
    fig = figure("plot_$(dataset)",figsize=(9,3))
    subplots_adjust(hspace=0.5,wspace=0.3)
#    subplots_adjust(hspace=0.1) # Set the vertical spacing between axes
	font1 = Dict("color"=>"k","size"=>14)

    # 1st fig --------
    subplot(121)
    p = plot(x,y0,color="0.4",linestyle="-",marker="8",markeredgecolor="None",markersize=8,label="Bethe free energy")
    ax = gca()
    ylabel("Bethe free energy",fontdict=font1)
    setp(ax[:get_yticklabels](),color="0.2") # Y Axis font formatting
    xlabel(L"Number of clusters $\, \mathit{q}$",fontdict=font1)
    ax[:set_xlim](Bmin-0.2,Bmax+0.2)
    Mx = matplotlib[:ticker][:MultipleLocator](1) # Define interval
    ax[:xaxis][:set_major_locator](Mx) # Set interval 

    # 2nd fig --------
    subplot(122)
    p = plot(x,y2,color="#27AE60",linestyle="-",linewidth=2,marker="^",markersize=8,markerfacecolor="white",markeredgecolor="#27AE60",markeredgewidth=1.5,zorder=10,label="EGP")
    p = fill_between(x,vec(y2-y2error),vec(y2+y2error),color="#2ECC71",alpha=0.7,edgecolor="None",zorder=9)
    p = plot(x,y1,color="#C0392B",linestyle="-",linewidth=2,marker="o",markersize=7,markerfacecolor="white",markeredgecolor="#C0392B",markeredgewidth=1.5,zorder=8,label="EBayes")
    p = fill_between(x,vec(y1-y1error),vec(y1+y1error),color="#E74C3C",alpha=0.7,edgecolor="None",zorder=7)
    p = plot(x,y3,color="#2980B9",linestyle="-",linewidth=2,marker="D",markersize=7,markerfacecolor="white",markeredgecolor="#2980B9",markeredgewidth=1.5,zorder=6,label="EGT")
    p = fill_between(x,vec(y3-y3error),vec(y3+y3error),color="#3498DB",alpha=0.7,edgecolor="None",zorder=5)
    p = plot(x,y4,color="#F39C12",linestyle="-",linewidth=2,marker="s",markersize=7,markerfacecolor="white",markeredgecolor="#F39C12",markeredgewidth=1.5,zorder=4,label="EMAP")
    p = fill_between(x,vec(y4-y4error),vec(y4+y4error),color="#F1C40F",alpha=0.7,edgecolor="None",zorder=3)
    ax = gca()
    ylabel("Prediction/training error",fontdict=font1)
    setp(ax[:get_yticklabels](),color="k") # Y Axis font formatting
    ax[:set_xlim](Bmin-0.2,Bmax+0.2)
    xlabel(L"Number of clusters $\, \mathit{q}$",fontdict=font1)
    Mx = matplotlib[:ticker][:MultipleLocator](1) # Define interval
    ax[:xaxis][:set_major_locator](Mx) # Set interval 

    axis("tight")

#    fig[:canvas][:draw]() # Update the figure
    suptitle(dataset,fontdict=font1)
    savefig("output_$(dataset)/assessment_$(dataset).pdf",bbox_inches="tight",pad_inches=0.1)
end


function grCrsMatrices(grDictbb,CrsDictbb)
    B = length(grDictbb)
    Binmax = 100
    grCrsMatrix = zeros(Binmax,Binmax)
    binsizes = floor.(Int64,Binmax*grDictbb)
    offsetr = 0
    for r = 1:B
        offsets = 0
        for s = 1:B
            if r == B && s < B
                grCrsMatrix[offsetr+1:Binmax,offsets+1:offsets+binsizes[s]] = fill(CrsDictbb[r,s],(Binmax-offsetr,binsizes[s]))
            elseif r < B && s == B
                grCrsMatrix[offsetr+1:offsetr+binsizes[r],offsets+1:Binmax] = fill(CrsDictbb[r,s], (binsizes[r],Binmax-offsets))
            elseif r == B && s == B
                grCrsMatrix[offsetr+1:Binmax,offsets+1:Binmax] = fill(CrsDictbb[r,s], (Binmax-offsetr,Binmax-offsets))
            else
                grCrsMatrix[offsetr+1:offsetr+binsizes[r],offsets+1:offsets+binsizes[s]] = fill(CrsDictbb[r,s], (binsizes[r],binsizes[s]))
            end
            offsets += binsizes[s]
        end
        offsetr += binsizes[r]
    end
    return grCrsMatrix
end


function PlotAffinityMatrix(Bsize,grDict,CrsDict,dataset)
    plt = PyPlot
    fig=plt.figure(figsize=(2*Bsize, 2))
    for bb = 1:Bsize
        ax = fig[:add_subplot](1,Bsize,bb)
        grCrsMatrix = grCrsMatrices(grDict[bb],CrsDict[bb])
        plt.imshow(grCrsMatrix,interpolation="nearest",cmap=ColorMap("Blues"))
        setp(ax[:get_xticklabels](),visible=false) # Disable x tick labels
        setp(ax[:get_yticklabels](),visible=false) # Disable y tick labels
    end
#    fig[:canvas][:draw]() # Update the figure
    savefig("output_$(dataset)/structures_$(dataset).pdf",bbox_inches="tight",pad_inches=0.1)
end
###############################
# END: Functions for plots
###############################




######################
# Keyboard inputs ####
doc = """

Usage:
  sbm.jl <filename> [--dataset=<dataset>] [--dc=<dc>] [--q=Blist] [--init=partition...] [--initnum=<samples>] [--itrmax=<itrmax>] [--prior=priorlearning] [--learning_rate=<learningrate>] [--alluvial=alluvial]
  sbm.jl -h | --help
  sbm.jl --version
  

Options:
  -h --help                 		Show this screen.
  --version                 		Show version.
  --dataset							Name of the dataset [default: "nothing"]
  --q=Blist                 		List of number of clusters. [default: 2:3]
  --init=partition...       		Initial partition. [default: normalizedLaplacian]
  --initnum=<samples>       		Number of initial states. [default: 3]
  --dc=<dc>                 		Degree correction. [default: true]
  --itrmax=<itrmax>       			Maximum number of BP iteration. [default: 256]
  --prior=priorlearning     		Learn cluster sizes. [default: true]
  --learning_rate=<learningrate>    Learning rate. [default: 0.3]
  --alluvial=alluvial       		Generate smap files for the alluvial diagram. [default: false]

========================
Bayesian inference for the stochastic block model using EM algorithm + belief propagation with the leave-one-out cross-validation.

+ Inference for the degree-corrected SBM by default. Set `--dc=false` for the standard SBM.
+ Convergence criterion = 10^(-6) by default.
+ Note that the result varies depending on the initial values of the hyperparameters (cluster size & affinity matrix). 
To be cautious, try multiple `--init` and large `initnum`. 
To select the initial values of the hyperparameters, specify `--init`. 
The options for `--init` are 
	- normalizedLaplacian: Spectral clustering with the normalized Laplacian + k-means algorithm. 
	- random: Equal size clusters & randomly polarized affinity matrix. 
	- uniformAssortative: Equal size clusters & equal size assortative block structure. 
	- uniformDisassortative: Equal size clusters & equal size disassortative block structure. 

Examples: 
Inference of `edgelist.txt` for the standard SBM with q = 2 to 6:
julia sbm.jl edgelist.txt --dc=false --q=2:6 --init={normalizedLaplacian,random} --initnum=5

Inference of `edgelist.txt` for the degree-corrected SBM with q = 2, 4, and 6:
julia sbm.jl edgelist.txt --dc=true --q=2,4,6 --init={normalizedLaplacian,random} --initnum=5
or 
julia sbm.jl edgelist.txt --dc=true --q=2:2:6 --init={normalizedLaplacian,random} --initnum=5
========================
Author: Tatsuro Kawamoto: kawamoto.tatsuro@gmail.com
Reference: Tatsuro Kawamoto and Yoshiyuki Kabashima, 'Cross-validation estimate of the number of clusters in a network', Scientific Reports 7, 3327 (2017).

"""

using DocOpt  # import docopt function

args = docopt(doc, version=v"0.2.6-12-julia1.1")
strdataset = args["<filename>"]
if args["--dataset"] === Nothing()
    dataset = splitext(basename(args["<filename>"]))[1]
else
    dataset = args["--dataset"]
end
Blist = args["--q"]
initialconditions = args["--init"]
samples = parse(Int64,args["--initnum"])
degreecorrection = args["--dc"]
degreecorrection == "true" ? degreecorrection = true : degreecorrection = false
itrmax = parse(Int64,args["--itrmax"])
learningrate = parse(Float32,args["--learning_rate"])
priorlearning = args["--prior"]
priorlearning == "true" ? priorlearning = true : priorlearning = false

Blistarray = split(Blist,":")
if length(Blistarray) == 2
	Barray = [parse(Int64,Blistarray[1]):parse(Int64,Blistarray[2]);]
elseif length(Blistarray) == 3
	Barray = [parse(Int64,Blistarray[1]):parse(Int64,Blistarray[2]):parse(Int64,Blistarray[3]);]
else
    Barray = Int64[]
    for bb in split(Blist,",")
        push!(Barray,parse(Int64,bb))
    end
end

#mkdir("output_$(dataset)")
######################
######################


#strdataset = "edgelist_Nblock=[100, 100, 100]_cm=8_epsilon=0.1_1.txt"
#dataset = "overlappingSBM-2blocks"
Ltotinput = countlines(open( strdataset, "r" ))
fpmeta = open("output_$(dataset)/summary.txt","w")
write(fpmeta, "dataset: $(strdataset)\n")
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
    write(fpmeta, "number of edges (input, converted to simple graph): $(Int64(0.5*size(links,1)))\n")

    Nthreshold = round(Ntotinput*2/3)
#    Barray = [2:3;]
    Bsize = length(Barray)
    Bmax = maximum(Barray)
#    initialconditions = ["uniformAssortative"]#["normalizedLaplacian","random","uniformAssortative","uniformDisassortative","Bipartite"]
#    samples = 5
#    degreecorrection = false
#    learningrate = 0.1
#    itrmax = 1024
#    priorlearning = true
    spectral = false
    alluvial = true
    write(fpmeta, "initial conditions: $(join(initialconditions, ", "))\n")
    write(fpmeta, "numbmer of samples for each initial condition: $(samples)\n")
    write(fpmeta, "degree correction: $(degreecorrection)\n")
    write(fpmeta, "maximum number of iteration: $(itrmax)\n")
    write(fpmeta, "cluster size learning: $(priorlearning)\n")
    write(fpmeta, "learning rate: $(learningrate)\n")
    
    nb = [links[links[:,1].==lnode,2] for lnode = 1:Ntotinput]


    # cc = DFS(nb,1) # Assume node 1 belongs to the connected component
    # println("connected component identified...")
    # (Ntot,links) = LinksConnected(links,Ntotinput,cc)
    # println("nodes & links updated... : Ntot = $(Ntot) 2L = $(size(links,1))")
    # if Ntot < Nthreshold
    #     println("This is not a giant component... try again.")
    # end

    Ntot = 0
    for nn = 1:Ntotinput/2 
        cc = DFS(nb,nn) # Assume node 1 belongs to the connected component
        println("connected component identified...")
        (Ntot,newlinks) = LinksConnected(links,Ntotinput,cc)
        println("nodes & links updated... : Ntot = $(Ntot) 2L = $(size(newlinks,1))")
        if Ntot < Nthreshold
            println("This is not a giant component... try again.")
        else
            links = newlinks
            break
        end
    end



    Ltot = Int64(0.5*size(links,1))
    blockprev = zeros(Ntot,1)
    assignment = zeros(Int64,Ntot,Bsize+1) # 1st column is the node label, but B starts from 2
    assignment[:,1] = collect(1:Ntot)
    write(fpmeta, "numbmer of vertices (giant component): $(Ntot)\n")
    write(fpmeta, "numbmer of edges (giant component): $(Ltot)\n")

	nullerror = UniformGraphError(links,Ntot,degreecorrection)
	if degreecorrection == true
    	write(fpmeta, "Error for q=1 (degree-corrected): $(nullerror)\n")
	else
    	write(fpmeta, "Error for q=1 (degree-uncorrected): $(nullerror)\n")
	end
    
    if spectral == true
        degrees = degreesequence(links,Ntot)
        (NBlambdas,~) = nonbacktracking(links,degrees,Ntot,Bmax)
        spectralradius = sqrt.(real(NBlambdas[1]))
        BwithNBT = count(real(NBlambdas).>spectralradius)
    else
        spectralradius = "-"
        BwithNBT = 0
    end
    if BwithNBT == Bmax
        BwithNBT = 0
    end
    println("BwithNBT = $(BwithNBT)")
    write(fpmeta, "spectral radius of the non-backtracking matrix = $(spectralradius)\n")
    BwithNBT == 0 ? write(fpmeta, "non-backtracking matrix: q* = -\n") : write(fpmeta, "non-backtracking matrix: q* = $(BwithNBT)\n")
    
    fpAssessment = open("output_$(dataset)/assessment.txt","w")
    fpPartition = open("output_$(dataset)/assignment.txt","w")
    
    FEvec = zeros(Bsize,1)
    CVBayesvec = zeros(Bsize,2)
    CVGPvec = zeros(Bsize,2)
    CVGTvec = zeros(Bsize,2)
    CVMAPvec = zeros(Bsize,2)
    grDict = Dict()
    CrsDict = Dict()
    
    for bb = 1:Bsize
        B = Barray[bb]
        println("B = $(B)")
        FEs = 100*ones(samples,length(initialconditions))
        CVBayess = 100*ones(samples,length(initialconditions))
        CVGPs = 100*ones(samples,length(initialconditions))
        CVGTs = 100*ones(samples,length(initialconditions))
        CVMAPs = 100*ones(samples,length(initialconditions))
        varCVBayess = 100*ones(samples,length(initialconditions))
        varCVGPs = 100*ones(samples,length(initialconditions))
        varCVGTs = 100*ones(samples,length(initialconditions))
        varCVMAPs = 100*ones(samples,length(initialconditions))
        FEmin = 0
        itrnumopt = 0
        PSIopt = zeros(Ntot,B)
        gropt = zeros(B,1)
        Crsopt = zeros(B,B)
        
        maxCrs = Ntot
        
        for init = 1:length(initialconditions)
            println("Initial partition = $(initialconditions[init])")
            sm = 0
            giveup = 0
            overflow = 0
            while sm < samples
                (gr,Crs,PSI,FE,CVBayes,CVGP,CVGT,CVMAP,varCVBayes,varCVGP,varCVGT,varCVMAP,fail,cnv,itrnum) = EM(Ntot,B,links,maxCrs,initialconditions[init],degreecorrection,itrmax,priorlearning,learningrate)
                if cnv == false
                    # sm += 1
                    continue
                end
                if isnan(maximum(gr)) == true || isnan(maximum(Crs)) == true || maximum(PSI) == Inf
                    overflow += 1
                    if overflow > 2
                        println("overflow occurs too often : Use different initial partition or modifiy maxCrs.")
                        break
                    else
                        println("overflow... trying again.")
                        continue
                    end
                end
                if fail == true
                    giveup += 1
                    if giveup > 2
                        println("too many fails: give up!")
                        break
                    else
                        continue
                    end
                else
                    sm += 1
                end

                #println("sm = $(sm)")        
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
        SECVBayes = sqrt.(varCVBayess[findmin(CVBayess)[2]]/Ltot)
        SECVGP = sqrt.(varCVGPs[findmin(CVGPs)[2]]/Ltot)
        SECVGT = sqrt.(varCVGTs[findmin(CVGTs)[2]]/Ltot)
        SECVMAP = sqrt.(varCVMAPs[findmin(CVMAPs)[2]]/Ltot)
        
        FEvec[bb] = minimum(FEs)
        CVBayesvec[bb,1] = minimum(CVBayess)
        CVBayesvec[bb,2] = SECVBayes
        CVGPvec[bb,1] = minimum(CVGPs)
        CVGPvec[bb,2] = SECVGP
        CVGTvec[bb,1] = minimum(CVGTs)
        CVGTvec[bb,2] = SECVGT
        CVMAPvec[bb,1] = minimum(CVMAPs)
        CVMAPvec[bb,2] = SECVMAP
        grDict[bb] = gropt
        CrsDict[bb] = Crsopt

        (maxPSIval, ind) = findmax(PSIopt,dims=2)
        ind = Cartesian2ind(PSIopt,ind)
        block = [ind2sub((Ntot,B),index)[2] for index in ind]
        assignment[:,bb+1] = block # 1st column is the node label, but B starts from 2
        actualB = length(unique(vec(block)))
        write(fpmeta, "q = $(B): actual q = $(actualB), number of iteration = $(itrnumopt)\n")
        
        block = sortblock(block,blockprev)
        blockprev = block[:]
        if alluvial == true
            AlluvialDiagram(B,block,maxPSIval,dataset)
            AssignmentProbs(B,collect(1:Ntot),PSIopt,dataset)
        end
    end # B-loop
    
    # OUTPUT RESULTS :::::::::::::::::::::::::::::::::::::::::::::::::
    for i = 1:Ntot
        write(fpPartition, join(assignment[i,:], " ")*"\n")
    end

    write(fpAssessment, "Bethe free energy:\n")
    write(fpAssessment, "(q, assessed value)\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment, string(B)*" $(FEvec[bb])\n")
    end
    write(fpAssessment, "\nBayes prediction error:\n")
    write(fpAssessment, "(q, assessed value, standard error)\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment, "$(B) $(CVBayesvec[bb,1]) $(CVBayesvec[bb,2])\n")
    end
    write(fpAssessment, "\nGibbs prediction error:\n")
    write(fpAssessment, "(q, assessed value, standard error)\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment,"$(B) $(CVGPvec[bb,1]) $(CVGPvec[bb,2])\n")
    end
    write(fpAssessment, "\nGibbs training error:\n")
    write(fpAssessment, "(q, assessed value, standard error)\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment,"$(B) $(CVGTvec[bb,1]) $(CVGTvec[bb,2])\n")
    end
    write(fpAssessment, "\nGibbs prediction error (MAP estimate):\n")
    write(fpAssessment, "(q, assessed value, standard error)\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment,"$(B) $(CVMAPvec[bb,1]) $(CVMAPvec[bb,2])\n")
    end
    write(fpAssessment, "\nCluster sizes:\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment,"q = $(B)\n")
        write(fpAssessment,"$(grDict[bb])\n")
    end
    write(fpAssessment, "\nRescaled affinity matrix C:\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment,"q = $(B)\n")
        write(fpAssessment,"$(CrsDict[bb])\n")
    end
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    # PLOT RESULTS .................................................
    x = Barray
    y0 = FEvec
    y1 = CVBayesvec[:,1]
    y1error = CVBayesvec[:,2]
    y2 = CVGPvec[:,1]
    y2error = CVGPvec[:,2]
    y3 = CVGTvec[:,1]
    y3error = CVGTvec[:,2]
    y4 = CVMAPvec[:,1]
    y4error = CVMAPvec[:,2]    
    PlotAssessments(x,y0,y1,y1error,y2,y2error,y3,y3error,y4,y4error,dataset)

    PlotAffinityMatrix(Bsize,grDict,CrsDict,dataset)
    #..........................................................................
    
    close(fpAssessment)
    close(fpPartition)

end
close(fpmeta)