using PyPlot

function EM(Ntot,B,links,maxCrs,initial,degreecorrection,itrmax,priorlearning,learningrate)

    function logsumexp(array)
        array = vec(sortcols(array))
        for j = 1:length(array)
            array[end] - array[1] > log(10^(16.0)) ? shift!(array) : break # this cutoff must be smaller than cutoff in normalize_logprob: e.g. all elements are below cutoff.
        end
        if maximum(array) - minimum(array) > 700
            println("overflow or underflow is UNAVOIDABLE: restrict Crs to smaller values.")
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
            logPSIi = logunnormalizedMessage(i)
            PSI[i,:] = normalize_logprob(logPSIi)
        end
        return PSI
    end
        
    function logunnormalizedMessage(i)
        logmessages = 0
        for s in nb[i]
            indsi = sub2ind((Ntot,Ntot),s,i)
            logmessages += log(theta[i]*theta[s]*PSIcav[indsi]*Crs)
        end
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
        for i in randperm(Ntot)
            logPSIi = logunnormalizedMessage(i)
            for j in nb[i]
                indij = sub2ind((Ntot,Ntot),i,j)
                indji = sub2ind((Ntot,Ntot),j,i)
                PSIcav[indij] = logPSIi - log(theta[j]*theta[i]*PSIcav[indji]*Crs) # this is actually log(PSIcav[indij]) 
                PSIcav[indij] = normalize_logprob(PSIcav[indij])
            end
            
            hprev = theta[i]*PSI[i,:]*Crs/Ntot
			if priorlearning == true
            	grprev = PSI[i,:]/Ntot
			end
            prev = PSI[i,:]/Ntot
            logPSIi = logunnormalizedMessage(i) # new PSI with new PSIcav
            PSI[i,:] = normalize_logprob(logPSIi)
            h += theta[i]*PSI[i,:]*Crs/Ntot - hprev
			if priorlearning == true
				gr += PSI[i,:]/Ntot - grprev # learningrate omitted here: learning rate is much slower than Crs if learningrate is included.
			end
            conv += sum(abs(PSI[i,:]/Ntot - prev))
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
                    push!(logbiprob, log(PSIcav[indij][s])+log(theta[i])+log(Crs[s,t])+log(theta[j])+log(PSIcav[indji][t]))
                end
            end
            logZijs[cnt] = logsumexp(logbiprob') - log(Ntot)
            # average loglikelihood: 
            CVGPij = 0
            for s = 1:B
                for t=1:B
                    CVGPij += PSIcav[indij][s]*(log(theta[i])+log(Crs[s,t])+log(theta[j])-log(Ntot))*PSIcav[indji][t]
                end
            end
            CVGPs[cnt] = CVGPij
            # average likelihood with full prob. (training error)
            CVGTij = 0
            for s = 1:B
                for t=1:B
                    CVGTij += PSIcav[indij][s]*theta[i]*Crs[s,t]*theta[j]*(log(theta[i])+log(Crs[s,t])+log(theta[j])-log(Ntot))*PSIcav[indji][t]/(Ntot*exp(logZijs[cnt])) # factor N cancels at Crs and Zij
                end
            end
            CVGTs[cnt] = CVGTij
            # MAP estimate
            (MAPij, s) = findmax(PSIcav[indij])
            (MAPji, t) = findmax(PSIcav[indji])
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
    degrees = sum(A,2)

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

    if priorlearning == false
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





################################################
function simplegraph(links)
    links = vcat(links,hcat(links[:,2],links[:,1]))
    links = unique(links,1)
    # remove self-loops
    boolean = trues(size(links,1))
    for i = 1:size(links,1)
        links[i,1] == links[i,2] ? boolean[i] = false : continue
    end
    links = links[boolean,:]
    
    return links
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
        links[u,1] -= countnz(defects.<links[u,1])
        links[u,2] -= countnz(defects.<links[u,2])
    end
    
    return (Ntot,links)
end
################################################





function PlotAssessments(x,y0,y1,y1error,y2,y2error,y3,y3error,y4,y4error,dataset)
    Bmin = minimum(x)
    Bmax = maximum(x)
    
    fig = figure("plot_$(dataset)",figsize=(4,6))
#    subplots_adjust(hspace=0.5,wspace=0.3)
    subplots_adjust(hspace=0.1) # Set the vertical spacing between axes

    # 1st fig --------
    subplot(211)
    p = plot(x,y0,color="0.4",linestyle="-",marker="8",markeredgecolor="None",markersize=8,label="Bethe free energy")
    ax = gca()
	setp(ax[:get_xticklabels](),visible=false) # Disable x tick labels
    font1 = Dict("color"=>"k")
    ylabel("Bethe free energy",fontdict=font1)
    setp(ax[:get_yticklabels](),color="0.2") # Y Axis font formatting
#    xlabel(L"Number of clusters $\, \mathit{q}$",fontdict=font1)
    ax[:set_xlim](Bmin-0.2,Bmax+0.2)
    Mx = matplotlib[:ticker][:MultipleLocator](1) # Define interval
    ax[:xaxis][:set_major_locator](Mx) # Set interval 

    # 2nd fig --------
    subplot(212)
    p = plot(x,y2,color="#27AE60",linestyle="-",linewidth=2,marker="^",markersize=8,markerfacecolor="white",markeredgecolor="#27AE60",markeredgewidth=1.5,zorder=10,label="EGP")
    p = fill_between(x,vec(y2-y2error),vec(y2+y2error),color="#2ECC71",alpha=0.7,edgecolor="None",zorder=9)
    p = plot(x,y1,color="#C0392B",linestyle="-",linewidth=2,marker="o",markersize=7,markerfacecolor="white",markeredgecolor="#C0392B",markeredgewidth=1.5,zorder=8,label="EBayes")
    p = fill_between(x,vec(y1-y1error),vec(y1+y1error),color="#E74C3C",alpha=0.7,edgecolor="None",zorder=7)
    p = plot(x,y3,color="#2980B9",linestyle="-",linewidth=2,marker="D",markersize=7,markerfacecolor="white",markeredgecolor="#2980B9",markeredgewidth=1.5,zorder=6,label="EGT")
    p = fill_between(x,vec(y3-y3error),vec(y3+y3error),color="#3498DB",alpha=0.7,edgecolor="None",zorder=5)
    p = plot(x,y4,color="#F39C12",linestyle="-",linewidth=2,marker="s",markersize=7,markerfacecolor="white",markeredgecolor="#F39C12",markeredgewidth=1.5,zorder=4,label="EMAP")
    p = fill_between(x,vec(y4-y4error),vec(y4+y4error),color="#F1C40F",alpha=0.7,edgecolor="None",zorder=3)
    ax = gca()
    font1 = Dict("color"=>"k")
    ylabel("Prediction/training error",fontdict=font1)
    setp(ax[:get_yticklabels](),color="k") # Y Axis font formatting
    ax[:set_xlim](Bmin-0.2,Bmax+0.2)
    xlabel(L"Number of clusters $\, \mathit{q}$",fontdict=font1)
    Mx = matplotlib[:ticker][:MultipleLocator](1) # Define interval
    ax[:xaxis][:set_major_locator](Mx) # Set interval 

    axis("tight")

#    fig[:canvas][:draw]() # Update the figure
    suptitle(dataset,fontdict=font1)
    savefig("assessment_$(dataset).pdf",bbox_inches="tight",pad_inches=0.1)
end


function grCrsMatrices(grDictbb,CrsDictbb)
    B = length(grDictbb)
    Binmax = 100
    grCrsMatrix = zeros(Binmax,Binmax)
    binsizes = floor(Int64,Binmax*grDictbb)
    offsetr = 0
    for r = 1:B
        offsets = 0
        for s = 1:B
            if r == B && s < B
                grCrsMatrix[offsetr+1:Binmax,offsets+1:offsets+binsizes[s]] = CrsDictbb[r,s]
            elseif r < B && s == B
                grCrsMatrix[offsetr+1:offsetr+binsizes[r],offsets+1:Binmax] = CrsDictbb[r,s]
            elseif r == B && s == B
                grCrsMatrix[offsetr+1:Binmax,offsets+1:Binmax] = CrsDictbb[r,s]
            else
                grCrsMatrix[offsetr+1:offsetr+binsizes[r],offsets+1:offsets+binsizes[s]] = CrsDictbb[r,s]
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
    savefig("structures_$(dataset).pdf",bbox_inches="tight",pad_inches=0.1)
end






######################
# Keyboard inputs ####
doc = """

Usage:
  sbm.jl <filename> [--dataset=<dataset>] [--dc=<dc>] [--q=Blist] [--init=partition...] [--initnum=<samples>] [--itrmax=<itrmax>] [--learning_rate=<learningrate>] [--prior=priorlearning]
  sbm.jl -h | --help
  sbm.jl --version
  

Options:
  -h --help                 		Show this screen.
  --version                 		Show version.
  --dataset							Name of the dataset [default: ""]
  --q=Blist                 		List of number of clusters. [default: 2:6]
  --init=partition...       		Initial partition. [default: normalizedLaplacian]
  --initnum=<samples>       		Number of initial states. [default: 10]
  --dc=<dc>                 		Degree correction. [default: true]
  --itrmax=<itrmax>       			Maximum number of BP iteration. [default: 128]
  --prior=priorlearning     		Learn cluster sizes. [default: true]
  --learning_rate=<learningrate>	Learning rate. [default: 1]

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
Reference: arXiv:1605.07915 (2016).

"""

using DocOpt  # import docopt function

args = docopt(doc, version=v"0.2.6") #0.2.6.3
strdataset = args["<filename>"]
dataset = args["--dataset"]
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
######################
######################






#strdataset = "polbooks_newman.txt"
#dataset = "political books"
Ltotinput = countlines(open( strdataset, "r" ))
fpmeta = open("summary.txt","w")
write(fpmeta, "dataset: $(strdataset)\n")
@time open( strdataset, "r" ) do fp
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

    Nthreshold = round(Ntotinput/2)
#    Barray = [2:8;]
    Bsize = length(Barray)
#    samples = 10
#    degreecorrection = false
#    initialconditions = ["normalizedLaplacian","random","uniformAssortative","uniformDisassortative"]
#    learningrate = 0.3
#    itrmax = 1024
#    priorlearning = true
    write(fpmeta, "initial conditions: $(join(initialconditions, ", "))\n")
    write(fpmeta, "numbmer of samples for each initial condition: $(samples)\n")
    write(fpmeta, "degree correction: $(degreecorrection)\n")
    write(fpmeta, "maximum number of iteration: $(itrmax)\n")
    write(fpmeta, "cluster size learning: $(priorlearning)\n")
    write(fpmeta, "learning rate: $(learningrate)\n")
    
    nb = [links[links[:,1].==lnode,2] for lnode = 1:Ntotinput]
    cc = DFS(nb,1) # Assume node 1 belongs to the connected component
    println("connected component identified...")
    (Ntot,links) = LinksConnected(links,Ntotinput,cc)
    println("nodes & links updated... : Ntot = $(Ntot) 2L = $(size(links,1))")
    if Ntot < Nthreshold
        println("This is not a giant component... try again.")
    end
    Ltot = Int64(0.5*size(links,1))
    assignment = zeros(Int64,Ntot,Bsize+1) # 1st column is the node label, but B starts from 2
    assignment[:,1] = collect(1:Ntot)
    write(fpmeta, "numbmer of vertices (giant component): $(Ntot)\n")
    write(fpmeta, "numbmer of edges (giant component): $(Ltot)\n")

    fpAssessment = open("assessment.txt","w")
    fpPartition = open("assignment.txt","w")
    
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
        SECVBayes = sqrt(varCVBayess[findmin(CVBayess)[2]]/Ltot)
        SECVGP = sqrt(varCVGPs[findmin(CVGPs)[2]]/Ltot)
        SECVGT = sqrt(varCVGTs[findmin(CVGTs)[2]]/Ltot)
        SECVMAP = sqrt(varCVMAPs[findmin(CVMAPs)[2]]/Ltot)
        
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

        (val, ind) = findmax(PSIopt,2)
        block = ind2sub((Ntot,B),vec(ind))[2]
        assignment[:,bb+1] = block # 1st column is the node label, but B starts from 2
        actualB = length(unique(vec(block)))
        write(fpmeta, "q = $(B): actual q = $(actualB), number of iteration = $(itrnumopt)\n")
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