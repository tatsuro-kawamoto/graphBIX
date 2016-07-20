using PyPlot

function EM(gr,Ntot,B,links,maxomega,itrmax,BPconvthreshold,betafrac,initialnoise,learning,priorlearning,dc)

    function logsumexp(array)
        array = vec(sortcols(array))
        for j = 1:length(array)
            array[end] - array[1] > log(10^(16.0)) ? shift!(array) : break # this cutoff must be smaller than cutoff in normalize_logprob: e.g. all elements are below cutoff.
        end
        if maximum(array) - minimum(array) > 700
            println("overflow or underflow is UNAVOIDABLE: restrict omega to smaller values.")
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

    function update_h()
        h = degrees'*PSI
    end

    function update_gr(PSI)
        gr = mean(PSI,1)
    end
    
    function updatePSI()
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
            logmessages += log(ones(1,B) + PSIcav[indsi]*(exp(beta)-1))
        end

        if dc == false
            logmessages = logmessages - alpha*beta*h + log(gr)
        else
            logmessages = logmessages - alpha*beta*length(nb[i])*h + log(gr)
        end        
        
        return logmessages
    end

    function BP()
        conv = 0
        h = update_h()
        for i in randperm(Ntot)
            logPSIi = logunnormalizedMessage(i)
            for j in nb[i]
                indij = sub2ind((Ntot,Ntot),i,j)
                indji = sub2ind((Ntot,Ntot),j,i)
                PSIcav[indij] = logPSIi - log(ones(1,B) + PSIcav[indji]*(exp(beta)-1))
                PSIcav[indij] = normalize_logprob(PSIcav[indij])
            end

            hprev = degrees[i]*PSI[i,:]
            prev = PSI[i,:]/Ntot
            logPSIi = logunnormalizedMessage(i) # new PSI with new PSIcav
            PSI[i,:] = normalize_logprob(logPSIi)
            h += degrees[i]*PSI[i,:] - hprev
            conv += sum(abs(PSI[i,:]/Ntot - prev))
        end
        return (conv, PSI)
    end
    
    function update_omega()
        omegainnew = 0
        for ij = 1:Ktot
            (i,j) = (links[ij,1],links[ij,2])
            if i > j
                continue
            end
            indij = sub2ind((Ntot,Ntot),i,j)
            indji = sub2ind((Ntot,Ntot),j,i)
            sumPSIPSI = (PSIcav[indij]*PSIcav[indji]')[1] # \sum_sigma PSI^{i\to j}_sigma*PSI^{j\to i}_sigma
            omegainnew +=  omegain*sumPSIPSI/((omegain-omegaout)*sumPSIPSI+omegaout)
        end
        dsigmanorm = norm(update_h())^2
        omegain = 2*omegainnew/dsigmanorm
        if dc == false
            omegaout = 2*(0.5*Ktot - omegainnew)/(Ntot^2 - dsigmanorm)
        else
            omegaout = 2*(0.5*Ktot - omegainnew)/(Ktot^2 - dsigmanorm)
        end
        
        if omegain > maxomega
            omegain = maxomega
        elseif omegain < 0
            omegain = 0
        elseif omegaout < 0
            omegaout = 0
        end
        
        return (omegain, omegaout)
    end
    
    function update_alphabeta()
        beta = log(omegain/omegaout)
        alpha = (omegain - omegaout)/beta
        return (alpha, beta)
    end
    
    function freeenergy()
        sumlogZi = 0
        for i = 1:Ntot
            sumlogZi += logsumexp(logunnormalizedMessage(i))
        end
        
        Ltot = Int64(0.5*size(links,1))
        logZijs = zeros(Ltot)
        CVBPs = zeros(Ltot)
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
            sumPSIPSI = (PSIcav[indij]*PSIcav[indji]')[1] # \sum_sigma PSI^{i\to j}_sigma*PSI^{j\to i}_sigma
            logZijs[cnt] = log(1 + (exp(beta)-1)*sumPSIPSI)
            
            # Bayes prediction error: 
            CVBPs[cnt] = logZijs[cnt] + log(degrees[i]) + log(omegaout) + log(degrees[j])
            # Gibbs prediction error: 
            CVGPs[cnt] = beta*sumPSIPSI
            # Gibbs training error: 
            CVGTs[cnt] = degrees[i]*degrees[j]*(omegaout*log(omegaout) + (omegain*log(omegain) - omegaout*log(omegaout))*sumPSIPSI)/exp(CVBPs[cnt])
            # MAP estimate of the Gibbs prediction error: 
            (MAPij, s) = findmax(PSIcav[indij])
            (MAPji, t) = findmax(PSIcav[indji])
            if s == t
                CVMAPs[cnt] = log(omegain)
            else
                CVMAPs[cnt] = log(omegaout)
            end
            #################
        end
        
        constshift = sum([degrees[i]*log(degrees[i]) for i in 1:size(degrees,1)])/Ltot
        
        #FEmodbp = -(sumlogZi - sum(logZijs) + 0.5*alpha*beta*norm(update_h())^2 )/(beta*Ntot)
        FE = -(sumlogZi - sum(logZijs) + 0.5*alpha*beta*norm(update_h())^2 )/(beta*Ntot) - (Ltot/Ntot)*(2*Ltot*omegaout+log(omegaout)+constshift)/beta
        CVBayes = -sum(CVBPs)/Ltot + 1
        CVGP = -sum(CVGPs)/Ltot - log(omegaout) + 1 - constshift
        CVGT = -sum(CVGTs)/Ltot + 1 - constshift
        CVMAP = -sum(CVMAPs)/Ltot + 1 - constshift
        varCVBayes = var(logZijs)
        varCVGP = var(CVGPs)
        varCVGT = var(CVGTs)
        varCVMAP = var(CVMAPs)
        return (FE, CVBayes, CVGP, CVGT, CVMAP, varCVBayes, varCVGP, varCVGT, varCVMAP)
    end    

# initial state ########
    cnv = false
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
    #=
    # high degree cutoff
    cutoff = 10
    for i = 1:Ntot
        if length(nb[i]) > cutoff
            nb[i] = nb[i][1:cutoff]
        end
    end
    =#
    ###############
    degrees = sum(A,2)
    excm = (degrees'*degrees)[1]/(Ntot*mean(degrees)) - 1# average excess degree
    beta0 = log(1 + B/(excm-1))
    betaast = log(1 + B/(sqrt(excm)-1))
    
    if dc == false
        degrees = ones(Ntot,1) #### Remove degree-correction #######
    end

    alpha = 1/Ktot
    beta = betaast - betafrac*(betaast - beta0)
    omegain = exp(beta)*alpha*beta/(exp(beta)-1)
    omegaout = alpha*beta/(exp(beta)-1)
    
    PSIcav = Dict()
    for ind in inds
        PSIcav[ind] = abs( ones(1,B) + initialnoise*(0.5*ones(1,B) - rand(1,B)) ) # noise strength is an important factor.
        PSIcav[ind] = PSIcav[ind]/sum(PSIcav[ind])
    end
    h = zeros(1,B)
    PSI = zeros(Ntot,B)
    PSI = updatePSI()
    FE = 0
    CVBayes = 0
    CVGP = 0
    CVGT = 0
    CVMAP = 0
    varCVBayes = 0
    varCVGP = 0
    varCVGT = 0
    varCVMAP = 0
################
    
    #itrmax = 128
    #BPconvthreshold = 0.00001
    for itr = 1:itrmax
        (BPconv, PSI) = BP()
        if priorlearning == true
            gr = update_gr(PSI)
        end        
        if learning == true
            (omegain, omegaout) = update_omega()
            (alpha, beta) = update_alphabeta()
        end
        if BPconv < BPconvthreshold
            println("converged! ^_^: itr = $(itr)")
            cnv = true
            itrnum = itr
            break
        elseif itr == itrmax
            println("NOT converged... T_T: BPconv = $(Float16(BPconv))")
        end
    end

    h = update_h()
    (FE, CVBayes, CVGP, CVGT, CVMAP, varCVBayes, varCVGP, varCVGT, varCVMAP) = freeenergy()
    
    return (excm,alpha,beta,omegain,omegaout,PSI,FE,CVBayes,CVGP,CVGT,CVMAP,varCVBayes,varCVGP,varCVGT,varCVMAP,cnv,itrnum)
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

function degreesequence(links,Ntot)
    Ktot = size(links,1)
    A = sparse(links[:,1],links[:,2],ones(Ktot),Ntot,Ntot)
    degrees = sum(A,2)
    return degrees
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
######################################










function retrievalQ(links,degrees,block)
    Ktot = size(links,1)
    retQ = 0
    for k = 1:size(links,1)
        (i,j) = tuple(links[k,:]...)
        if i<j
            continue
        end
        if block[i] == block[j]
            retQ += 2/Ktot  # = 1/L
        end
    end
    B = maximum(block)
    blockdegrees = zeros(B)
    for i = 1:size(block,1)
        blockdegrees[block[i]] += degrees[i]
    end
    retQ = retQ - (blockdegrees'*blockdegrees)[1]/(Ktot^2)
    return retQ
end


function MinimumDescriptionLength(links,degrees,block)
    # MDL of the map equation
    Ktot = size(links,1)
    B = maximum(block)
    lin = zeros(Int64,B)
    lout = zeros(Int64,B)
    cut = 0
    for k = 1:size(links,1)
        (i,j) = tuple(links[k,:]...)
        if i<j
            continue
        end
        if block[i] == block[j]
            lin[block[i]] += 1
        else
            lout[block[i]] += 1
            lout[block[j]] += 1
            cut += 1
        end
    end
    MDL = 0
    for k = 1:B
        lout[k] == 0 ? continue : MDL += -lout[k]*log2(lout[k]) + (lin[k]+lout[k])*log2((lin[k]+lout[k]))
    end
    MDL = 2*MDL
    MDL += 2*cut*(1+log2(2*cut)) + Ktot
    for i = 1:size(degrees,1)
        if degrees[i] == 0
            println("degree = 0 exists!")
        end
        MDL -= degrees[i]*log2(degrees[i])
    end
    MDL = MDL/Ktot
    return MDL
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








function sortblock(block,blockprev)
    Bnow = maximum(block)
    Bprev = maximum(blockprev)
    if Bprev > 1
        blbl = hcat(block,blockprev)
        M = zeros(Int64,Bnow,Bprev)
        for i = 1:size(block,1)
            M[blbl[i,1],blbl[i,2]] += 1
        end
        (val,ind) = findmax(M,2)
        maxind = ind2sub((Bnow,Bprev),vec(ind))[2]
        varM = var(M,2)
        c = collect(1:Bnow)
        MM = sortrows(hcat(maxind,varM,c), by=x->(x[1],-x[2]))
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

# Alluvial diagram: smap file generator ##########
function AlluvialDiagram(B,block,maxPSIval)
    significant = 0.7
    Ntot = size(maxPSIval,1)
    stralluvial = "partition$(B).smap"
    fpalluvial = open(stralluvial,"w")
    
    modules = zeros(Int64,Ntot,3) # [node num., block label, significance]
    modules[:,1] = collect(1:Ntot)
    modules[:,2] = block
    modules = sortrows(modules, by=x->x[2])
    
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
######################################



function PlotResults(x,w1,w2,y1,y1error,y2,y2error,y3,y3error,y4,y4error,z1,z2,z3,BwithNBT,excm,dataset)
    #Bmax = size(x,1) + 1
    Bmin = minimum(x)
    Bmax = maximum(x)
    
    fig = figure("plot_$(dataset)",figsize=(4,9))
    subplots_adjust(hspace=0.1) # Set the vertical spacing between axes
    
    # 1st fig --------
    subplot(311)
    p = plot(x,w2,color="crimson",linestyle="-",marker="+",markersize=8,markeredgewidth=2,label="beta",zorder=2)
    beta0vec = log(1 + [Bmin:Bmax;]/(excm-1))
    betaastvec = log(1 + [Bmin:Bmax;]/(sqrt(excm)-1))
    p = fill_between([Bmin:Bmax;],vec(beta0vec),vec(betaastvec),color="lightgray",alpha=0.4,linestyle=":",edgecolors="white",zorder=1)
    ax = gca()
    setp(ax[:get_xticklabels](),visible=false) # Disable x tick labels
    fontw2 = Dict("color"=>"crimson","size"=>18)
    ylabel(L"$\beta$",fontdict=fontw2)
    setp(ax[:get_yticklabels](),color="crimson") # Y Axis font formatting

    axw1 = ax[:twinx]() # Create another axis on top of the current axis
    p = plot(x,w1,color="mediumblue",linestyle="-",marker="x",markersize=8,markeredgewidth=2,zorder=3,label="alpha")
    p = axvline(x=BwithNBT,linestyle="--",color="rosybrown",zorder=2)
    fontw1 = Dict("color"=>"mediumblue","size"=>16)
    ylabel(L"$\alpha$",fontdict=fontw1)
    setp(axw1[:get_yticklabels](),color="mediumblue") # Y Axis font formatting
    ax[:set_xlim](Bmin-0.2,Bmax+0.2)

    # 2nd fig --------
    subplot(312)
    p = plot(x,y2,color="darksage",linestyle="-",marker="^",markersize=6,markerfacecolor="white",markeredgecolor="darksage",markeredgewidth=2,zorder=10,label="EGP")
    p = fill_between(x,vec(y2-y2error),vec(y2+y2error),color="lawngreen",alpha=0.2,edgecolor="None",zorder=9)
    p = plot(x,y1,color="firebrick",linestyle="-",marker="o",markersize=6,markerfacecolor="white",markeredgecolor="firebrick",markeredgewidth=2,zorder=8,label="EBayes")
    p = fill_between(x,vec(y1-y1error),vec(y1+y1error),color="red",alpha=0.2,edgecolor="None",zorder=7)
    p = plot(x,y3,color="deepskyblue",linestyle="-",marker="D",markersize=6,markerfacecolor="white",markeredgecolor="deepskyblue",markeredgewidth=2,zorder=6,label="EGT")
    p = fill_between(x,vec(y3-y3error),vec(y3+y3error),color="deepskyblue",alpha=0.2,edgecolor="None",zorder=5)
    p = plot(x,y4,color="goldenrod",linestyle="-",marker="s",markersize=6,markerfacecolor="white",markeredgecolor="goldenrod",markeredgewidth=2,zorder=4,label="EMAP")
    p = fill_between(x,vec(y4-y4error),vec(y4+y4error),color="goldenrod",alpha=0.2,edgecolor="None",zorder=3)
    p = axvline(x=BwithNBT,linestyle="--",color="rosybrown",zorder=2)
    ax = gca()
    setp(ax[:get_xticklabels](),visible=false) # Disable x tick labels
    font1 = Dict("color"=>"k")
    ylabel("Prediction/training error",fontdict=font1)
    setp(ax[:get_yticklabels](),color="k") # Y Axis font formatting
    ax[:set_xlim](Bmin-0.2,Bmax+0.2)

    # 3rd fig --------
    subplot(313)
    p = plot(x,z1,color="0.4",linestyle="-",marker="8",markeredgecolor="None",markersize=8,label="Bethe free energy")
    p = axvline(x=BwithNBT,linestyle="--",color="rosybrown")
    ax = gca()
    font1 = Dict("color"=>"0.2")
    ylabel("Bethe free energy",fontdict=font1)
    setp(ax[:get_yticklabels](),color="0.2") # Y Axis font formatting
    fontLarge = Dict("weight"=>"normal","size"=>16)
    xlabel(L"Number of clusters $\mathit{q}$",fontdict=fontLarge)

        ################
        #  Other Axes  #
        ################
        ax2 = ax[:twinx]() # Create another axis on top of the current axis
        font2 = Dict("color"=>"darkorange")
        ylabel("Modularity",fontdict=font2)
        p = plot_date(x,z2,color="darkorange",linestyle="-",marker="p",markeredgecolor="None",markersize=8,label="Modularity")
        setp(ax2[:get_yticklabels](),color="darkorange") # Y Axis font formatting
        ax2[:yaxis][:set_major_formatter]

        ax3 = ax[:twinx]() # Create another axis on top of the current axis
        ax3[:spines]["right"][:set_position](("axes",1.28)) # Offset the y-axis label from the axis itself so it doesn't overlap the second axis
        font3 = Dict("color"=>"royalblue")
        ylabel("MDL (map equation)",fontdict=font3)
        p = plot_date(x,z3,color="royalblue",linestyle="-",marker="h",markeredgecolor="None",markersize=8,label="MDL") 
        setp(ax3[:get_yticklabels](),color="royalblue") # Y Axis font formatting

        # Enable just the right part of the frame
        ax3[:set_frame_on](true) # Make the entire frame visible
        ax3[:patch][:set_visible](false) # Make the patch (background) invisible so it doesn't cover up the other axes' plots
        ax3[:spines]["top"][:set_visible](false) # Hide the top edge of the axis
        ax3[:spines]["bottom"][:set_visible](false) # Hide the bottom edge of the axis

    #axis("tight")

    ax[:set_xlim](Bmin-0.2,Bmax+0.2)
#    fig[:canvas][:draw]() # Update the figure
    suptitle(dataset,fontdict=fontLarge)
    savefig("plot_$(dataset).pdf",bbox_inches="tight",pad_inches=0.1)
    
end










######################
# Keyboard inputs ####
doc = """

Usage:
  mod.jl <filename> [--q=Blist] [--dc=dc] [--learning=learning] [--prior=priorlearning] [--alluvial=alluvial] [--initnum=<samples>] [--itrmax=<itrmax>] [--conv=<BPconvthreshold>] [--noise=<initialnoise>]
  mod.jl -h | --help
  mod.jl --version
  

Options:
  -h --help                 	Show this screen.
  --version                 	Show version.
  --q=Blist                 	List of number of clusters. [default: 2:6]
  --dc=dc           			Fit the degree-corrected SBM. [default: true]  
  --learning=learning           Learn affinity matrix. [default: true]
  --prior=priorlearning         Learn cluster sizes. [default: false]
  --alluvial=alluvial           Generate smap files for the Alluvial diagram. [default: false]
  --initnum=<samples>       	Number of initial states. [default: 3]
  --itrmax=<itrmax>         	Maximum number of iterations in BP. [default: 128]
  --conv=<BPconvthreshold>      Convergence criterion of BP. [default: 0.00001]
  --noise=<initialnoise>        Noise strength in initial cavity bias psi^{i->j}. [default: 0.1]  

========================
Bayesian inference for the degree-corrected stochastic block model with a restricted affinity matrix 
using EM algorithm + belief propagation with the leave-one-out cross-validation.

Examples: 
Inference & model assessment of `edgelist.txt` for the SBM with q = 2, 4, and 6:
julia mod.jl edgelist.txt --q=2,4,6
or 
julia mod.jl edgelist.txt --q=2:2:6
========================
Author: Tatsuro Kawamoto: kawamoto.tatsuro@gmail.com
Reference: Tatsuro Kawamoto and Yoshiyuki Kabashima, arXiv:1606.07668 (2016).

"""

using DocOpt  # import docopt function

args = docopt(doc, version=v"0.1.3")
strdataset = args["<filename>"]
Blist = args["--q"]
dc = args["--dc"]
dc == "true" ? dc = true : dc = false
learning = args["--learning"]
learning == "true" ? learning = true : learning = false
priorlearning = args["--prior"]
priorlearning == "true" ? priorlearning = true : priorlearning = false
alluvial = args["--alluvial"]
alluvial == "true" ? alluvial = true : alluvial = false
samples = parse(Int64,args["--initnum"])
itrmax = parse(Int64,args["--itrmax"])
BPconvthreshold = parse(Float64,args["--conv"])
initialnoise = parse(Float64,args["--noise"])

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













dataset = "model_assessment" # plot title
#strdataset = "connected-edgelist-polbooks.txt"
Ltotinput = countlines(open( strdataset, "r" ))
fpmeta = open("summary.txt","w")
write(fpmeta, "dataset: $(strdataset)\n\n")
open( strdataset, "r" ) do fp
    cnt = 0
    excm = 0
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
#	dc = true # degree-correction
#	priorlearning = false
    plots = true
#    learning = true
    spectral = true
#    alluvial = false
#    Barray = [2:1:5;]
    Bsize = length(Barray)
    Bmax = maximum(Barray)
    Bmin = minimum(Barray)
#    initialnoise = 0.1
#    itrmax = 64
#    BPconvthreshold = 0.000001
#    samples = 1
    
    A = sparse(links[:,1],links[:,2],ones(size(links,1)),Ntotinput,Ntotinput)
    nb = Array[]
    row = rowvals(A)
    for j = 1:Ntotinput
        push!(nb,Int64[row[i] for i in nzrange(A,j)])
    end
    cc = DFS(nb,1) # Assume node 1 belongs to the connected component
    println("connected component identified...")
    (Ntot,links) = LinksConnected(links,Ntotinput,cc)
    println("vertices & edges updated... : N = $(Ntot) 2L = $(size(links,1))")
    if Ntot < Nthreshold
        println("This is not a giant component... try again.")
    end
    Ltot = Int64(0.5*size(links,1))
    degrees = degreesequence(links,Ntot)
    blockprev = zeros(Ntot,1)
    assignment = zeros(Int64,Ntot,Bsize+1) # 1st column is the node label, but B starts from 2
    assignment[:,1] = collect(1:Ntot)
    BwithretQ = Bmax
    BwithMDL = Bmax
    BwithCVBP = Bmax
    BwithCVGP = Bmax
    retQopt = -1000000
    MDLopt = 1000000
    CVBPopt = 1000000
    CVGPopt = 1000000
    write(fpmeta, "numbmer of vertices (giant component): $(Ntot)\n")
    write(fpmeta, "numbmer of edges (giant component): $(Ltot)\n\n")
    write(fpmeta, "Hyperparameter learning = $(learning)\n")
    write(fpmeta, "Noise strength of initial cavity biases = $(initialnoise)\n")
    write(fpmeta, "Max number of iteration = $(itrmax)\n")
    write(fpmeta, "BP convergence threshold = $(BPconvthreshold)\n")
    write(fpmeta, "Number of initial states = $(samples)\n\n")

    strAssessment = "assessment.txt"
#    strNBspectra = "spectraNonBacktracking.txt"
    
    strPartition = "assignment.txt"
    
    fpAssessment = open(strAssessment,"w")
    fpPartition = open(strPartition,"w")
#    fpNBspectra = open(strNBspectra,"w")
    
    retQvec = zeros(Bsize)
    MDLvec = zeros(Bsize)
    FEvec = zeros(Bsize)
    CVBayesvec = zeros(Bsize,2)
    CVGPvec = zeros(Bsize,2)
    CVGTvec = zeros(Bsize,2)
    CVMAPvec = zeros(Bsize,2)
    Alphavec = zeros(Bsize)
    Betavec = zeros(Bsize)
    Omegavec = zeros(Bsize,2)

    betafrac = 0
    for bb = 1:Bsize
        B = Barray[bb]
        println("B = $(B)")
        FEs = zeros(samples,1)
        CVBPs = zeros(samples,1)
        CVGPs = zeros(samples,1)
        CVGTs = zeros(samples,1)
        CVMAPs = zeros(samples,1)
        varCVBPs = zeros(samples,1)
        varCVGPs = zeros(samples,1)
        varCVGTs = zeros(samples,1)
        varCVMAPs = zeros(samples,1)
        FEmin = 100000
        itrnumopt = 0
        PSIopt = zeros(Ntot,B)
        omegaopt = zeros(2)
        alphabetaopt = zeros(2)
                
        gr = ones(1,B)/B # prior distribution
		
        ## upper bound of omega ===================
        maxomega = 1
        ##=================================
        
        sm = 0
        overflow = 0
        while sm < samples
            (excm,alpha,beta,omegain,omegaout,PSI,FE,CVBayes,CVGP,CVGT,CVMAP,varCVBayes,varCVGP,varCVGT,varCVMAP,cnv,itrnum) = EM(gr,Ntot,B,links,maxomega,itrmax,BPconvthreshold,betafrac,initialnoise,learning,priorlearning,dc)
            if cnv == false
                betafrac += 0.1
                if betafrac <= 1
                    continue
                else
                    println("BP does not converge.... : Raise itrnum or/and BPconv.")
                end
            end
            if isnan(maximum(PSI)) == true || maximum(PSI) == Inf || abs(FE) == Inf || abs(CVGP) == Inf || abs(CVMAP) == Inf
                overflow += 1
                if overflow > 10
                    println("overflow occurs too often...")
                    break
                else
                    println("overflow...")
                    continue
                end
            end
            sm += 1

            FEs[sm] = FE
            CVBPs[sm] = CVBayes
            CVGPs[sm] = CVGP
            CVGTs[sm] = CVGT
            CVMAPs[sm] = CVMAP
            varCVBPs[sm] = varCVBayes
            varCVGPs[sm] = varCVGP
            varCVGTs[sm] = varCVGT
            varCVMAPs[sm] = varCVMAP                
            if FE < FEmin
                FEmin = FE
                itrnumopt = itrnum
                PSIopt = PSI
                omegaopt = [omegain, omegaout]
                alphabetaopt = [alpha, beta]
            end
        end # while sample loop

        # standard errors of CVs
        SECVBayes = sqrt(varCVBPs[findmin(CVBPs)[2]]/Ltot)
        SECVGP = sqrt(varCVGPs[findmin(CVGPs)[2]]/Ltot)
        SECVGT = sqrt(varCVGTs[findmin(CVGTs)[2]]/Ltot)
        SECVMAP = sqrt(varCVMAPs[findmin(CVMAPs)[2]]/Ltot)
        
        FEvec[bb] = minimum(FEs)
        CVBayesvec[bb,1] = minimum(CVBPs)
        CVBayesvec[bb,2] = SECVBayes
        CVGPvec[bb,1] = minimum(CVGPs)
        CVGPvec[bb,2] = SECVGP
        CVGTvec[bb,1] = minimum(CVGTs)
        CVGTvec[bb,2] = SECVGT
        CVMAPvec[bb,1] = minimum(CVMAPs)
        CVMAPvec[bb,2] = SECVMAP
        Alphavec[bb] = 2*Ltot*alphabetaopt[1]
        Betavec[bb] = alphabetaopt[2]
        Omegavec[bb,1] = omegaopt[1]
        Omegavec[bb,2] = omegaopt[2]
        if minimum(CVBPs) < CVBPopt
            CVBPopt = minimum(CVBPs)
            BwithCVBP = B
        end        
        if minimum(CVGPs) < CVGPopt
            CVGPopt = minimum(CVGPs)
            BwithCVGP = B
        end        
        
        (maxPSIval, ind) = findmax(PSIopt,2)
        block = ind2sub((Ntot,B),vec(ind))[2]
        assignment[:,bb+1] = block # 1st column is the node label, but B starts from 2
        actualB = length(unique(vec(block)))
        write(fpmeta, "q = $(B): actual q = $(actualB), number of iteration = $(itrnumopt) \n")            
        
        retQ = retrievalQ(links,degrees,block)
        retQvec[bb] = retQ
        if retQ > retQopt
            retQopt = retQ
            BwithretQ = B
        end        
        MDL = MinimumDescriptionLength(links,degrees,block)
        MDLvec[bb] = MDL
        if MDL < MDLopt
            MDLopt = MDL
            BwithMDL = B
        end
        
        block = sortblock(block,blockprev)
        blockprev = block[:]
        if alluvial == true
            AlluvialDiagram(B,block,maxPSIval)
        end
    end # B-loop
    if spectral == true
        (NBlambdas,~) = nonbacktracking(links,degrees,Ntot,Bmax)
        spectralradius = sqrt(real(NBlambdas[1]))
        BwithNBT = countnz(real(NBlambdas).>spectralradius)
        #=
        for k = 1:size(NBlambdas,1)
            write(fpNBspectra, "$(NBlambdas[k])\n")
        end
        =#
    else
        spectralradius = "-"
        BwithNBT = 0
    end
    if BwithNBT == Bmax
        BwithNBT = 0
    end

    write(fpmeta, "\n")
    write(fpmeta, "mean excess degree = $(Float32(excm))\n\n")
    write(fpmeta, "spectral radius of the non-backtracking matrix = $(spectralradius)\n")
    BwithNBT == 0 ? write(fpmeta, "non-backtracking matrix: q* = -\n") : write(fpmeta, "non-backtracking matrix: q* = $(BwithNBT)\n")
    write(fpmeta, "modularity: q* = $(BwithretQ) (This may not be parsimonius.)\n")
    write(fpmeta, "map equation: q* = $(BwithMDL) (This may not be parsimonius.)\n")
    write(fpmeta, "CV(Bayes Prediction): q* = $(BwithCVBP) (This may not be parsimonius.)\n")
    write(fpmeta, "CV(Gibbs Prediction): q* = $(BwithCVGP) (This may not be parsimonius.)\n")
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
    write(fpAssessment, "\nModularity:\n")
    write(fpAssessment, "(q, assessed value)\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment,"$(B) $(retQvec[bb])\n")
    end
    write(fpAssessment, "\nMDL (map equation):\n")
    write(fpAssessment, "(q, assessed value)\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment,"$(B) $(MDLvec[bb])\n")
    end
    write(fpAssessment, "\nalpha:\n")
    write(fpAssessment, "(q, learned value)\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment,"$(B) $(Alphavec[bb])\n")
    end
    write(fpAssessment, "\nbeta:\n")
    write(fpAssessment, "(q, learned value)\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment,"$(B) $(Betavec[bb])\n")
    end
    write(fpAssessment, "\nomega (affinity matrix elements):\n")
    write(fpAssessment, "(q, omega_in, omega_out)\n")
    for bb = 1:Bsize
        B = Barray[bb]
        write(fpAssessment,"$(B) $(Omegavec[bb,1]) $(Omegavec[bb,2])\n")
    end
    
    
    if plots == true
        x = Barray#[2:Bmax;]
        w1 = Alphavec#[2:Bmax]
        w2 = Betavec#[2:Bmax]
        y1 = CVBayesvec[:,1]#[2:Bmax,1]
        y1error = CVBayesvec[:,2]#[2:Bmax,2]
        y2 = CVGPvec[:,1]#[2:Bmax,1]
        y2error = CVGPvec[:,2]#[2:Bmax,2]
        y3 = CVGTvec[:,1]#[2:Bmax,1]
        y3error = CVGTvec[:,2]#[2:Bmax,2]
        y4 = CVMAPvec[:,1]#[2:Bmax,1]
        y4error = CVMAPvec[:,2]#[2:Bmax,2]
        z1 = FEvec#[2:Bmax]
        z2 = retQvec#[2:Bmax]
        z3 = MDLvec#[2:Bmax]
        PlotResults(x,w1,w2,y1,y1error,y2,y2error,y3,y3error,y4,y4error,z1,z2,z3,BwithNBT,excm,dataset)
    end

    close(fpAssessment)
    close(fpPartition)
#    close(fpNBspectra)

end # open
close(fpmeta)

#inputs: 
#    dataset = "karate club"
#    strdataset = "karateclub.txt"
#    learning = true
#    Barray = [2:6;]
#    samples = 10
#
#options: 
#    itrmax = 64
#    BPconv = 10^-6
#    initialnoise = 0.1
#    plots = true
#    alluvial = true
#    spectral = true

# Elapsed time can be measured by @time or tic()&toc(). You receive a Tkinter error, but it is a known error of Tkinter and there is no problem.