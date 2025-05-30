function loadstate(folder, file_name, step)
    data = h5open(folder*file_name*"_state_step=$(step).h5", "r");
    state = reconstructcomplexState(read(data["mps"]));
    return state
end

function readBonddimension(path)
    data = h5open(path*"_observables.h5","r")
    return read(data["Diagnostics/Bond dimension"])    
end

"""
    Returns the von Neumann entropy at each bond.
"""
function readEntanglement(path)
    data = h5open(path*"_observables.h5","r")
    return read(data["Entropy/Bond entropy"])    
end


function readOTOC(L, alpha, NSV, step, location, ALG; DeltaS=0.1, eps_tw=1e-3, OTOC="A", rerun=false)
    ALG=="SS" && rerun==false && (filename = "single_site_observables.h5");
    ALG=="SS" && rerun==true && (filename = "single_site_rerun_observables.h5");
    ALG=="ADAP" && (filename = "adaptive_DeltaS=$(DeltaS)_eps_tw_$(eps_tw)_observables.h5");
    path = location*"/L$L/alpha_$alpha/NSV_$NSV/OTOC_step$step/OTOC_$OTOC/$filename";
    data = h5open(path, "r");
    OTOCs = read(data["OTOC"])
    return OTOCs
end

function readOTOC(α, AorB; step_end=100)
    path = datadir("OTOCS/OTOC_$(α)_$AorB", "single_site_OTOCS.h5")
    data = h5open(path,"r")

    AorB == "A"  && (OTOC = zeros(31,20);)
    AorB == "B"  && (OTOC = zeros(15,20);)
    for (i,tstep) ∈ enumerate(5:5:step_end)
        OTOC[:,i] = read(data["OTOC/step_$tstep"])
    end
    return OTOC
end



function readEntanglement(L, alpha, NSV, location, ALG; DeltaS=0.1, eps_tw=1e-3)
    ALG=="SS" && (filename = "single_site_forward_observables.h5");
    ALG=="ADAP" && (filename = "adaptive_DeltaS=$(DeltaS)_eps_tw_$(eps_tw)_observables.h5");
    path = location*"/L$L/alpha_$alpha/NSV_$NSV/$filename";
    
    data = h5open(path,"r")
    S = read(data["Entropy/Bond entropy"])
    M = read(data["Diagnostics/Bond dimension"]) 
    return S, M
end


"""
# Returns magnetizations [Sx, Sy, Sz], entropy, and Dmax
"""
function readObservables(L, alpha, NSV, location, ALG; DeltaS=0.1, eps_tw=1e-3)
    ALG=="SS" && (filename = "single_site_observables.h5");
    ALG=="ADAP" && (filename = "adaptive_DeltaS=$(DeltaS)_eps_tw_$(eps_tw)_observables.h5");
    path = location*"/L$L/alpha_$alpha/NSV_$NSV/$filename";
    
    data = h5open(path,"r")
    Sx = read(data["Magnetization/Sx"])
    Sy = read(data["Magnetization/Sy"])
    Sz = read(data["Magnetization/Sz"])
    S = read(data["Entropy/Bond entropy"])
    M = read(data["Diagnostics/Bond dimension"]) 
    return [Sx, Sy, Sz], S, M
end

"""
# Returns OTOCs, [Sx, Sy, Sz], entropy, and Dmax
"""
function readObservables(L, alpha, NSV, step, location, ALG; DeltaS=0.1, eps_tw=1e-3, OTOC="A")
    ALG=="SS" && (filename = "single_site_observables.h5");
    ALG=="ADAP" && (filename = "adaptive_DeltaS=$(DeltaS)_eps_tw_$(eps_tw)_observables.h5");
    path = location*"/L$L/alpha_$alpha/NSV_$NSV/OTOC_step$step/OTOC_$OTOC/$filename";
    data = h5open(path, "r");
    Sx = read(data["Magnetization/Sx"])
    Sy = read(data["Magnetization/Sy"])
    Sz = read(data["Magnetization/Sz"])
    S = read(data["Entropy/Bond entropy"])
    M = read(data["Diagnostics/Bond dimension"]) 
    OTOCs = read(data["OTOC"])
    return OTOCs, [Sx, Sy, Sz], S, M
end



function loadBraStateOTOC(L, alpha, NSV, step, location, ALG; DeltaS=0.1, eps_tw=1e-3, OTOC="A")
    ALG=="SS" && (filename = "single_site_state_step=0.h5");
    ALG=="ADAP" && (filename = "adaptive_DeltaS=$(DeltaS)_eps_tw_$(eps_tw)_state_step=0.h5");
    if location == "obelix"
        path = "/mnt/obelix/TMI/L$L/alpha_$alpha/NSV_$NSV/OTOC_step$step/OTOC_$OTOC/$filename";
    elseif location == "lisa"
        path = "/mnt/lisa/L$L/alpha_$alpha/NSV_$NSV/OTOC_step$step/OTOC_$OTOC/$filename";
    end
    data = h5open(path, "r");
    mps = read(data["mps"])
    return mps
end

function loadKetStateOTOC(L, alpha, NSV, step, loc_V, loc_W, location, ALG; DeltaS=0.1, eps_tw=1e-3, OTOC="A")
    ALG=="SS" && (filename = "single_site_V_$(loc_V)W(t)_$loc_W.h5");
    ALG=="ADAP" && (filename = "adaptive_DeltaS=$(DeltaS)_eps_tw_$(eps_tw)_V_$(loc_V)W(t)_$loc_W.h5");
    if location == "obelix"
        path = "/mnt/obelix/TMI/L$L/alpha_$alpha/NSV_$NSV/OTOC_step$step/OTOC_$OTOC/$filename";
    elseif location == "lisa"
        path = "/mnt/lisa/L$L/alpha_$alpha/NSV_$NSV/OTOC_step$step/OTOC_$OTOC/$filename";
    end
    data = h5open(path, "r");
    mps = read(data["mps"])
    return mps
end


function loadStateReverse(L, alpha, NSV, step, step_reverse, location, ALG; DeltaS=0.1, eps_tw=1e-3, OTOC="A")
    ALG=="SS" && (filename = "single_site_state_step=$step_reverse.h5");
    ALG=="ADAP" && (filename = "adaptive_DeltaS=$(DeltaS)_eps_tw_$(eps_tw)_state_step=$step_reverse.h5");

    if location == "obelix"
        path = "/mnt/obelix/TMI/L$L/alpha_$alpha/NSV_$NSV/OTOC_step$step/OTOC_$OTOC/$filename";
    elseif location == "lisa"
        path = "/mnt/lisa/L$L/alpha_$alpha/NSV_$NSV/OTOC_step$step/OTOC_$OTOC/$filename";
    end
    data = h5open(path, "r");
    mps = read(data["mps"])
    return mps
end

function loadStateForward(L, alpha, NSV, step, location, ALG; DeltaS=0.1, eps_tw=1e-3, OTOC="A")
    ALG=="SS" && (filename = "single_site_state_step=$step.h5");
    ALG=="ADAP" && (filename = "adaptive_DeltaS=$(DeltaS)_eps_tw_$(eps_tw)_state_step=$step.h5");

    if location == "obelix"
        path = "/mnt/obelix/TMI/L$L/alpha_$alpha/NSV_$NSV/$filename";
    elseif location == "lisa"
        path = "/mnt/lisa/L$L/alpha_$alpha/NSV_$NSV/$filename";
    end
    data = h5open(path, "r");
    mps = read(data["mps"])
    return mps
end
