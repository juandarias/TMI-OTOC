module scrambling_methods

    using Interpolations, LsqFit

    export velocityOTOC

    function velocityOTOC(OTOC, d_i, d_f, AorB; tfinal=10)
        t_range = 0.5:0.5:tfinal
        AorB=="A" && (d_range = 1:1:31; dmax=31);
        AorB=="B" && (d_range = 1:1:15; dmax=15);
        itp = interpolate(OTOC, BSpline(Cubic(Line(OnGrid()))));
        sitp = scale(itp, d_range, t_range)

        t_v_OTOC = zeros(d_f-d_i+1);
        for (i,d) ∈ enumerate(d_i:d_f)
            for t ∈ 0.5:0.005:tfinal
                C_t = round(sitp(d,t), digits=2)
                C_t == 0.5 && (t_v_OTOC[i] = t;)
            end
        end

        num_data = d_f-d_i+1;
        model(t, a) = a[1]*ones(num_data) + a[2]*t;
        a0 = [dmax, 0.4]
        fit = curve_fit(model, collect(d_i:d_f), t_v_OTOC, a0)
        sol = fit.param
        
        solmodel(t,a) = a[1]*ones(dmax) + a[2]*t;
        t_fitted = solmodel(d_range, sol)
        
        return sol, t_v_OTOC, t_fitted
    end
end