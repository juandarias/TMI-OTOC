########## Comparision methods
memplot = scatterplot(collect(Dmin:5:Dmax), m_minimum_B2, name = "Mem_min_B",  yscale = :log10, xlabel = "D")
scatterplot!(memplot, collect(Dmin:5:Dmax), m_minimum, name = "Mem_min_A")


timeplot = scatterplot(collect(Dmin:5:Dmax), t_minimum_B2, name = "Time_min_B",  yscale = :log10, xlabel = "D")
scatterplot!(timeplot, collect(Dmin:5:Dmax), t_minimum, name = "Time_min_A")


time(minimum(brA))*1e-9 # nanoseconds to seconds
memory(minimum(brA))/1024^2 # bytes to MB

using LsqFit

@. model(x,p) = p[1] + p[2]*x^4
p0 = [0.5, 1.0]

fmem = curve_fit(model, collect(Dmin:5:Dmax), m_mean, p0)
sol_par = fmem.param
fit_res = model(collect(Dmin:5:Dmax), sol_par)

memplot = scatterplot(collect(Dmin:5:Dmax), m_mean, name = "Mem_mean", xlabel = "D")
scatterplot!(memplot, collect(Dmin:5:Dmax), m_minimum, name = "Mem_min")
scatterplot!(memplot, collect(Dmin:5:Dmax), fit_res, name = "Fit Mem_min")


@. model_time(x,p) = p[1] + p[2]*x^7 + p[3]*x^8

p0 = [1000.0, 0.1, 2.0]

ftime = curve_fit(model_time, collect(Dmin:5:Dmax), t_minimum, p0);
sol_time = ftime.param
fit_time = model_time(collect(Dmin:5:Dmax), sol_time)

time_plot = scatterplot(collect(Dmin:5:Dmax), t_minimum, name = "Time_min", xlabel = "D")
scatterplot!(time_plot, collect(Dmin:5:Dmax), fit_time, name = "Fit_Time_min")


fit_time = model_time(collect(Dmin:5:400), sol_time)
ns_to_min(t) = t*1e-9/3600

scatterplot(collect(Dmin:5:400), ns_to_min.(fit_time), name = "Fit_Time_min")

sf = fit(collect(Dmin:5:Dmax), t_minimum, 8)

Ds = collect(Dmin:5:400)
time_fit = sf.(Ds)
ns_to_min(time_fit)