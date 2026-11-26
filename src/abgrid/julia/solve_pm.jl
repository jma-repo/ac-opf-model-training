# Cross-check: solve a MATPOWER case with PowerModels.jl (ACPPowerModel) + Ipopt.
# usage: julia solve_pm.jl case.m out.json
using PowerModels, Ipopt, JSON
PowerModels.silence()
data = PowerModels.parse_file(ARGS[1])
t = @elapsed res = solve_ac_opf(data, optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "max_iter" => 3000))
sol = res["solution"]
out = Dict("termination_status" => string(res["termination_status"]),
           "objective" => res["objective"], "solve_time_s" => t,
           "vm" => Dict(k => b["vm"] for (k, b) in sol["bus"]),
           "pg" => Dict(k => g["pg"] * data["baseMVA"] for (k, g) in sol["gen"]))
open(ARGS[2], "w") do f; JSON.print(f, out); end
println(out["termination_status"], " obj=", out["objective"], " t=", round(t, digits=2), "s")
