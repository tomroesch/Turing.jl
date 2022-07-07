###
### DynamicHMC backend - https://github.com/tpapp/DynamicHMC.jl
###

"""
    DynamicNUTS

Dynamic No U-Turn Sampling algorithm provided by the DynamicHMC package.

To use it, make sure you have DynamicHMC package (version >= 2) loaded:
```julia
using DynamicHMC
```
""" 
struct DynamicNUTS{AD,space} <: Hamiltonian{AD} end

DynamicNUTS(args...) = DynamicNUTS{ADBackend()}(args...)
DynamicNUTS{AD}(space::Symbol...) where AD = DynamicNUTS{AD, space}()

DynamicPPL.getspace(::DynamicNUTS{<:Any, space}) where {space} = space

# Only define traits for `DynamicNUTS` sampler to avoid type piracy and surprises
# TODO: Implement generally with `LogDensityProblems`
const DynamicHMCLogDensity{M<:Model,S<:Sampler{<:DynamicNUTS},V<:AbstractVarInfo} = Turing.SamplingLogDensityFunction{V,M,S,DynamicPPL.DefaultContext}

function DynamicHMC.dimension(ℓ::DynamicHMCLogDensity)
    return length(ℓ.varinfo[ℓ.sampler])
end

function DynamicHMC.capabilities(::Type{<:DynamicHMCLogDensity})
    return DynamicHMC.LogDensityOrder{1}()
end

function DynamicHMC.logdensity_and_gradient(
    ℓ::DynamicHMCLogDensity,
    x::AbstractVector,
)
    return gradient_logp(x, ℓ.varinfo, ℓ.model, ℓ.sampler, ℓ.context)
end

"""
    DynamicNUTSState

State of the [`DynamicNUTS`](@ref) sampler.

# Fields
$(TYPEDFIELDS)
"""
struct DynamicNUTSState{V<:AbstractVarInfo,C,M,S}
    vi::V
    "Cache of sample, log density, and gradient of log density."
    cache::C
    metric::M
    stepsize::S
end

# Implement interface of `Gibbs` sampler
function gibbs_state(
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    state::DynamicNUTSState,
    varinfo::AbstractVarInfo,
)
    # Update the previous evaluation.
    ℓ = Turing.LogDensityFunction(varinfo, model, spl, DynamicPPL.DefaultContext())
    Q = DynamicHMC.evaluate_ℓ(ℓ, varinfo[spl])
    return DynamicNUTSState(varinfo, Q, state.metric, state.stepsize)
end

DynamicPPL.initialsampler(::Sampler{<:DynamicNUTS}) = SampleFromUniform()

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    vi::AbstractVarInfo;
    kwargs...
)
    # Ensure that initial sample is in unconstrained space.
    if !DynamicPPL.islinked(vi, spl)
        # TODO(torfjelde): Update this once we have a way of transforming variables.
        DynamicPPL.link!(vi, spl)
        model(rng, vi, spl)
    end

    # Perform initial step.
    ℓ = Turing.LogDensityFunction(vi, model, spl, DynamicPPL.DefaultContext())
    results = DynamicHMC.mcmc_keep_warmup(
        rng,
        ℓ,
        0;
        initialization = (q = first(DynamicPPL.linearize(vi)),),
        reporter = DynamicHMC.NoProgressReport(),
    )
    steps = DynamicHMC.mcmc_steps(results.sampling_logdensity, results.final_warmup_state)
    Q, _ = DynamicHMC.mcmc_next_step(steps, results.final_warmup_state.Q)

    # Update the variables.
    vi = DynamicPPL.setlogp!!(ℓ.unflatten(Q.q), Q.ℓq)

    # Create first sample and state.
    sample = Transition(vi)
    state = DynamicNUTSState(vi, Q, steps.H.κ, steps.ϵ)

    return sample, state
end

function AbstractMCMC.step(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:DynamicNUTS},
    state::DynamicNUTSState;
    kwargs...
)
    # Compute next sample.
    vi = state.vi
    ℓ = Turing.LogDensityFunction(vi, model, spl, DynamicPPL.DefaultContext())
    steps = DynamicHMC.mcmc_steps(
        rng,
        DynamicHMC.NUTS(),
        state.metric,
        ℓ,
        state.stepsize,
    )
    Q, _ = DynamicHMC.mcmc_next_step(steps, state.cache)

    # Update the variables.
    vi = DynamicPPL.setlogp!!(ℓ.unflatten(Q.q), Q.ℓq)

    # Create next sample and state.
    sample = Transition(vi)
    newstate = DynamicNUTSState(vi, Q, state.metric, state.stepsize)

    return sample, newstate
end
