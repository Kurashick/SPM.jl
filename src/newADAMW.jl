
"""
    Adam(η = 0.001, β = (0.9, 0.999), ϵ = 1e-8)

[Adam](https://arxiv.org/abs/1412.6980) optimiser.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
@def struct Adam <: AbstractRule
  eta = 0.001
  beta = (0.9, 0.999)
  epsilon = 1e-8
end

init(o::Adam, x::AbstractArray{T}) where T = (zero(x), zero(x), T.(o.beta))

function apply!(o::Adam, state, x::AbstractArray{T}, dx) where T
  η, β, ϵ = T(o.eta), T.(o.beta), T(o.epsilon)
  mt, vt, βt = state

  @.. mt = β[1] * mt + (1 - β[1]) * dx
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(dx)
  dx′ = @lazy mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ) * η

  return (mt, vt, βt .* β), dx′
end








"""
    WeightDecay(γ = 5e-4)

Decay weights by ``γ``, that is, add `γ .* x` to the gradient `x̄` which will be
subtracted from `x`.

Typically composed  with other optimisers as the first transformation in an [`OptimiserChain`](@ref).
This is equivalent to adding ``L_2`` regularization with coefficient ``γ`` to the loss.

# Parameters
- Weight decay (`γ`): Decay applied to weights during optimisation.
"""
@def struct WeightDecay <: AbstractRule
  gamma = 5e-4
end

init(o::WeightDecay, x::AbstractArray) = nothing

function apply!(o::WeightDecay, state, x::AbstractArray{T}, dx) where T
  γ = T(o.gamma)
  dx′ = @lazy dx + γ * x

  return state, dx′
end


"""
    AdamW(η = 0.001, β = (0.9, 0.999), γ = 0, ϵ = 1e-8)

[AdamW](https://arxiv.org/abs/1711.05101) is a variant of Adam fixing (as in repairing) its
weight decay regularization.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- Weight decay (`γ`): Decay applied to weights during optimisation.
- Machine epsilon (`ϵ`): Constant to prevent division by zero
                         (no need to change default)
"""
AdamW(η = 0.001, β = (0.9, 0.999), γ = 0, ϵ = 1e-8) =
  OptimiserChain(Adam(η, β, ϵ), WeightDecay(γ))









"""
    AdamW(η = 0.001, β::Tuple = (0.9, 0.999), decay = 0)

[AdamW](https://arxiv.org/abs/1711.05101) is a variant of Adam fixing (as in repairing) its
weight decay regularization.

# Parameters
- Learning rate (`η`): Amount by which gradients are discounted before updating
                       the weights.
- Decay of momentums (`β::Tuple`): Exponential decay for the first (β1) and the
                                   second (β2) momentum estimate.
- `decay`: Decay applied to weights during optimisation.

# Examples
```julia
opt = AdamW()

opt = AdamW(0.001, (0.89, 0.995), 0.1)
```
"""
AdamW(η = 0.001, β = (0.9, 0.999), decay = 0) =
  Optimiser(Adam(η, β), WeightDecay(decay))
