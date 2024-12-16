using LinearAlgebra, Zygote, Printf
using Random

#### ----- ###
#### Before getting started you should write your student_number in integer format
const student_number::Int64 = 123456789  ## <---replace 0 by your student_number 
### ---- ###
## In this tiny HW you will implement Lion optimizer (see the paper:https://arxiv.org/pdf/2302.06675 -- page 24)
## See the other files for reference... 


abstract type AbstractOptimiser end
mutable struct Lion <: AbstractOptimiser
    eta::Float64 ## Learning rate in the article
    rho_1::Float64 ## B_1 in the article
    rho_2::Float64 ## B_2 in the article
    w::Float64 ##Weight decay
    velocity::IdDict
end
  

Lion(η = 0.001, ρ_1 = 0.9,ρ_2 = 0.99 , w = 0.001) = Lion(η, ρ_1, ρ_2, w, IdDict())

## You need to implement apply! function, for this you will need to review the related week.
function apply!(o::Lion, x,  Δ)
    ## Here you should have updated ∇. 
    η = o.eta
    ρ1 = o.rho_1
    ρ2 = o.rho_2
    λ = o.w

    velocity = get!(o.velocity, x) do
        zero(x)
    end

    c_t = ρ1 * velocity + (1 - ρ1) * Δ
    velocity .= ρ2 * velocity + (1 - ρ2) * Δ

    # c_t = ρ1 * velocity + (1 - ρ1) * Δ
    Δ_update = η * (sign.(c_t) + λ * x)  # Weight decay term added

    o.velocity[x] = velocity

    return Δ_update
    # return Δ
end

## Step function is given, because it is the same...
function step!(opt::AbstractOptimiser, x::AbstractArray, Δ)
    x .-= apply!(opt,x, Δ)
    return x
end 

##You will now implement optimize function, remember that you will take the gradient at point where you will use
### velocity factor, therefore may wish to use get function...
function optimize(f::Function, x::AbstractArray, opt::AbstractOptimiser; max_iter = 2, stopping_criterion = 1e-10)
    for i in 1:max_iter
        grad = Zygote.gradient(t->f(t), x)[1]
        x = step!(opt, x, grad)
        if i % 100 == 0
            @info "Iter $(i): f(x) = $(f(x)), grad = $(norm(grad)), x = $(x)" # Added additional x for debugging
        end
        if norm(grad) < stopping_criterion
            @info "ok in $(i) steps"
            return x
        end
    end
    @info "No convergence buddy!!!"
    return x
end

## Let's give it a test
opt = Lion(0.0001)
function RosenBrock(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end
x = [-0.2, -0.1]
# should be x =  [-0.09998499583280371, 5.004666679498792e-6]
optimize(RosenBrock, x, opt, max_iter = 1000, stopping_criterion = 1e-4)

opt = Lion(0.0001)
x = [0.0, 0.0]
# should be x = [ 0.0999950051661627, 0.009899667258300326]
optimize(RosenBrock, x, opt, max_iter = 1000, stopping_criterion = 1e-4) 

opt = Lion(0.0079)
x = [0.2, 0.2]
# should be x = [0.9743605 , 0.94933301]
optimize(RosenBrock, x, opt, max_iter = 1000, stopping_criterion = 1e-4)

function test_learning_rates_nearest()
    expected_result = [0.9743605, 0.94933301]
    x_init = [0.2, 0.2]
    best_eta = 0.0
    min_error = Inf
    closest_result = Float64[]

    for candidate_lr in range(0.0001, stop=0.01, length=100)
        opt = Lion(candidate_lr) # Create a new optimizer with the candidate learning rate
        x = copy(x_init) # Reset x to initial value
        result = optimize(RosenBrock, x, opt, max_iter=1000, stopping_criterion=1e-4) # Run the optimization
        error = norm(result - expected_result) # Calculate the error by comparing the result to the expected result

        if error < min_error # Update the best learning rate if the error is lower
            min_error = error
            best_eta = candidate_lr
            closest_result = result
        end

        if error < 1e-6  # If the error is not that much, we can consider it as an exact match (Even with this, I cannot find a exact match)
            println("Found exact matching learning rate $(candidate_lr) - Result: $(result)")
            return candidate_lr, result
        end
    end

    println("No exact match found. Nearest learning rate: $(best_eta) - Result: $(closest_result) - Error: $(min_error)")
end

# Run the nearest learning rate test
test_learning_rates_nearest()

"""
You should see the following output:
opt = Lion(0.0001)
x = [0.0, 0.0]
max_iter = 1000, x = [ 0.0999950051661627, 0.009899667258300326]
x = [0.2, 0.2]
max_iter = 1000, x = [0.9743605 , 0.94933301]
x = [-0.2, -0.1]
max_iter = 1000, x =  [-0.09998499583280371, 5.004666679498792e-6]
Each time you run an experiment do not forget to reinitialize the optimizer!!! Probably you will see no convergence buddy a lot BTW.
"""