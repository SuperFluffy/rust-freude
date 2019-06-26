# freude

The `freude` crate will provide steppers and integrators to solve ODEs
(ordinary differential equations). It is inspired by `boost::numeric::odeint`.

## Features

+ Explicit fixed-step ODE solvers:
    + Euler
    + Heun
    + Classical 4-th order Runge Kutta (RK4)

## Todo:

+ Implicit methods
+ Adaptive steppers
    + DOPRI, RKF45
+ Symplectic solvers
+ Generalized Runge Kutta methods (maybe via Butcher tableaus?)

## Recent changes

+ 0.7.0
    + Require Debug bounds on the steppers (breaking change)
+ 0.6.0
    + Update ergonomics
+ 0.5.0
    + Require `ndarray` 0.11, bump all related dependencies
+ 0.4.0-dev
    + Update benchmarks to work with v0.4.0
+ 0.4.0
    + Complete rework and simplification of the `Ode` and `Stepper` logic
        + `Stepper` no longer contains an `Ode` system but acts on `Ode::State` borrows
        + Removal of `Integrator`: absorbed into `Stepper`
        + Removal of `Observer` trait
    + Bump to `ndarray 0.10` 
+ 0.3.1
    + Implement steppers to work on tuples as defined in the [tuple crate](https://crates.io/crates/tuple);
    + Implement `ODE` trait for generic functions/closures on tuples.
+ 0.3.0
    + Update to `ndarray 0.9`
    + Unify `Vec` and `ArrayBase` stepper states through `ndarray`'s `Zip` and `IntoNdProducer` traits
    + Provide several examples as benchmarks (Kuramoto and Chaotic Neural Network models)
+ 0.2.0
    + Complete rework of the ODE, stepper, and integrator logic;
    + System state is no longer considered an internal property of an ODE, but a
    parameter passed to the stepper.
+ 0.1.1
    + Implement Euler and Heun methods
+ 0.1.0
    + Initial release
    + Definition of explicit, fixed step integrators, steppers
    + Definition of ODEs
    + Implementation of Runge-Kutta-4 method

## Trivia

The crate's name `freude` is inspired by Beethoven's *Ode an die Freude* (“Ode to
Joy”), and can be pronounced either /ˈfʀɔɪ̯də/ or, alternatively, *froy-D-E* (as
in O-D-E).
