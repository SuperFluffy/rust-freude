# freude

The `freude` crate will provide steppers and integrators to solve ODEs
(ordinary differential equations). It is inspired by `boost::numeric::odeint`.
At the moment it only supports the explicit, fixed-step Euler, Heun, and 4-th
order Runge Kutta (RK4) methods.

## Todo

Pretty much everything. :-)

## Recent changes

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
