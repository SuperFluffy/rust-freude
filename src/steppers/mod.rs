use traits::ODE;

mod euler;
mod heun;
mod runge_kutta_4;

pub use self::euler::*;
pub use self::heun::*;
pub use self::runge_kutta_4::*;

/// A trait defining the interface of and integration method.
pub trait Stepper {
    type System: ODE<State = Self::State>;
    type State: Clone;

    fn do_step(&mut self, &mut Self::State);

    fn system_ref(&self) -> &Self::System;
    fn system_mut(&mut self) -> &mut Self::System;

    fn timestep(&self) -> f64;
}
