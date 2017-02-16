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

    fn do_step(&mut self);

    fn timestep(&self) -> f64;

    fn get_state(&self) -> &Self::State;
    fn get_state_mut(&mut self) -> &mut Self::State;

    fn get_system(&self) -> &Self::System;
    fn get_system_mut(&mut self) -> &mut Self::System;
}
