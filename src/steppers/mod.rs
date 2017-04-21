use ndarray::{
    ArrayBase,
    Data,
    Dimension
};

use traits::ODE;

mod euler;
mod heun;
mod runge_kutta_4;

pub use self::euler::*;
pub use self::heun::*;
pub use self::runge_kutta_4::*;

/// A trait defining the interface of an integration method.
pub trait Stepper
{
    type State: Clone;
    type System: ODE<State = Self::State>;

    fn do_step(&mut self, &mut Self::State);

    fn system_ref(&self) -> &Self::System;
    fn system_mut(&mut self) -> &mut Self::System;

    fn timestep(&self) -> f64;
}

/// An internal marker trait to avoid trait impl conflicts.
pub trait ZipMarker {}

impl<T> ZipMarker for Vec<T> {}
impl<D: Dimension, S: Data> ZipMarker for ArrayBase<S,D> {}
