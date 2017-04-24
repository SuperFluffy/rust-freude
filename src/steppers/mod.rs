use ndarray::{
    ArrayBase,
    Data,
    Dimension
};

use ode::ODE;

mod impl_euler;
mod impl_heun;
mod impl_runge_kutta_4;

#[cfg(feature="tuple")]
mod tuple_impls;

pub struct Euler<T> {
    dt: f64,

    temp: T,
}

pub struct Heun<T> {
    dt: f64,
    dt_2: f64,

    temp: T,
    k1: T,
    k2: T,
}

pub struct RungeKutta4<T> {
    dt: f64,
    dt_2: f64,
    dt_3: f64,
    dt_6: f64,

    temp: T,

    k1: T,
    k2: T,
    k3: T,
    k4: T,
}

/// A trait defining the interface of an integration method.
pub trait Stepper
{
    type State: Clone;

    fn do_step<Sy>(&mut self, &mut Sy, &mut Self::State)
        where Sy: ODE<State = Self::State> + 'static;

    fn timestep(&self) -> f64;
}

/// An internal marker trait to avoid trait impl conflicts.
pub trait ZipMarker {}

impl<T> ZipMarker for Vec<T> {}
impl<D: Dimension, S: Data> ZipMarker for ArrayBase<S,D> {}
