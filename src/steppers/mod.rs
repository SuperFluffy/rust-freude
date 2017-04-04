<<<<<<< HEAD
use ndarray::{
    ArrayBase,
    Data,
    Dimension
};

use traits::ODE;

mod impl_euler;
mod impl_heun;
mod impl_runge_kutta_4;

#[cfg(feature = "tuple")]
mod impl_tuples;

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

pub struct Euler<S, T> {
    dt: f64,
    system: S,

    temp: T,
}

pub struct Heun<S, T> {
    dt: f64,
    dt_2: f64,

    system: S,

    temp: T,
    k1: T,
    k2: T,
}

pub struct RungeKutta4<S, T> {
    dt: f64,
    dt_2: f64,
    dt_3: f64,
    dt_6: f64,

    system: S,
    temp: T,

    k1: T,
    k2: T,
    k3: T,
    k4: T,
}
