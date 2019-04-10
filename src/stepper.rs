use ndarray::{ArrayBase, Data, Dimension};

use crate::ode::Ode;

mod euler;
mod heun;
mod runge_kutta_4;

pub use euler::Euler;
pub use heun::Heun;
pub use runge_kutta_4::RungeKutta4;

#[cfg(feature = "tuple")]
mod tuple_impls;

/// A trait defining the interface of an integration method.
pub trait Stepper {
    type State: Clone;

    fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
    where
        Sy: Ode<State = Self::State>;

    fn timestep(&self) -> f64;

    fn integrate_n_steps<Sy>(&mut self, system: &mut Sy, state: &mut Self::State, n: usize) -> f64
    where
        Sy: Ode<State = Self::State>,
    {
        let mut tacc = 0f64;;

        let dt = self.timestep();

        // Ensure t is not exceeded
        for _ in 0..n {
            self.do_step(system, state);
            tacc += dt;
        }
        tacc
    }

    fn integrate_time<Sy>(
        &mut self,
        system: &mut Sy,
        state: &mut Self::State,
        t: f64,
    ) -> (f64, usize)
    where
        Sy: Ode<State = Self::State>,
    {
        let mut tacc = 0f64;;
        let mut count = 0;

        let dt = self.timestep();

        // Ensure t is not exceeded
        while (tacc + dt) <= t {
            self.do_step(system, state);
            tacc += dt;
            count += 1;
        }
        (tacc, count)
    }
}

/// An internal marker trait to avoid trait impl conflicts.
pub trait ZipMarker {}

impl<T> ZipMarker for Vec<T> {}
impl<D: Dimension, S: Data> ZipMarker for ArrayBase<S, D> {}
