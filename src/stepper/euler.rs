use ndarray::{Dimension, IntoNdProducer, Zip};
use std::fmt::Debug;

use crate::ode::Ode;

use super::{Stepper, ZipMarker};

pub struct Euler<T: Debug> {
    pub(crate) dt: f64,

    pub(crate) temp: T,
}

impl<T> Euler<T>
where
    T: Clone + Debug,
{
    pub fn new(state: &T, dt: f64) -> Self {
        let temp = state.clone();

        Euler { dt: dt, temp: temp }
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}

impl Stepper for Euler<f64> {
    type State = f64;

    fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
    where
        Sy: Ode<State = f64>,
    {
        system.differentiate_into(state, &mut self.temp);
        self.temp = *state + &(self.dt * &self.temp);
        system.update_state(state, &self.temp);
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}

impl<D, P: ZipMarker> Stepper for Euler<P>
where
    P: Clone + Debug,
    D: Dimension,
    for<'a> &'a P: IntoNdProducer<Dim = D, Item = &'a f64>,
    for<'a> &'a mut P: IntoNdProducer<Dim = D, Item = &'a mut f64>,
{
    type State = P;

    fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
    where
        Sy: Ode<State = P>,
    {
        let dt = self.dt;
        system.differentiate_into(state, &mut self.temp);

        Zip::from(&mut self.temp)
            .and(&*state)
            .apply(|next_x, &x| *next_x = x + dt * *next_x);

        system.update_state(state, &self.temp);
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}
