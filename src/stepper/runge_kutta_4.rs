use ndarray::{Dimension, IntoNdProducer, Zip};

use crate::ode::Ode;

use super::{Stepper, ZipMarker};

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

impl<T> RungeKutta4<T>
where
    T: Clone,
{
    pub fn new(state: &T, dt: f64) -> Self {
        let dt_2 = dt / 2.0;
        let dt_3 = dt / 3.0;
        let dt_6 = dt / 6.0;

        let temp = state.clone();
        let k1 = state.clone();
        let k2 = state.clone();
        let k3 = state.clone();
        let k4 = state.clone();

        RungeKutta4 {
            dt: dt,
            dt_2: dt_2,
            dt_3: dt_3,
            dt_6: dt_6,

            temp: temp,
            k1: k1,
            k2: k2,
            k3: k3,
            k4: k4,
        }
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}

impl Stepper for RungeKutta4<f64> {
    type State = f64;

    fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
    where
        Sy: Ode<State = f64>,
    {
        system.differentiate_into(state, &mut self.k1);
        system.differentiate_into(&(*state + &self.k1 * self.dt_2), &mut self.k2);
        system.differentiate_into(&(*state + &self.k2 * self.dt_2), &mut self.k3);
        system.differentiate_into(&(*state + &self.k3 * self.dt), &mut self.k4);
        self.temp = *state
            + &(self.dt_6 * &self.k1)
            + &(self.dt_3 * &self.k2)
            + &(self.dt_3 * &self.k3)
            + &(self.dt_6 * &self.k4);
        system.update_state(state, &self.temp);
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}

impl<D, P: ZipMarker> Stepper for RungeKutta4<P>
where
    P: Clone,
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
        let dt_2 = self.dt_2;
        let dt_3 = self.dt_3;
        let dt_6 = self.dt_6;

        system.differentiate_into(state, &mut self.k1);

        Zip::from(&mut self.temp)
            .and(&*state)
            .and(&self.k1)
            .apply(|next_x, &x, &x_k1| *next_x = x + dt_2 * x_k1);

        system.differentiate_into(&self.temp, &mut self.k2);

        Zip::from(&mut self.temp)
            .and(&*state)
            .and(&self.k2)
            .apply(|next_x, &x, &x_k2| *next_x = x + dt_2 * x_k2);

        system.differentiate_into(&self.temp, &mut self.k3);

        Zip::from(&mut self.temp)
            .and(&*state)
            .and(&self.k3)
            .apply(|next_x, &x, &x_k3| *next_x = x + dt * x_k3);

        system.differentiate_into(&self.temp, &mut self.k4);

        Zip::from(&mut self.temp)
            .and(&*state)
            .and(&self.k1)
            .and(&self.k2)
            .and(&self.k3)
            .and(&self.k4)
            .apply(|next_x, &x, &x_k1, &x_k2, &x_k3, &x_k4| {
                *next_x = x + dt_6 * x_k1 + dt_3 * x_k2 + dt_3 * x_k3 + dt_6 * x_k4
            });

        system.update_state(state, &self.temp);
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}
