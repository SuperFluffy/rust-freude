use ndarray::{
    Dimension,
    IntoNdProducer,
    Zip,
};

use crate::ode::Ode;

use super::{
    Stepper,
    ZipMarker,
};

pub struct Heun<T> {
    dt: f64,
    dt_2: f64,

    temp: T,
    k1: T,
    k2: T,
}

impl<T> Heun<T>
    where T: Clone,
{
    pub fn new(state: &T, dt: f64) -> Self {
        let temp = state.clone();
        let k1 = state.clone();
        let k2 = state.clone();

        let dt_2 = dt/2.0;

        Heun {
            dt: dt,
            dt_2: dt_2,

            temp: temp,
            k1: k1,
            k2: k2,
        }
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}

impl Stepper for Heun<f64>
{
    type State = f64;

    fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
        where Sy: Ode<State=f64>,
    {
        system.differentiate_into(state, &mut self.k1);
        system.differentiate_into(&(*state + &(self.dt * &self.k1)), &mut self.k2);
        self.temp = *state + &(self.dt_2 * &self.k1) + &(self.dt_2 * &self.k2);
        system.update_state(state, &self.temp);
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}

impl<D, P: ZipMarker> Stepper for Heun<P>
    where P: Clone,
          D: Dimension,
          for<'a> &'a P: IntoNdProducer<Dim=D, Item=&'a f64>,
          for<'a> &'a mut P: IntoNdProducer<Dim=D, Item=&'a mut f64>,
{
    type State = P;

    fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
        where Sy: Ode<State=P>,
    {
        let dt = self.dt;
        let dt_2 = self.dt_2;

        system.differentiate_into(state, &mut self.k1);

        Zip::from(&mut self.temp)
            .and(&*state)
            .and(&self.k1)
            .apply(|next_x, &x, &x_k1|
                *next_x = x + dt * x_k1
        );

        system.differentiate_into(&self.temp, &mut self.k2);

        Zip::from(&mut self.temp)
            .and(&*state)
            .and(&self.k1)
            .and(&self.k2)
            .apply(|next_x, &x, &x_k1, &x_k2|
                *next_x = x + dt_2 * (x_k1 + x_k2)
        );

        system.update_state(state, &self.temp);
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}
