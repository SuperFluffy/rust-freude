use ndarray::Dimension;
use ndarray::IntoNdProducer;

use ode::Ode;

use super::{
    RungeKutta4,
    Stepper,
    ZipMarker,
};

impl<T> RungeKutta4<T>
    where T: Clone,
{
    pub fn new(state: &T, dt: f64) -> Self {
        let dt_2 = dt/2.0;
        let dt_3 = dt/3.0;
        let dt_6 = dt/6.0;

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

impl Stepper for RungeKutta4<f64>
{
    type State = f64;

    fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
        where Sy: Ode<State=f64>,
    {
        system.differentiate_into(state, &mut self.k1);
        system.differentiate_into(&(*state + &self.k1 * self.dt_2), &mut self.k2);
        system.differentiate_into(&(*state + &self.k2 * self.dt_2), &mut self.k3);
        system.differentiate_into(&(*state + &self.k3 * self.dt), &mut self.k4);
        self.temp = *state + &(self.dt_6 * &self.k1) + &(self.dt_3 * &self.k2) +
            &(self.dt_3 * &self.k3) + &(self.dt_6 * &self.k4);
        system.update_state(state, &self.temp);
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}

impl<D, P: ZipMarker> Stepper for RungeKutta4<P>
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
        let dt_3 = self.dt_3;
        let dt_6 = self.dt_6;

        system.differentiate_into(state, &mut self.k1);

        azip!(mut t (&mut self.temp), s (&*state), k1 (&self.k1) in {
            *t = s + dt_2 * k1
        });
        system.differentiate_into(&self.temp, &mut self.k2);

        azip!(mut t (&mut self.temp), s (&*state), k2 (&self.k2) in {
            *t = s + dt_2 * k2
        });
        system.differentiate_into(&self.temp, &mut self.k3);

        azip!(mut t (&mut self.temp), s (&*state), k3 (&self.k3) in {
            *t = s + dt * k3
        });
        system.differentiate_into(&self.temp, &mut self.k4);

        azip!(mut t (&mut self.temp), s (&*state), k1 (&self.k1), k2 (&self.k2), k3 (&self.k3), k4 (&self.k4) in {
            *t = s + dt_6 * k1 + dt_3 * k2 + dt_3 * k3 + dt_6 * k4;
        });
        system.update_state(state, &self.temp);
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}
