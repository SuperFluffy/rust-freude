use ndarray::Dimension;
use ndarray::IntoNdProducer;

use ode::Ode;

use super::{
    Euler,
    Stepper,
    ZipMarker,
};

impl<T> Euler<T>
    where T: Clone,
{
    pub fn new(state: &T, dt: f64) -> Self {
        let temp = state.clone();

        Euler {
            dt: dt,

            temp: temp,
        }
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}

impl Stepper for Euler<f64>
{
    type State = f64;

    fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
        where Sy: Ode<State=f64> + 'static,
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
    where P: Clone,
          D: Dimension,
          for<'a> &'a P: IntoNdProducer<Dim=D, Item=&'a f64>,
          for<'a> &'a mut P: IntoNdProducer<Dim=D, Item=&'a mut f64>,
{
    type State = P;

    fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
        where Sy: Ode<State=P> + 'static,
    {
        let dt = self.dt;
        system.differentiate_into(state, &mut self.temp);

        azip!(mut t (&mut self.temp), s (&*state) in {
            *t = s + dt * *t;
        });

        system.update_state(state, &self.temp);
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}
