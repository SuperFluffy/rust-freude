use ndarray::Dimension;
use ndarray::IntoNdProducer;

use super::ODE;

use super::{
    Stepper,
    ZipMarker,
};

pub struct Euler<S, T> {
    dt: f64,
    system: S,

    temp: T,
}

impl<S,T> Euler<S,T>
    where S: ODE<State=T>,
          T: Clone,
{
    pub fn new(mut system: S, dt: f64, state: &T) -> Self {
        let temp = system.differentiate(&state);

        Euler {
            dt: dt,
            system: system,

            temp: temp,
        }
    }

    fn system_ref(&self) -> &S {
        &self.system
    }

    fn system_mut(&mut self) -> &mut S {
        &mut self.system
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}

impl<Sy> Stepper for Euler<Sy,f64>
    where Sy: ODE<State=f64> + 'static,
{
    type System = Sy;
    type State = f64;

    fn do_step(&mut self, state: &mut Self::State) {
        self.system.differentiate_into(state, &mut self.temp);
        self.temp = *state + &(self.dt * &self.temp);
        self.system.update_state(state, &self.temp);
    }

    fn system_ref(&self) -> &Self::System {
        self.system_ref()
    }

    fn system_mut(&mut self) -> &mut Self::System {
        self.system_mut()
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}

impl<D, S, P: ZipMarker> Stepper for Euler<S, P>
    where S: ODE<State=P> + 'static,
          P: Clone,
          D: Dimension,
          for<'a> &'a P: IntoNdProducer<Dim=D, Item=&'a f64>,
          for<'a> &'a mut P: IntoNdProducer<Dim=D, Item=&'a mut f64>,
{
    type System = S;
    type State = P;

    fn do_step(&mut self, state: &mut Self::State) {
        let dt = self.dt;
        self.system.differentiate_into(state, &mut self.temp);

        azip!(mut t (&mut self.temp), s (&*state) in {
            *t = s + dt * *t;
        });

        self.system.update_state(state, &self.temp);
    }

    fn system_ref(&self) -> &Self::System {
        self.system_ref()
    }

    fn system_mut(&mut self) -> &mut Self::System {
        self.system_mut()
    }

    fn timestep(&self) -> f64 {
        self.timestep()
    }
}
