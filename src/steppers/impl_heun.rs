use ndarray::Dimension;
use ndarray::IntoNdProducer;

use traits::ODE;

use super::{
    Heun,
    Stepper,
    ZipMarker,
};

impl<S,T> Heun<S,T>
    where S: ODE<State=T>,
          T: Clone,
{
    pub fn new(mut system: S, dt: f64, state: &T) -> Self {
        let temp = system.differentiate(&state);
        let k1 = system.differentiate(&state);
        let k2 = system.differentiate(&state);

        let dt_2 = dt/2.0;

        Heun {
            dt: dt,
            dt_2: dt_2,
            system: system,

            temp: temp,
            k1: k1,
            k2: k2,
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

impl<Sy> Stepper for Heun<Sy,f64>
    where Sy: ODE<State=f64> + 'static
{
    type System = Sy;
    type State = f64;

    fn do_step(&mut self, state: &mut Self::State) {
        self.system.differentiate_into(state, &mut self.k1);
        self.system.differentiate_into(&(*state + &(self.dt * &self.k1)), &mut self.k2);
        self.temp = *state + &(self.dt_2 * &self.k1) + &(self.dt_2 * &self.k2);
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

impl<D, S, P: ZipMarker> Stepper for Heun<S, P>
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
        let dt_2 = self.dt_2;

        self.system.differentiate_into(state, &mut self.k1);

        azip!(mut t (&mut self.temp), s (&*state), k1 (&self.k1) in {
            *t = s + dt * k1
        });
        self.system.differentiate_into(&self.temp, &mut self.k2);

        azip!(mut t (&mut self.temp), s (&*state), k1 (&self.k1), k2 (&self.k2) in {
            *t = s + dt_2 * ( k1 + k2 )
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
