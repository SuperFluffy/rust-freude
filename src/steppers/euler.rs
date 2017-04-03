use ndarray::{ArrayBase,DataMut,DataClone,Dimension};

use traits::ODE;

use super::Stepper;

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

impl<S> Stepper for Euler<S,f64>
    where S: ODE<State=f64> + 'static
{
    type System = S;
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

macro_rules! functions_for_zippable {
    () => {
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
};}

impl<S> Stepper for Euler<S, Vec<f64>>
    where S: ODE<State=Vec<f64>> + 'static
{
    type System = S;
    type State = Vec<f64>;

    functions_for_zippable!();
}

impl<S,T,D> Stepper for Euler<T, ArrayBase<S, D>>
    where D: Dimension,
          S: DataMut<Elem=f64> + DataClone,
          T: ODE<State=ArrayBase<S, D>> + 'static,
{
    type System = T;
    type State = ArrayBase<S,D>;

    functions_for_zippable!();
}
