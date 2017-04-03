use ndarray::{ArrayBase,DataMut,DataClone,Dimension};

use traits::ODE;

use super::Stepper;

pub struct Heun<S, T> {
    dt: f64,
    dt_2: f64,

    system: S,

    temp: T,
    k1: T,
    k2: T,
}

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

impl<S> Stepper for Heun<S,f64>
    where S: ODE<State=f64> + 'static
{
    type System = S;
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

macro_rules! functions_for_zippable {
    () => {
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
};}

impl<S> Stepper for Heun<S, Vec<f64>>
    where S: ODE<State=Vec<f64>> + 'static
{
    type System = S;
    type State = Vec<f64>;

    functions_for_zippable!();
}

impl<S,T,D> Stepper for Heun<T, ArrayBase<S, D>>
    where D: Dimension,
          S: DataMut<Elem=f64> + DataClone,
          T: ODE<State=ArrayBase<S, D>> + 'static,
{
    type System = T;
    type State = ArrayBase<S,D>;

    functions_for_zippable!();
}
