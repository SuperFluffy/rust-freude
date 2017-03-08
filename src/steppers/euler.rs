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

impl<S> Stepper for Euler<S, Vec<f64>>
    where S: ODE<State=Vec<f64>> + 'static
{
    type System = S;
    type State = Vec<f64>;

    fn do_step(&mut self, state: &mut Self::State) {
        self.system.differentiate_into(state, &mut self.temp);

        for (t, s) in izip!(self.temp.iter_mut(), state.iter()) {
            *t = s + self.dt * *t;
        }
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

impl<S,T,D> Stepper for Euler<T, ArrayBase<S, D>>
    where D: Dimension,
          S: DataMut<Elem=f64> + DataClone,
          T: ODE<State=ArrayBase<S, D>> + 'static,
{
    type System = T;
    type State = ArrayBase<S,D>;

    fn do_step(&mut self, state: &mut Self::State) {
        self.system.differentiate_into(state, &mut self.temp);

        // Need to assign the values here, because closures try to immutably borrow the entire
        // self, which fails because self.temp is borrowed mutably.
        let dt = self.dt;

        self.temp.zip_mut_with(&state.view(), |t,s| {
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
