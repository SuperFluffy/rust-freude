use ndarray::{ArrayBase,DataMut,DataClone,Dimension};

use traits::ODE;

use super::Stepper;

pub struct Euler<S, T> {
    system: S,

    initial: T,
    temp: T,

    dt: f64,
}

impl<S,T> Euler<S,T>
    where S: ODE<State=T>,
          T: Clone
{
    pub fn new(system: S, dt: f64) -> Self {

        let initial = system.get_state().clone();
        let temp = system.get_state().clone();

        Euler {
            system: system,
            initial: initial,
            temp: temp,
            dt: dt,
        }
    }

    pub fn unwrap(self) -> (S, f64) {
        (self.system, self.dt)
    }
}

impl<S> Stepper for Euler<S,f64>
    where S: ODE<State=f64> + 'static
{
    type System = S;
    type State = f64;

    fn do_step(&mut self) {
        self.initial = self.system.get_state().clone();
        self.system.differentiate_into(&self.initial, &mut self.temp);
        self.temp = self.initial + &(self.dt * &self.temp);
        self.system.update_state(&self.temp);
    }

    fn get_state(&self) -> &Self::State {
        self.system.get_state()
    }

    fn get_state_mut(&mut self) -> &mut Self::State {
        self.system.get_state_mut()
    }

    fn get_system(&self) -> &Self::System {
        &self.system
    }

    fn get_system_mut(&mut self) -> &mut Self::System {
        &mut self.system
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}

impl<S> Stepper for Euler<S, Vec<f64>>
    where S: ODE<State=Vec<f64>> + 'static
{
    type System = S;
    type State = Vec<f64>;

    fn do_step(&mut self) {
        self.initial.copy_from_slice(self.system.get_state());

        self.system.differentiate_into(&mut self.initial, &mut self.temp);

        for (t, i) in izip!(self.temp.iter_mut(), self.initial.iter()) {
            *t = i + self.dt * *t;
        }
        self.system.update_state(&self.temp);
    }

    fn get_state(&self) -> &Self::State {
        self.system.get_state()
    }

    fn get_state_mut(&mut self) -> &mut Self::State {
        self.system.get_state_mut()
    }

    fn get_system(&self) -> &Self::System {
        &self.system
    }

    fn get_system_mut(&mut self) -> &mut Self::System {
        &mut self.system
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}

impl<S,T,D> Stepper for Euler<T, ArrayBase<S, D>>
    where D: Dimension,
          S: DataMut<Elem=f64> + DataClone,
          T: ODE<State=ArrayBase<S, D>> + 'static,
{
    type System = T;
    type State = ArrayBase<S,D>;

    fn do_step(&mut self) {
        self.initial.assign(self.system.get_state());

        self.system.differentiate_into(&self.initial, &mut self.temp);

        // Need to assign the values here, because closures try to immutably borrow the entire
        // self, which fails because self.temp is borrowed mutably.
        let dt = self.dt;

        self.temp.zip_mut_with(&self.initial.view(), |t,i,| {
            *t = i + dt * *t;
        });

        self.system.update_state(&self.temp);
    }

    fn get_state(&self) -> &Self::State {
        self.system.get_state()
    }

    fn get_state_mut(&mut self) -> &mut Self::State {
        self.system.get_state_mut()
    }

    fn get_system(&self) -> &Self::System {
        &self.system
    }

    fn get_system_mut(&mut self) -> &mut Self::System {
        &mut self.system
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}
