use ndarray::{ArrayBase,DataMut,DataClone,Dimension};

use traits::ODE;
use utils::{zip_mut_with_2,zip_mut_with_3};

use super::Stepper;

pub struct Heun<S, T> {
    system: S,

    initial: T,
    temp: T,
    k1: T,
    k2: T,

    dt: f64,
    dt_2: f64,
}

impl<S,T> Heun<S,T>
    where S: ODE<State=T>,
          T: Clone
{
    pub fn new(system: S, dt: f64) -> Self {

        let initial = system.get_state().clone();
        let temp = system.get_state().clone();
        let k1 = system.get_state().clone();
        let k2 = system.get_state().clone();

        Heun {
            system: system,
            initial: initial,
            temp: temp,
            k1: k1,
            k2: k2,
            dt: dt,
            dt_2: dt/2.0,
        }
    }

    pub fn unwrap(self) -> (S, f64) {
        (self.system, self.dt)
    }
}

impl<S> Stepper for Heun<S,f64>
    where S: ODE<State=f64> + 'static
{
    type System = S;
    type State = f64;

    fn do_step(&mut self) {
        self.initial = self.system.get_state().clone();
        self.system.differentiate_into(&self.initial, &mut self.k1);
        self.system.differentiate_into(&(self.initial + &(self.dt * &self.k1)), &mut self.k2);
        self.temp = &self.initial + &(self.dt_2 * &self.k1) + &(self.dt_2 * &self.k2);
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

impl<S> Stepper for Heun<S, Vec<f64>>
    where S: ODE<State=Vec<f64>> + 'static
{
    type System = S;
    type State = Vec<f64>;

    fn do_step(&mut self) {
        self.initial.copy_from_slice(self.system.get_state());

        self.system.differentiate_into(&mut self.initial, &mut self.k1);

        for (t, i, k) in izip!(self.temp.iter_mut(), self.initial.iter(), self.k1.iter()) {
            *t = *i + self.dt * k;
        }
        self.system.differentiate_into(&mut self.temp, &mut self.k2);

        for (t, i, k1, k2) in izip!(self.temp.iter_mut(),
            self.initial.iter(),
            self.k1.iter(),
            self.k2.iter())
        {
            *t = i + self.dt_2 * k1 + self.dt_2 * k2;
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

impl<S,T,D> Stepper for Heun<T, ArrayBase<S, D>>
    where D: Dimension,
          S: DataMut<Elem=f64> + DataClone,
          T: ODE<State=ArrayBase<S, D>> + 'static,
{
    type System = T;
    type State = ArrayBase<S,D>;

    fn do_step(&mut self) {
        self.initial.assign(self.system.get_state());

        self.system.differentiate_into(&self.initial, &mut self.k1);

        // Need to assign the values here, because closures try to immutably borrow the entire
        // self, which fails because self.temp is borrowed mutably.
        let dt = self.dt;
        let dt_2 = self.dt_2;

        zip_mut_with_2(&mut self.temp, self.initial.view(), self.k1.view(), |t,i,k| {
            *t = i + dt * k;
        }).unwrap();
        self.system.differentiate_into(&self.temp, &mut self.k2);

        zip_mut_with_3(&mut self.temp, self.initial.view(), self.k1.view(), self.k2.view(),
            |t,i,k1,k2| {
                *t = i + dt_2 * k1 + dt_2 * k2;
        }).unwrap();

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
