use ndarray::{ArrayBase,DataMut,DataClone,Dimension};

use traits::ODE;
use utils::{zip_mut_with_2,zip_mut_with_5};

use super::Stepper;

pub struct RungeKutta4<S, T> {
    system: S,

    initial: T,
    temp: T,
    k1: T,
    k2: T,
    k3: T,
    k4: T,

    dt: f64,
    dt_2: f64,
    dt_3: f64,
    dt_6: f64,
}

impl<S,T> RungeKutta4<S,T>
    where S: ODE<State=T>,
          T: Clone
{
    pub fn new(system: S, dt: f64) -> Self {

        let initial = system.get_state().clone();
        let temp = system.get_state().clone();
        let k1 = system.get_state().clone();
        let k2 = system.get_state().clone();
        let k3 = system.get_state().clone();
        let k4 = system.get_state().clone();

        RungeKutta4 {
            system: system,
            initial: initial,
            temp: temp,
            k1: k1,
            k2: k2,
            k3: k3,
            k4: k4,
            dt: dt,
            dt_2: dt/2.0,
            dt_3: dt/3.0,
            dt_6: dt/6.0,
        }
    }

    pub fn unwrap(self) -> (S, f64) {
        (self.system, self.dt)
    }
}

impl<S> Stepper for RungeKutta4<S,f64>
    where S: ODE<State=f64> + 'static
{
    type System = S;
    type State = f64;

    fn do_step(&mut self) {
        self.initial = self.system.get_state().clone();
        self.system.differentiate_into(&self.initial, &mut self.k1);
        self.system.differentiate_into(&(self.initial + &self.k1 * self.dt_2), &mut self.k2);
        self.system.differentiate_into(&(self.initial + &self.k2 * self.dt_2), &mut self.k3);
        self.system.differentiate_into(&(self.initial + &self.k3 * self.dt), &mut self.k4);
        self.temp = self.initial + &(self.dt_6 * &self.k1) + &(self.dt_3 * &self.k2) +
                    &(self.dt_3 * &self.k3) + &(self.dt_6 * &self.k4);
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

impl<S> Stepper for RungeKutta4<S, Vec<f64>>
    where S: ODE<State=Vec<f64>> + 'static
{
    type System = S;
    type State = Vec<f64>;

    fn do_step(&mut self) {
        self.initial.copy_from_slice(self.system.get_state());

        self.system.differentiate_into(&mut self.initial, &mut self.k1);

        for (t, i, k) in izip!(self.temp.iter_mut(), self.initial.iter(), self.k1.iter()) {
            *t = *i + self.dt_2 * k;
        }
        self.system.differentiate_into(&mut self.temp, &mut self.k2);

        for (t, i, k) in izip!(self.temp.iter_mut(), self.initial.iter(), self.k2.iter()) {
            *t = *i + self.dt_2 * k;
        }
        self.system.differentiate_into(&mut self.temp, &mut self.k3);

        for (t, i, k) in izip!(self.temp.iter_mut(), self.initial.iter(), self.k3.iter()) {
            *t = *i + self.dt * k;
        }
        self.system.differentiate_into(&mut self.temp, &mut self.k4);

        for (t, i, k1, k2, k3, k4) in izip!(self.temp.iter_mut(),
            self.initial.iter(),
            self.k1.iter(),
            self.k2.iter(),
            self.k3.iter(),
            self.k4.iter())
        {
            *t = i + self.dt_6 * k1 + self.dt_3 * k2 + self.dt_3 * k3 + self.dt_6 * k4;
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

impl<S,T,D> Stepper for RungeKutta4<T, ArrayBase<S, D>>
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
        let dt_3 = self.dt_3;
        let dt_6 = self.dt_6;

        zip_mut_with_2(&mut self.temp, self.initial.view(), self.k1.view(), |t,i,k| {
            *t = i + dt_2 * k;
        }).unwrap();
        self.system.differentiate_into(&self.temp, &mut self.k2);

        zip_mut_with_2(&mut self.temp, self.initial.view(), self.k2.view(), |t,i,k| {
            *t = i + dt_2 * k;
        }).unwrap();
        self.system.differentiate_into(&self.temp, &mut self.k3);

        zip_mut_with_2(&mut self.temp, self.initial.view(), self.k3.view(), |t,i,k| {
            *t = i + dt * k;
        }).unwrap();
        self.system.differentiate_into(&self.temp, &mut self.k4);

        zip_mut_with_5(&mut self.temp, self.initial.view(), self.k1.view(), self.k2.view(), self.k3.view(), self.k4.view(),
            |t,i,k1,k2,k3,k4| {
                *t = i + dt_6 * k1 + dt_3 * k2 + dt_3 * k3 + dt_6 * k4;
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
