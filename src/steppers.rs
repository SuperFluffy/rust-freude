// use std::ops::{Add, Mul};

// use ndarray::{Dimension, Scalar};
use ndarray::Dimension;
use ndarray::OwnedArray;

use traits::ODE;
use utils::{zip_mut_with_2,zip_mut_with_5};

pub trait Stepper {
    type State;

    fn do_step(&mut self);

    fn timestep(&self) -> f64;

    fn get_state(&self) -> &Self::State;

    fn get_system(&self) -> &ODE<State = Self::State>;
    fn get_system_mut(&mut self) -> &mut (ODE<State = Self::State> + 'static);
}

pub struct RungeKutta4<T> {
    system: Box<ODE<State = T>>,

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

// impl<T> RungeKutta4<T>
//     where
//         T: Clone,
// {
//     pub fn new(system: Box<ODE<State=T>>) -> Self {
//         let temp = system.get_state().clone();
//         let k1 = system.get_state().clone();
//         let k2 = system.get_state().clone();
//         let k3 = system.get_state().clone();
//         let k4 = system.get_state().clone();

//         RungeKutta4 {
//             system: system,
//             temp: temp,
//             k1: k1,
//             k2: k2,
//             k3: k3,
//             k4: k4,
//         }
//     }
// }

// impl<T> Stepper for RungeKutta4<T>
//     where
//         for<'a, 'b> T: 'a + Clone + Add<&'b T, Output=T>,
//         for<'a, 'b> &'a T: Add<T, Output=T> + Add<&'b T, Output=T> + Mul<f64, Output=T>,
//         for<'b> f64: Mul<&'b T, Output=T>,
// {
//     fn do_step (&mut self) {
//         let dt_2 = dt / 2.;
//         let dt_3 = dt / 3.;
//         let dt_6 = dt / 6.;
//         {
//             let initial_state = self.system.get_state();
//             self.system.differentiate_into(initial_state, &mut self.k1);
//             self.system.differentiate_into(&(initial_state + &self.k1 * dt_2), &mut self.k2);
//             self.system.differentiate_into(&(initial_state + &self.k2 * dt_2), &mut self.k3);
//             self.system.differentiate_into(&(initial_state + &self.k3 * dt), &mut self.k4);
//             self.temp = initial_state + &(dt_6 * &self.k1) + &(dt_3 * &self.k2) + &(dt_3 * &self.k3) + &(dt_6 * &self.k4);
//         }
//         self.system.update_state(&self.temp);
//     }
// }

impl RungeKutta4<f64> {
    pub fn new(system: Box<ODE<State = f64>>, dt: f64) -> Self {
        let temp = system.get_state().clone();
        let k1 = system.get_state().clone();
        let k2 = system.get_state().clone();
        let k3 = system.get_state().clone();
        let k4 = system.get_state().clone();

        RungeKutta4 {
            system: system,
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
}

impl RungeKutta4<Vec<f64>> {
    pub fn new(system: Box<ODE<State = Vec<f64>>>, dt: f64) -> Self {
        let temp = system.get_state().clone();
        let k1 = system.get_state().clone();
        let k2 = system.get_state().clone();
        let k3 = system.get_state().clone();
        let k4 = system.get_state().clone();

        RungeKutta4 {
            system: system,
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
}

impl Stepper for RungeKutta4<f64> {
    type State = f64;

    fn do_step(&mut self) {
        {
            let mut initial_state = self.system.get_state().clone();
            self.system.differentiate_into(&mut initial_state, &mut self.k1);
            self.system.differentiate_into(&mut (initial_state + &self.k1 * self.dt_2), &mut self.k2);
            self.system.differentiate_into(&mut (initial_state + &self.k2 * self.dt_2), &mut self.k3);
            self.system.differentiate_into(&mut (initial_state + &self.k3 * self.dt), &mut self.k4);
            self.temp = initial_state + &(self.dt_6 * &self.k1) + &(self.dt_3 * &self.k2) +
                        &(self.dt_3 * &self.k3) + &(self.dt_6 * &self.k4);
        }
        self.system.update_state(&self.temp);
    }

    fn get_state(&self) -> &Self::State {
        self.system.get_state()
    }

    fn get_system(&self) -> &ODE<State = Self::State> {
        &*self.system
    }

    fn get_system_mut(&mut self) -> &mut (ODE<State = Self::State> + 'static) {
        &mut *self.system
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}

impl Stepper for RungeKutta4<Vec<f64>> {
    type State = Vec<f64>;

    fn do_step(&mut self) {
        {
            let mut initial_state = self.system.get_state().clone();
            self.system.differentiate_into(&mut initial_state, &mut self.k1);

            for (t, i, k) in izip!(self.temp.iter_mut(), initial_state.iter(), self.k1.iter()) {
                *t = *i + self.dt_2 * k;
            }
            self.system.differentiate_into(&mut self.temp, &mut self.k2);

            for (t, i, k) in izip!(self.temp.iter_mut(), initial_state.iter(), self.k2.iter()) {
                *t = *i + self.dt_2 * k;
            }
            self.system.differentiate_into(&mut self.temp, &mut self.k3);

            for (t, i, k) in izip!(self.temp.iter_mut(), initial_state.iter(), self.k3.iter()) {
                *t = *i + self.dt * k;
            }
            self.system.differentiate_into(&mut self.temp, &mut self.k4);

            for (t, i, k1, k2, k3, k4) in izip!(self.temp.iter_mut(),
                                                initial_state.iter(),
                                                self.k1.iter(),
                                                self.k2.iter(),
                                                self.k3.iter(),
                                                self.k4.iter()) {
                *t = i + self.dt_6 * k1 + self.dt_3 * k2 + self.dt_3 * k3 + self.dt_6 * k4;
            }
        }
        self.system.update_state(&self.temp);
    }

    fn get_state(&self) -> &Self::State {
        self.system.get_state()
    }

    fn get_system(&self) -> &ODE<State = Self::State> {
        &*self.system
    }

    fn get_system_mut(&mut self) -> &mut (ODE<State = Self::State> + 'static) {
        &mut *self.system
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}

impl<D> RungeKutta4<OwnedArray<f64, D>>
    where D: Dimension
{
    pub fn new(system: Box<ODE<State = OwnedArray<f64, D>>>, dt: f64) -> Self {

        let temp = system.get_state().clone();
        let k1 = system.get_state().clone();
        let k2 = system.get_state().clone();
        let k3 = system.get_state().clone();
        let k4 = system.get_state().clone();

        RungeKutta4 {
            system: system,
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
}

impl<D> Stepper for RungeKutta4<OwnedArray<f64, D>>
    where D: Dimension
{
    type State = OwnedArray<f64,D>;

    fn do_step(&mut self) {
        {
            let mut initial_state = self.system.get_state().clone();

            self.system.differentiate_into(&mut initial_state, &mut self.k1);

            // Need to assign the values here, because closures try to immutably borrow the entire
            // self, which fails because self.temp is borrowed mutably.
            let dt = self.dt;
            let dt_2 = self.dt_2;
            let dt_3 = self.dt_3;
            let dt_6 = self.dt_6;

            zip_mut_with_2(&mut self.temp, initial_state.view(), self.k1.view(), |t,i,k| {
                *t = i + dt_2 * k;
            }).unwrap();
            self.system.differentiate_into(&mut self.temp, &mut self.k2);

            zip_mut_with_2(&mut self.temp, initial_state.view(), self.k2.view(), |t,i,k| {
                *t = i + dt_2 * k;
            }).unwrap();
            self.system.differentiate_into(&mut self.temp, &mut self.k3);

            zip_mut_with_2(&mut self.temp, initial_state.view(), self.k3.view(), |t,i,k| {
                *t = i + dt * k;
            }).unwrap();
            self.system.differentiate_into(&mut self.temp, &mut self.k4);

            zip_mut_with_5(&mut self.temp, initial_state.view(), self.k1.view(), self.k2.view(), self.k3.view(), self.k4.view(),
                |t,i,k1,k2,k3,k4| {
                    *t = i + dt_6 * k1 + dt_3 * k2 + dt_3 * k3 + dt_6 * k4;
            }).unwrap();
        }

        self.system.update_state(&self.temp);
    }

    fn get_state(&self) -> &Self::State {
        self.system.get_state()
    }

    fn get_system(&self) -> &ODE<State = Self::State> {
        &*self.system
    }

    fn get_system_mut(&mut self) -> &mut (ODE<State = Self::State> + 'static) {
        &mut *self.system
    }

    fn timestep(&self) -> f64 {
        self.dt
    }
}
