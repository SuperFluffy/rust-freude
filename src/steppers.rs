use std::ops::{Add,Mul};

use ndarray::{Dimension, Scalar};
use ndarray::OwnedArray;

use traits::ODE;

pub trait Stepper {
    fn do_step(&mut self);
}

pub struct RungeKutta4<T> {
    pub system: Box<ODE<State=T>>,
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

impl<T> RungeKutta4<T>
    where
        T: Clone,
{
    pub fn new(system: Box<ODE<State=T>>, dt: f64) -> Self {
        let temp = system.get_state().clone();
        let k1 = system.get_state().clone();
        let k2 = system.get_state().clone();
        let k3 = system.get_state().clone();
        let k4 = system.get_state().clone();

        let dt_2 = dt / 2.;
        let dt_3 = dt / 3.;
        let dt_6 = dt / 6.;

        RungeKutta4 {
            system: system,
            temp: temp,
            k1: k1,
            k2: k2,
            k3: k3,
            k4: k4,

            dt: dt,
            dt_2: dt_2,
            dt_3: dt_3,
            dt_6: dt_6,
        }
    }
}

// impl<T> Stepper for RungeKutta4<T>
//     where
//         for<'a, 'b> T: 'a + Clone + Add<&'b T, Output=T>,
//         for<'a, 'b> &'a T: Add<T, Output=T> + Add<&'b T, Output=T> + Mul<f64, Output=T>,
//         for<'b> f64: Mul<&'b T, Output=T>,
// {
//     fn do_step (&mut self) {
//         {
//             let initial_state = self.system.get_state();
//             self.system.differentiate_into(initial_state, &mut self.k1);
//             self.system.differentiate_into(&(initial_state + &self.k1 * self.dt_2), &mut self.k2);
//             self.system.differentiate_into(&(initial_state + &self.k2 * self.dt_2), &mut self.k3);
//             self.system.differentiate_into(&(initial_state + &self.k3 * self.dt), &mut self.k4);
//             self.temp = initial_state + &(self.dt_6 * &self.k1) + &(self.dt_3 * &self.k2) + &(self.dt_3 * &self.k3) + &(self.dt_6 * &self.k4);
//         }
//         self.system.update_state(&self.temp);
//     }
// }

impl Stepper for RungeKutta4<f64>
{
    fn do_step (&mut self) {
        {
            let initial_state = self.system.get_state();
            self.system.differentiate_into(initial_state, &mut self.k1);
            self.system.differentiate_into(&(initial_state + &self.k1 * self.dt_2), &mut self.k2);
            self.system.differentiate_into(&(initial_state + &self.k2 * self.dt_2), &mut self.k3);
            self.system.differentiate_into(&(initial_state + &self.k3 * self.dt), &mut self.k4);
            self.temp = initial_state + &(self.dt_6 * &self.k1) + &(self.dt_3 * &self.k2) + &(self.dt_3 * &self.k3) + &(self.dt_6 * &self.k4);
        }
        self.system.update_state(&self.temp);
    }
}

impl Stepper for RungeKutta4<Vec<f64>>
{
    fn do_step (&mut self) {
        {
            let initial_state = self.system.get_state();
            self.system.differentiate_into(initial_state, &mut self.k1);

            for (t,i,k) in izip!(self.temp.iter_mut(), initial_state.iter(), self.k1.iter()) {
                *t = *i + self.dt_2 * k;
            }
            self.system.differentiate_into(&self.temp, &mut self.k2);

            for (t,i,k) in izip!(self.temp.iter_mut(), initial_state.iter(), self.k2.iter()) {
                *t = *i + self.dt_2 * k;
            }
            self.system.differentiate_into(&self.temp, &mut self.k3);

            for (t,i,k) in izip!(self.temp.iter_mut(), initial_state.iter(), self.k3.iter()) {
                *t = *i + self.dt * k;
            }
            self.system.differentiate_into(&self.temp, &mut self.k4);

            for (t,i,k1,k2,k3,k4) in izip!(self.temp.iter_mut(), initial_state.iter(), self.k1.iter(), self.k2.iter(), self.k3.iter(), self.k4.iter()) {
                *t = i + self.dt_6 * k1 + self.dt_3 * k2 + self.dt_3 * k3 + self.dt_6 * k4;
            }
        }
        self.system.update_state(&self.temp);
    }
}

impl<D> RungeKutta4<OwnedArray<f64,D>>
    where D: Dimension
{
    pub fn new(system: Box<ODE<State=OwnedArray<f64,D>>>, dt: f64) -> Self {

        let temp = system.get_state().clone();
        let k1 = system.get_state().clone();
        let k2 = system.get_state().clone();
        let k3 = system.get_state().clone();
        let k4 = system.get_state().clone();

        let dt_2 = dt / 2.;
        let dt_3 = dt / 3.;
        let dt_6 = dt / 6.;

        RungeKutta4 {
            system: system,
            temp: temp,
            k1: k1,
            k2: k2,
            k3: k3,
            k4: k4,

            dt: dt,
            dt_2: dt_2,
            dt_3: dt_3,
            dt_6: dt_6,
        }
    }
}

impl<D> Stepper for RungeKutta4<OwnedArray<f64,D>>
    where D: Dimension
{
    fn do_step (&mut self) {
        {
            let initial_state = self.system.get_state();

            self.system.differentiate_into(initial_state, &mut self.k1);

            self.temp.assign(&self.k1);
            self.temp *= self.dt_2;
            self.temp += initial_state;
            self.system.differentiate_into(&self.temp, &mut self.k2);

            self.temp.assign(&self.k2);
            self.temp *= self.dt_2;
            self.temp += initial_state;
            self.system.differentiate_into(&self.temp, &mut self.k3);

            self.temp.assign(&self.k3);
            self.temp *= self.dt;
            self.temp += initial_state;
            self.system.differentiate_into(&self.temp, &mut self.k4);

            self.temp.assign(initial_state);

            self.k1 *= self.dt_6;
            self.k2 *= self.dt_3;
            self.k3 *= self.dt_3;
            self.k4 *= self.dt_6;

            self.temp += &self.k1;
            self.temp += &self.k2;
            self.temp += &self.k3;
            self.temp += &self.k4;
        }

        self.system.update_state(&self.temp);
    }
}
