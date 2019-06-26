use crate::Ode;
use crate::{Euler, Heun, RungeKutta4, Stepper};

use tuple::{Splat, A1, A10, A11, A12, A2, A3, A4, A5, A6, A7, A8, A9};

macro_rules! impl_ode_for_tuples {
    ( $tup:ty ) => {
        impl Ode for Box<Fn($tup) -> $tup>
        {
            type State = $tup;

            fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
                derivative.clone_from(&self(*state));
            }
        }

        impl<'a> Ode for &'a Fn($tup) -> $tup
        {
            type State = $tup;

            fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
                derivative.clone_from(&self(*state));
            }
        }

        impl Ode for Box<FnMut($tup) -> $tup>
        {
            type State = $tup;

            fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
                derivative.clone_from(&self(*state));
            }
        }

        impl<'a> Ode for &'a mut FnMut($tup) -> $tup
        {
            type State = $tup;

            fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
                derivative.clone_from(&self(*state));
            }
        }
    };

    ( $( $tup:ty ),+ ) => {
        $(
            impl_ode_for_tuples!($tup);
        )+
    };
}

macro_rules! impl_stepper_for_tuples {
    ( $tuple:ty ) => {
        impl Stepper for Euler<$tuple>
        {
            type State = $tuple;

            fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
                where Sy: Ode<State = Self::State>,
            {
                system.differentiate_into(state, &mut self.temp);
                self.temp = *state + self.temp * <$tuple as Splat<_>>::splat(self.dt);
                system.update_state(state, &self.temp);
            }

            fn timestep(&self) -> f64 {
                self.dt
            }
        }

        impl Stepper for Heun<$tuple>
        {
            type State = $tuple;

            fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
                where Sy: Ode<State = Self::State>,
            {
               let dt = <$tuple as Splat<_>>::splat(self.dt);
               let dt_2 = <$tuple as Splat<_>>::splat(self.dt_2);

               system.differentiate_into(state, &mut self.k1);
               system.differentiate_into(&(*state + dt * self.k1), &mut self.k2);
               self.temp = *state + dt_2 * (self.k1 + self.k2);
               system.update_state(state, &self.temp);
            }

            fn timestep(&self) -> f64 {
                self.dt
            }
        }

        impl Stepper for RungeKutta4<$tuple>
        {
            type State = $tuple;

            fn do_step<Sy>(&mut self, system: &mut Sy, state: &mut Self::State)
                where Sy: Ode<State = Self::State>,
            {
                let dt = <$tuple as Splat<_>>::splat(self.dt);
                let dt_2 = <$tuple as Splat<_>>::splat(self.dt_2);
                let dt_3 = <$tuple as Splat<_>>::splat(self.dt_3);
                let dt_6 = <$tuple as Splat<_>>::splat(self.dt_6);

                system.differentiate_into(state, &mut self.k1);
                system.differentiate_into(&(*state + dt_2 * self.k1), &mut self.k2);
                system.differentiate_into(&(*state + dt_2 * self.k2), &mut self.k3);
                system.differentiate_into(&(*state + dt * self.k3), &mut self.k4);

                self.temp = *state + dt_6 * (self.k1 + self.k4)+ dt_3 * (self.k2 + self.k3);
                system.update_state(state, &self.temp);
            }

            fn timestep(&self) -> f64 {
                self.dt
            }
        }
    };
    ( $( $tuple:ty ),+ ) => {
        $(
            impl_stepper_for_tuples!($tuple);
        )+
    };
}

impl_ode_for_tuples!(
    A1<f64>,
    A2<f64>,
    A3<f64>,
    A4<f64>,
    A5<f64>,
    A6<f64>,
    A7<f64>,
    A8<f64>,
    A9<f64>,
    A10<f64>,
    A11<f64>,
    A12<f64>
);

impl_stepper_for_tuples!(
    A1<f64>,
    A2<f64>,
    A3<f64>,
    A4<f64>,
    A5<f64>,
    A6<f64>,
    A7<f64>,
    A8<f64>,
    A9<f64>,
    A10<f64>,
    A11<f64>,
    A12<f64>
);
