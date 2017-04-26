use tuple::*;

use super::Euler;
use super::Heun;
use super::RungeKutta4;
use super::Stepper;

use ode::*;

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

impl_stepper_for_tuples!(
     UT1<f64>,
     UT2<f64>,
     UT3<f64>,
     UT4<f64>,
     UT5<f64>,
     UT6<f64>,
     UT7<f64>,
     UT8<f64>,
     UT9<f64>,
    UT10<f64>,
    UT11<f64>,
    UT12<f64>
);
