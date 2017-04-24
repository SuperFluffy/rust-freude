use tuple::*;

// use super::ODE;
use super::Euler;
use super::Heun;
use super::RungeKutta4;
use super::Stepper;

use ode::*;

macro_rules! impl_stepper_for_tuples {
    ( $tuple:ty ) => {
        impl<S> Stepper for Euler<S,$tuple>
            where S: ODE<State=$tuple> + 'static
        {
            type System = S;
            type State = $tuple;

            fn do_step(&mut self, state: &mut Self::State) {
                self.system.differentiate_into(state, &mut self.temp);
                self.temp = *state + self.temp * <$tuple as Splat<_>>::splat(self.dt);
                self.system.update_state(state, &self.temp);
            }

            fn system_ref(&self) -> &Self::System {
                &self.system
            }

            fn system_mut(&mut self) -> &mut Self::System {
                &mut self.system
            }

            fn timestep(&self) -> f64 {
                self.dt
            }
        }

        impl<S> Stepper for Heun<S,$tuple>
            where S: ODE<State=$tuple> + 'static
        {
            type System = S;
            type State = $tuple;

            fn do_step(&mut self, state: &mut Self::State) {
               let dt = <$tuple as Splat<_>>::splat(self.dt);
               let dt_2 = <$tuple as Splat<_>>::splat(self.dt_2);

               self.system.differentiate_into(state, &mut self.k1);
               self.system.differentiate_into(&(*state + dt * self.k1), &mut self.k2);
               self.temp = *state + dt_2 * (self.k1 + self.k2);
               self.system.update_state(state, &self.temp);
            }

            fn system_ref(&self) -> &Self::System {
                &self.system
            }

            fn system_mut(&mut self) -> &mut Self::System {
                &mut self.system
            }

            fn timestep(&self) -> f64 {
                self.dt
            }
        }

        impl<S> Stepper for RungeKutta4<S,$tuple>
            where S: ODE<State=$tuple> + 'static
        {
            type System = S;
            type State = $tuple;

            fn do_step(&mut self, state: &mut Self::State) {
                let dt = <$tuple as Splat<_>>::splat(self.dt);
                let dt_2 = <$tuple as Splat<_>>::splat(self.dt_2);
                let dt_3 = <$tuple as Splat<_>>::splat(self.dt_3);
                let dt_6 = <$tuple as Splat<_>>::splat(self.dt_6);

                self.system.differentiate_into(state, &mut self.k1);
                self.system.differentiate_into(&(*state + dt_2 * self.k1), &mut self.k2);
                self.system.differentiate_into(&(*state + dt_2 * self.k2), &mut self.k3);
                self.system.differentiate_into(&(*state + dt * self.k3), &mut self.k4);

                self.temp = *state + dt_6 * (self.k1 + self.k4)+ dt_3 * (self.k2 + self.k3);
                self.system.update_state(state, &self.temp);
            }

            fn system_ref(&self) -> &Self::System {
                &self.system
            }

            fn system_mut(&mut self) -> &mut Self::System {
                &mut self.system
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
