use ndarray::{ArrayBase,DataMut,DataClone,Dimension};

use traits::ODE;

use super::Stepper;

pub struct RungeKutta4<S, T> {
    dt: f64,
    dt_2: f64,
    dt_3: f64,
    dt_6: f64,

    system: S,
    temp: T,

    k1: T,
    k2: T,
    k3: T,
    k4: T,
}

impl<S,T> RungeKutta4<S,T>
    where S: ODE<State=T>,
          T: Clone,
{
    pub fn new(mut system: S, dt: f64, state: &T) -> Self {
        let dt_2 = dt/2.0;
        let dt_3 = dt/3.0;
        let dt_6 = dt/6.0;

        let temp = system.differentiate(&state);
        let k1 = system.differentiate(&state);
        let k2 = system.differentiate(&state);
        let k3 = system.differentiate(&state);
        let k4 = system.differentiate(&state);

        RungeKutta4 {
            dt: dt,
            dt_2: dt_2,
            dt_3: dt_3,
            dt_6: dt_6,
            system: system,

            temp: temp,
            k1: k1,
            k2: k2,
            k3: k3,
            k4: k4,
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

impl<S> Stepper for RungeKutta4<S,f64>
    where S: ODE<State=f64> + 'static
{
    type System = S;
    type State = f64;

    fn do_step(&mut self, state: &mut Self::State) {
        self.system.differentiate_into(state, &mut self.k1);
        self.system.differentiate_into(&(*state + &self.k1 * self.dt_2), &mut self.k2);
        self.system.differentiate_into(&(*state + &self.k2 * self.dt_2), &mut self.k3);
        self.system.differentiate_into(&(*state + &self.k3 * self.dt), &mut self.k4);
        self.temp = *state + &(self.dt_6 * &self.k1) + &(self.dt_3 * &self.k2) +
            &(self.dt_3 * &self.k3) + &(self.dt_6 * &self.k4);
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

impl<S> Stepper for RungeKutta4<S, Vec<f64>>
    where S: ODE<State=Vec<f64>> + 'static
{
    type System = S;
    type State = Vec<f64>;

    fn do_step(&mut self, state: &mut Self::State) {
        let dt = self.dt;
        let dt_2 = self.dt_2;
        let dt_3 = self.dt_3;
        let dt_6 = self.dt_6;

        self.system.differentiate_into(state, &mut self.k1);

        azip!(mut t (&mut self.temp), s (&*state), k1 (&self.k1) in {
            *t = s + dt_2 * k1
        });
        self.system.differentiate_into(&self.temp, &mut self.k2);

        azip!(mut t (&mut self.temp), s (&*state), k2 (&self.k2) in {
            *t = s + dt_2 * k2
        });
        self.system.differentiate_into(&self.temp, &mut self.k3);

        azip!(mut t (&mut self.temp), s (&*state), k3 (&self.k3) in {
            *t = s + dt * k3
        });
        self.system.differentiate_into(&self.temp, &mut self.k4);

        azip!(mut t (&mut self.temp), s (&*state), k1 (&self.k1), k2 (&self.k2), k3 (&self.k3), k4 (&self.k4) in {
            *t = s + dt_6 * k1 + dt_3 * k2 + dt_3 * k3 + dt_6 * k4;
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

impl<S,T,D> Stepper for RungeKutta4<T, ArrayBase<S, D>>
    where D: Dimension,
          S: DataMut<Elem=f64> + DataClone,
          T: ODE<State=ArrayBase<S, D>> + 'static,
{
    type System = T;
    type State = ArrayBase<S,D>;

    fn do_step(&mut self, state: &mut Self::State) {
        self.system.differentiate_into(state, &mut self.k1);

        // Need to assign the values here, because closures try to immutably borrow the entire
        // self, which fails because self.temp is borrowed mutably.
        let dt = self.dt;
        let dt_2 = self.dt_2;
        let dt_3 = self.dt_3;
        let dt_6 = self.dt_6;

        azip!(mut t (&mut self.temp), s (&*state), k1 (&self.k1) in {
            *t = s + dt_2 * k1
        });

        self.system.differentiate_into(&self.temp, &mut self.k2);

        azip!(mut t (&mut self.temp), s (&*state), k2 (&self.k2) in {
            *t = s + dt_2 * k2
        });

        self.system.differentiate_into(&self.temp, &mut self.k3);

        azip!(mut t (&mut self.temp), s (&*state), k3 (&self.k3) in {
            *t = s + dt * k3
        });

        self.system.differentiate_into(&self.temp, &mut self.k4);

        azip!(mut t (&mut self.temp), s (&*state), k1 (&self.k1), k2 (&self.k2), k3 (&self.k3), k4 (&self.k4) in {
            *t = s + dt_6 * k1 + dt_3 * k2 + dt_3 * k3 + dt_6 * k4
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
