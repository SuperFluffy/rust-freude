use traits::{Observer,ODE};
use steppers::Stepper;

pub struct Integrator<T> {
    stepper: Box<Stepper<State=T>>,
}

impl<T> Integrator<T> {
    pub fn new(stepper: Box<Stepper<State=T>>) -> Self {
        Integrator {
            stepper: stepper,
        }
    }
    pub fn integrate_between(&mut self, t0: f64, tf: f64, dt: f64, obs: &mut Observer<ODE<State=T>>) -> (f64,u64) {
        let mut tacc = t0;
        let mut count = 0;

        // tacc+dt ensures that we don't exceed tf, and that tf - dt < t' <= tf
        while (tacc+dt) <= tf {
            self.stepper.do_step(dt);
            tacc += dt;
            count += 1;

            obs.observe(self.stepper.get_system_mut());
        }

        (tacc, count)
    }

    pub fn integrate_n_times(&mut self, n: u64, dt: f64, obs: &mut Observer<ODE<State=T>>) {
        for _i in 0..n {
            self.stepper.do_step(dt);
        }

        obs.observe(self.stepper.get_system_mut());
    }

    pub fn get_state(&self) -> &T {
        self.stepper.get_state()
    }
}
