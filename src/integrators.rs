use traits::{Observer,ODE};
use steppers::Stepper;

struct Integrator<T> {
    stepper: Box<Stepper<State=T>>,
}

impl<T> Integrator<T> {
    pub fn integrate_between(&mut self, t0: f64, tf: f64, dt: f64, obs: &mut Observer<Box<ODE<State=T>>>) -> (f64,u64) {
        let mut tacc = t0;
        let mut count = 0;

        while tacc <= tf {
            self.stepper.do_step(dt);
            tacc += dt;
            count += 1;

            obs.observe(self.stepper.give_system());
        }

        (tacc, count)
    }

    pub fn integrate_n_times(&mut self, n: u64, dt: f64, obs: &mut Observer<Box<ODE<State=T>>>) {
        for _i in 0..n {
            self.stepper.do_step(dt);
        }

        obs.observe(self.stepper.give_system());
    }
}
