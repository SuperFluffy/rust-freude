use traits::{Observer, ODE};
use steppers::Stepper;

pub struct Integrator<T> {
    stepper: Box<Stepper<State = T>>,
}

impl<T> Integrator<T> {
    pub fn new(stepper: Box<Stepper<State = T>>) -> Self {
        Integrator { stepper: stepper }
    }
    pub fn integrate_time(&mut self,
                             t: f64,
                             dt: f64,
                             obs: &mut Observer<ODE<State = T>>)
                             -> (f64, u64) {
        let mut tacc = 0f64;;
        let mut count = 0;

        // tacc+dt ensures that we don't exceed tf, and that tf - dt < t' <= tf
        while (tacc + dt) <= t {
            self.stepper.do_step(dt);
            tacc += dt;
            count += 1;
            obs.observe(self.stepper.get_system_mut());
        }

        (tacc, count)
    }

    pub fn integrate_time_range<I: IntoIterator<Item=f64>>(&mut self,
                             ts: I,
                             dt: f64,
                             innerobs: &mut Observer<ODE<State = T>>,
                             outerobs: &mut Observer<ODE<State = T>>)
                             -> (f64, u64) {
        let mut count = 0;
        let mut tacc = 0f64;

        for t in ts {
        // tacc+dt ensures that we don't exceed tf, and that tf - dt < t' <= tf
            let (totacc,tocount) = self.integrate_time(t, dt, innerobs);
            tacc += totacc;
            count += tocount;
            outerobs.observe(self.stepper.get_system_mut());
        }
        (tacc, count)
    }

    pub fn integrate_n_steps(&mut self, n: u64, dt: f64, obs: &mut Observer<ODE<State = T>>) {
        for _i in 0..n {
            self.stepper.do_step(dt);

            obs.observe(self.stepper.get_system_mut());
        }
    }

    pub fn get_state(&self) -> &T {
        self.stepper.get_state()
    }

    pub fn integrate_n_range<I: IntoIterator<Item=u64>>(&mut self,
                             ns: I,
                             dt: f64,
                             innerobs: &mut Observer<ODE<State = T>>,
                             outerobs: &mut Observer<ODE<State = T>>)
                             -> u64 {
        let mut count = 0;
        for n in ns.into_iter() {
            self.integrate_n_steps(n, dt, innerobs);
            count += n;
            outerobs.observe(self.stepper.get_system_mut());
        }
        count
    }
}
