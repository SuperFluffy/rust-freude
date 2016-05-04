use traits::{Observer, ODE};
use steppers::Stepper;

pub struct Integrator<T> {
    stepper: Box<Stepper<State = T>>,
}

impl<T> Integrator<T> {
    pub fn new(stepper: Box<Stepper<State = T>>) -> Self {
        Integrator { stepper: stepper }
    }

    pub fn get_state(&self) -> &T {
        self.stepper.get_state()
    }

    pub fn get_system(&self) -> &ODE<State=T> {
        self.stepper.get_system()
    }

    pub fn integrate_time(&mut self,
                             t: f64,
                             obs: &mut Observer<ODE<State = T>>)
                             -> (f64, usize)
    {
        let mut tacc = 0f64;;
        let mut count = 0;

        let dt = self.stepper.timestep();

        // tacc+dt ensures that we don't exceed tf, and that tf - dt < t' <= tf
        while (tacc + dt) <= t {
            self.stepper.do_step();
            tacc += dt;
            count += 1;
            obs.observe(self.stepper.get_system_mut(),1);
        }

        (tacc, count)
    }

    pub fn integrate_time_range<I: IntoIterator<Item=f64>>(&mut self,
                             ts: I,
                             innerobs: &mut Observer<ODE<State = T>>,
                             outerobs: &mut Observer<ODE<State = T>>)
                             -> (f64, usize)
    {
        let mut count = 0;
        let mut tacc = 0f64;

        for t in ts {
        // tacc+dt ensures that we don't exceed tf, and that tf - dt < t' <= tf
            let (totacc,tocount) = self.integrate_time(t, innerobs);
            tacc += totacc;
            count += tocount;
            outerobs.observe(self.stepper.get_system_mut(), tocount);
        }
        (tacc, count)
    }

    pub fn integrate_n_steps(&mut self, n: usize, obs: &mut Observer<ODE<State = T>>) {
        for _i in 0..n {
            self.stepper.do_step();

            obs.observe(self.stepper.get_system_mut(),1);
        }
    }

    pub fn integrate_n_range<I: IntoIterator<Item=usize>>(&mut self,
                             ns: I,
                             innerobs: &mut Observer<ODE<State = T>>,
                             outerobs: &mut Observer<ODE<State = T>>)
                             -> usize
    {
        let mut count = 0;
        for n in ns.into_iter() {
            self.integrate_n_steps(n, innerobs);
            count += n;
            outerobs.observe(self.stepper.get_system_mut(),n);
        }
        count
    }

    pub fn warmup_n_steps(&mut self, n: usize) {
        for _i in 0..n {
            self.stepper.do_step();
        }
    }

    pub fn warmup_time(&mut self, t: f64) -> (f64, usize) {
        let mut tacc = 0f64;;
        let mut count = 0;

        let dt = self.stepper.timestep();

        // tacc+dt ensures that we don't exceed tf, and that tf - dt < t' <= tf
        while (tacc + dt) <= t {
            self.stepper.do_step();
            tacc += dt;
            count += 1;
        }

        (tacc, count)
    }
}
