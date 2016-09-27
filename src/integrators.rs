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

    pub fn get_system_mut(&mut self) -> &mut (ODE<State=T> + 'static) {
        self.stepper.get_system_mut()
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
            obs.observe(self.stepper.get_system_mut(),dt);
        }

        (tacc, count)
    }

    pub fn integrate_time_range<I: IntoIterator<Item=f64>>(&mut self,
                             ts: I,
                             obs: &mut Observer<ODE<State = T>>)
                             -> (f64, usize)
    {
        let mut count = 0;
        let mut tacc = 0f64;

        for t in ts {
        // tacc+dt ensures that we don't exceed tf, and that tf - dt < t' <= tf
            let (totacc,tocount) = self.integrate_time(t, obs);
            tacc += totacc;
            count += tocount;
        }
        (tacc, count)
    }

    pub fn integrate_n_steps(&mut self, n: usize, obs: &mut Observer<ODE<State = T>>) -> (f64,usize)
    {
        let dt = self.stepper.timestep();
        let mut tacc = 0f64;
        for _i in 0..n {
            self.stepper.do_step();
            obs.observe(self.stepper.get_system_mut(), dt);
            tacc += dt;
        }
        (tacc, n)
    }

    pub fn integrate_n_range<I: IntoIterator<Item=usize>>(&mut self,
                             ns: I,
                             obs: &mut Observer<ODE<State = T>>)
                             -> (f64, usize)
    {
        let mut count = 0;
        let mut tacc = 0f64;
        for n in ns.into_iter() {
            let (totacc,tocount) = self.integrate_n_steps(n, obs);
            tacc += totacc;
            count += tocount; // Same as n
        }
        (tacc, count)
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
