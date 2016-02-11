use traits::{Observer,ODE};
use steppers::Stepper;

struct Integrator<T> {
    stepper: Box<Stepper<State=T>>,
}

impl<T> Integrator<T> {
    // If we define the argument obs as `&mut Observer<ODE<State=T> + 'a>`, where the lifetime 'a
    // is coupled to &' mut self, how we can make sure that the calls to do_step() and observe()
    // will stop borrowing after they are over.
    //
    // This also affects the return type of get_system_mut(), which of returning a mutable borrow
    // to a trait object, &mut ODE<State=T>, now has to return a &mut Box<ODE<State=T>>. See the
    // respective stepper definitions.
    //
    // pub fn integrate_between<'a>(&'a mut self, t0: f64, tf: f64, dt: f64, obs: &mut Observer<ODE<State=T> + 'a>) -> (f64,u64) {
    pub fn integrate_between(&mut self, t0: f64, tf: f64, dt: f64, obs: &mut Observer<Box<ODE<State=T>>>) -> (f64,u64) {
        let mut tacc = t0;
        let mut count = 0;

        while tacc <= tf {
            self.stepper.do_step(dt);
            tacc += dt;
            count += 1;

            obs.observe(self.stepper.get_system_mut());
        }

        (tacc, count)
    }

    // pub fn integrate_n_times<'a>(&'a mut self, n: u64, dt: f64, obs: &mut Observer<ODE<State=T> + 'a>) {
    pub fn integrate_n_times(&mut self, n: u64, dt: f64, obs: &mut Observer<Box<ODE<State=T>>>) {
        for _i in 0..n {
            self.stepper.do_step(dt);
        }

        obs.observe(self.stepper.get_system_mut());
    }
}
