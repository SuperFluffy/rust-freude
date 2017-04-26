use ode::Ode;
use stepper::Stepper;

pub trait Integrator {
    type State: Clone;
    type Stepper: Stepper<State = Self::State>;
    type System: Ode<State = Self::State>;

    fn integrate_n_steps(&mut self, n: usize) -> f64;

    fn integrate_time(&mut self, t: f64) -> (f64, usize);
}
