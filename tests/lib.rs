#[macro_use] extern crate approx;

extern crate freude;
extern crate ndarray;

use freude::{Integrator,Observer,ODE,RungeKutta4,Stepper};
use std::any::Any;

// Some generic ODE with dx/dt = a * x ⇒ c * exp(a*t)2
struct SimpleODE {
    a: f64,
    c: f64,
    x: f64,
}

impl ODE for SimpleODE {
    type State = f64;

    fn as_any(&self) -> &Any {
        self
    }

    fn get_state(&self) -> &f64 {
        &self.x
    }

    fn get_state_mut(&mut self) -> &mut f64 {
        &mut self.x
    }

    fn differentiate_into(&mut self, x: &f64, into: &mut f64) {
        *into = self.a * x;
    }

    fn update_state(&mut self, x: &f64) {
        self.x = *x;
    }
}

#[test]
fn stepper_rk4() {
    let sys = SimpleODE { a: 1., c: 1., x: 1. };

    let timestep = 0.1;
    let exact = sys.c * f64::exp(sys.a * timestep);

    let mut rk4 = RungeKutta4::<f64>::new(Box::new(sys), timestep);
    rk4.do_step();

    assert_relative_eq!(exact, rk4.get_state(), max_relative=timestep.powi(4));
}

#[test]
fn integrator_rk4() {
    struct UselessObserver { };

    impl<T: ?Sized> Observer<T> for UselessObserver {
        fn observe(&mut self, _ode: &mut T, _: f64) { }
    }

    let sys = SimpleODE { a: 1., c: 1., x: 1. };

    let timestep = 0.1;
    let steps = 100;
    let total_time = timestep * steps as f64;

    let exact = sys.c * f64::exp(sys.a * total_time);

    let rk4 = RungeKutta4::<f64>::new(Box::new(sys), timestep);
    let mut integrator = Integrator::new(Box::new(rk4));

    let mut obs = UselessObserver {};

    integrator.integrate_n_steps(steps, &mut obs); 

    assert_relative_eq!(exact, integrator.get_state(), max_relative=timestep.powi(4));
}

#[test]
fn integrator_steps_vs_range() {
    struct UselessObserver { };

    impl<T: ?Sized> Observer<T> for UselessObserver {
        fn observe(&mut self, _ode: &mut T, _: f64) { }
    }

    // FIXME
    // Implementing Clone for the ODEs, Steppers, and Integrators would significantly
    // remove the boiler plate here.
    let sys1 = SimpleODE { a: 1., c: 1., x: 1. };
    let sys2 = SimpleODE { a: 1., c: 1., x: 1. };

    let stepsize = 0.1;
    let steps = 10;
    let total_time = stepsize * (steps as f64);

    let step1 = RungeKutta4::<f64>::new(Box::new(sys1), stepsize);
    let step2 = RungeKutta4::<f64>::new(Box::new(sys2), stepsize);

    let mut integrator1 = Integrator::new(Box::new(step1));
    let mut integrator2 = Integrator::new(Box::new(step2));

    let mut obs = UselessObserver {};

    integrator1.integrate_n_steps(steps, &mut obs); 

    let (_tf,_n2) = integrator2.integrate_time(total_time, &mut obs); 

    assert_relative_eq!(integrator1.get_state(), integrator2.get_state());
}

#[test]
fn test_fold() {
    use freude::utils::fold_with_2;
    use ndarray::arr1;

    let a = arr1(&[1,2,3]);
    let b = arr1(&[1,2,3]);
    let c = arr1(&[1,2,3]);

    let acc = fold_with_2(a.view(), b.view(), c.view(), 0, |acc,a,b,c| acc + a * b * c).unwrap();

    assert_eq!(acc, 36);
}
