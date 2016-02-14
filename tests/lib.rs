#![feature(braced_empty_structs)]

extern crate float_cmp;

extern crate freude;

use freude::{Integrator,Observer,ODE,RungeKutta4,Stepper};

#[test]
fn stepper_rk4() {
    use float_cmp::ApproxEqUlps;
    // Some generic ODE with f'(x) = a * sin x
    struct SimpleODE {
        a: f64,
        x: f64,
    }

    impl ODE for SimpleODE {
        type State = f64;
        fn get_state(&self) -> &f64 {
            &self.x
        }

        fn differentiate_into(&mut self, x: &f64, into: &mut f64) {
            *into = self.a + x.sin();
        }

        fn update_state(&mut self, x: &f64) {
            self.x = *x;
        }
    }

    let result = 1.1886978786386662; // Checked by hand through numpy

    let sys = SimpleODE { a: 1., x: 1. };

    let mut rk4 = RungeKutta4::<f64>::new(Box::new(sys));
    rk4.do_step(0.1);

    assert!(result.approx_eq_ulps(rk4.get_state(), 2i64));
}

#[test]
fn integrator_rk4() {
    use float_cmp::ApproxEqUlps;
    // Some generic ODE with f'(x) = a * sin x
    struct SimpleODE {
        a: f64,
        x: f64,
    }

    struct UselessObserver { };

    impl ODE for SimpleODE {
        type State = f64;
        fn get_state(&self) -> &f64 {
            &self.x
        }

        fn differentiate_into(&mut self, x: &f64, into: &mut f64) {
            *into = self.a + x.sin();
        }

        fn update_state(&mut self, x: &f64) {
            self.x = *x;
        }
    }

    impl<T: ?Sized> Observer<T> for UselessObserver {
        fn observe(&mut self, _ode: &mut T) { }
    }

    let result = 2.8010679346446947; // Checked by hand through numpy

    let sys = SimpleODE { a: 1., x: 1. };
    let rk4 = RungeKutta4::<f64>::new(Box::new(sys));
    let mut integrator = Integrator::new(Box::new(rk4));

    let mut obs = UselessObserver {};

    integrator.integrate_n_times(10, 0.1, &mut obs); 

    assert!(result.approx_eq_ulps(integrator.get_state(), 2i64));
}

#[test]
fn integrator_steps_vs_range() {
    // Explicitly running for n steps should give the same result as running
    // for a time interval T with timesteps of size ΔT
    use float_cmp::ApproxEqUlps;
    // Some generic ODE with f'(x) = a * sin x
    struct SimpleODE {
        a: f64,
        x: f64,
    }

    struct UselessObserver { };

    impl ODE for SimpleODE {
        type State = f64;
        fn get_state(&self) -> &f64 {
            &self.x
        }

        fn differentiate_into(&mut self, x: &f64, into: &mut f64) {
            *into = self.a + x.sin();
        }

        fn update_state(&mut self, x: &f64) {
            self.x = *x;
        }
    }

    impl<T: ?Sized> Observer<T> for UselessObserver {
        fn observe(&mut self, _ode: &mut T) { }
    }

    // FIXME
    // Implementing Clone for the ODEs, Steppers, and Integrators would significantly
    // remove the boiler plate here.
    let sys1 = SimpleODE { a: 1., x: 1. };
    let sys2 = SimpleODE { a: 1., x: 1. };
    let step1 = RungeKutta4::<f64>::new(Box::new(sys1));
    let step2 = RungeKutta4::<f64>::new(Box::new(sys2));

    let mut integrator1 = Integrator::new(Box::new(step1));
    let mut integrator2 = Integrator::new(Box::new(step2));

    let mut obs = UselessObserver {};

    let n1 = 10;
    let dt = 0.1;
    let t0 = 0.0;
    let tf = t0 + dt * (n1 as f64);

    integrator1.integrate_n_times(n1, dt, &mut obs); 

    let (_tf,_n2) = integrator2.integrate_between(t0,tf,dt, &mut obs); 

    assert!(integrator1.get_state().approx_eq_ulps(integrator2.get_state(), 2i64));
}

