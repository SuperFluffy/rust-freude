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

        fn differentiate(&self, x: &f64) -> f64 {
            self.a + x.sin()
        }

        fn differentiate_into(&self, x: &f64, into: &mut f64) {
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

        fn differentiate(&self, x: &f64) -> f64 {
            self.a + x.sin()
        }

        fn differentiate_into(&self, x: &f64, into: &mut f64) {
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

