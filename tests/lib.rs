#[macro_use]
extern crate approx;

extern crate freude;
extern crate ndarray;

use freude::ODE;

// Some generic ODE with dx/dt = a * x ⇒ x(t) = c * exp(a*t)
#[derive(Clone)]
struct SimpleODE {
    a: f64,
    c: f64,
}

impl ODE for SimpleODE {
    type State = f64;

    fn differentiate_into(&mut self, x: &f64, into: &mut f64) {
        *into = self.a * x;
    }
}

macro_rules! mk_stepper_test {
    ($stepper:ident, $error_order:expr) => {
        #[allow(non_snake_case)]
        mod $stepper {
            use freude::*;

            use super::SimpleODE;

            #[test]
            fn stepper() {
                let mut sys = SimpleODE { a: 1., c: 1. };

                let mut x = 1.0;

                let timestep = 0.1;
                let exact = sys.c * f64::exp(sys.a * timestep);

                let mut stepper = $stepper::new(&x, timestep);
                stepper.do_step(&mut sys, &mut x);

                assert_relative_eq!(exact, x, max_relative=timestep.powi($error_order));
            }

            #[test]
            fn integrator() {
                let mut sys = SimpleODE { a: 1., c: 1. };
                let mut x = 1.0;

                let timestep = 0.1;
                let steps = 100;
                let total_time = timestep * steps as f64;

                let exact = sys.c * f64::exp(sys.a * total_time);

                let mut stepper = $stepper::new(&x, timestep);

                stepper.integrate_n_steps(&mut sys, &mut x, steps);

                assert_relative_eq!(exact, x, max_relative=(steps as f64 * timestep.powi($error_order)));
            }

            #[test]
            fn integrator_steps_vs_time() {
                // FIXME
                // Implementing Clone for the ODEs, Steppers, and Integrators would significantly
                // remove the boiler plate here.
                let mut sys = SimpleODE { a: 1., c: 1. };

                let mut x1 = 1.0;
                let mut x2 = x1;

                let timestep = 0.1;
                let steps = 10;
                let total_time = timestep * (steps as f64);

                let mut stepper = $stepper::new(&x1, timestep);

                stepper.integrate_n_steps(&mut sys, &mut x1, steps);
                stepper.integrate_time(&mut sys, &mut x2, total_time);

                assert_relative_eq!(x1, x2);
            }
        }
    };

    ($($stepper:ident:$error_order:expr),+) => {
        $(
            mk_stepper_test!($stepper, $error_order);
        )+
    };
}

mk_stepper_test!(Euler:1,Heun:2,RungeKutta4:4);
