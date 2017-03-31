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
            use freude::observers::NullObserver;

            use super::SimpleODE;

            #[test]
            fn stepper() {
                let sys = SimpleODE { a: 1., c: 1. };

                let mut x = 1.0;

                let timestep = 0.1;
                let exact = sys.c * f64::exp(sys.a * timestep);

                let mut stepper = $stepper::new(sys, timestep, &x);
                stepper.do_step(&mut x);

                assert_relative_eq!(exact, x, max_relative=timestep.powi($error_order));
            }

            #[test]
            fn integrator() {
                let sys = SimpleODE { a: 1., c: 1. };

                let x = 1.0;

                let timestep = 0.1;
                let steps = 100;
                let total_time = timestep * steps as f64;

                let exact = sys.c * f64::exp(sys.a * total_time);

                let nullobs = NullObserver::new();
                let mut integrator = Integrator::new($stepper::new(sys, timestep, &x), nullobs, x);

                integrator.integrate_n_steps(steps);

                assert_relative_eq!(exact, integrator.state_ref(), max_relative=(steps as f64 * timestep.powi($error_order)));
            }

            #[test]
            fn integrator_steps_vs_range() {
                // FIXME
                // Implementing Clone for the ODEs, Steppers, and Integrators would significantly
                // remove the boiler plate here.
                let sys1 = SimpleODE { a: 1., c: 1. };
                let sys2 = SimpleODE { a: 1., c: 1. };

                let x = 1.0;

                let stepsize = 0.1;
                let steps = 10;
                let total_time = stepsize * (steps as f64);


                let mut integrator1 = Integrator::new($stepper::new(sys1, stepsize, &x), NullObserver::new(), x);
                let mut integrator2 = Integrator::new($stepper::new(sys2, stepsize, &x), NullObserver::new(), x);
                integrator1.integrate_n_steps(steps);

                let (_tf,_n2) = integrator2.integrate_time(total_time);

                assert_relative_eq!(integrator1.state_ref(), integrator2.state_ref());
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
