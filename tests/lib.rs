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
    x: f64,
}

impl ODE for SimpleODE {
    type State = f64;

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
fn test_fold() {
    use freude::utils::fold_with_2;
    use ndarray::arr1;

    let a = arr1(&[1,2,3]);
    let b = arr1(&[1,2,3]);
    let c = arr1(&[1,2,3]);

    let acc = fold_with_2(a.view(), b.view(), c.view(), 0, |acc,a,b,c| acc + a * b * c).unwrap();

    assert_eq!(acc, 36);
}

macro_rules! mk_stepper_tests {
    ($stepper:ident, $error_order:expr) => {
        #[allow(non_snake_case)]
        mod $stepper {
            use freude::*;
            use freude::observers::NullObserver;

            use super::SimpleODE;

            #[test]
            fn stepper() {
                let sys = SimpleODE { a: 1., c: 1., x: 1. };

                let timestep = 0.1;
                let exact = sys.c * f64::exp(sys.a * timestep);

                let mut stepper = $stepper::new(sys, timestep);
                stepper.do_step();

                assert_relative_eq!(exact, stepper.get_state(), max_relative=timestep.powi($error_order));
            }

            #[test]
            fn integrator() {
                let sys = SimpleODE { a: 1., c: 1., x: 1. };

                let timestep = 0.1;
                let steps = 100;
                let total_time = timestep * steps as f64;

                let exact = sys.c * f64::exp(sys.a * total_time);

                let nullobs = NullObserver::new();
                let mut integrator = Integrator::new($stepper::new(sys, timestep), nullobs);

                integrator.integrate_n_steps(steps);

                assert_relative_eq!(exact, integrator.stepper_ref().get_state(), max_relative=(steps as f64 * timestep.powi($error_order)));
            }

            #[test]
            fn integrator_steps_vs_range() {
                // FIXME
                // Implementing Clone for the ODEs, Steppers, and Integrators would significantly
                // remove the boiler plate here.
                let sys1 = SimpleODE { a: 1., c: 1., x: 1. };
                let sys2 = SimpleODE { a: 1., c: 1., x: 1. };

                let stepsize = 0.1;
                let steps = 10;
                let total_time = stepsize * (steps as f64);


                let mut integrator1 = Integrator::new($stepper::new(sys1, stepsize), NullObserver::new());
                let mut integrator2 = Integrator::new($stepper::new(sys2, stepsize), NullObserver::new());
                integrator1.integrate_n_steps(steps);

                let (_tf,_n2) = integrator2.integrate_time(total_time);

                assert_relative_eq!(integrator1.stepper_ref().get_state(), integrator2.stepper_ref().get_state());
            }
        }
    };

    ($($stepper:ident:$error_order:expr),+) => {
        $(
            mk_stepper_tests!($stepper, $error_order);
        )+
    };
}

mk_stepper_tests!(Euler:1,Heun:2,RungeKutta4:4);
