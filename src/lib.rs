extern crate float_cmp;

mod traits;

use traits::ODE;
use traits::RungeKutta4;

#[test]
fn simple_rk4() {
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
            self.a *  x.sin()
        }

        fn differentiate_into(&self, x: &f64, into: &mut f64) {
            *into = self.a * x.sin();
        }

        fn update_state(&mut self, x: &f64) {
            self.x = *x;
        }
    }

    let result = 1.0863557235971943; // Checked by hand through numpy

    let sys = SimpleODE { a: 1., x: 1. };

    let mut rk4 = RungeKutta4::new(Box::new(sys), 0.1);
    rk4.do_step();

    assert!(result.approx_eq_ulps(rk4.system.get_state(), 2));
}
