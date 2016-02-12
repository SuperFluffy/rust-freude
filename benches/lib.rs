#![feature(test)]

extern crate freude;
extern crate test;

use freude::{ODE,RungeKutta4,Stepper};
use test::black_box;

#[bench]
fn rk4_freude(bench: &mut test::Bencher) {
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

        fn differentiate_into(&self, x: &f64, into: &mut f64) {
            *into = self.a + x.sin();
        }

        fn update_state(&mut self, x: &f64) {
            self.x = *x;
        }
    }

    let sys = SimpleODE { a: 1., x: 1. };

    let mut rk4 = RungeKutta4::<f64>::new(Box::new(sys));

    bench.iter(|| { rk4.do_step(0.1); });
}

#[bench]
fn rk4_manual(bench: &mut test::Bencher) {
    fn f(x: f64) -> f64 {
        1f64 + x.sin()
    }

    fn rk4(x: &mut f64, dt: f64) {
        let k1 = f(*x);
        let k2 = f(*x + dt/2.0 * k1);
        let k3 = f(*x + dt/2.0 * k2);
        let k4 = f(*x + dt * k3);

        *x = *x + dt/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    let mut x = black_box(1.0);
    bench.iter(|| { rk4(&mut x, 0.1); });
    let _x = black_box(x); // Don't optimize the bench away.
}
