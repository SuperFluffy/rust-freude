#![feature(test)]

extern crate freude;
extern crate test;

use freude::{ODE,RungeKutta4,Stepper};
use test::black_box;

#[bench]
fn rk4_freude(bench: &mut test::Bencher) {
    // Some generic ODE with f'(x) = a * sin x
    #[derive(Clone)]
    struct SimpleODE {
        a: f64,
        x: f64,
    }

    impl ODE for SimpleODE {
        type State = f64;

        fn differentiate_into(&mut self, x: &f64, into: &mut f64) {
            *into = self.a + x.sin();
        }
    }

    let mut x_init = 1.0;
    let sys = SimpleODE { a: 1., x: x_init };

    let mut rk4 = RungeKutta4::new(sys, 0.1, &x_init);
    bench.iter(|| { rk4.do_step(&mut x_init); });
    let _x = black_box(rk4);
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
