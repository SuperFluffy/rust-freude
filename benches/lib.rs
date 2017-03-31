#![feature(test)]

extern crate blas;
extern crate freude;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate test;

use blas::c as cblas;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;

use rand::distributions::{Normal, Range};
use std::f64;

use freude::{ODE,RungeKutta4,Stepper};
use test::black_box;

#[derive(Clone)]
struct ChaoticNeuralNet {
    coupling: Array2<f64>,
    nonlinearity: f64,
    size: usize,
    temp_tanh: Array1<f64>,
}

impl ODE for ChaoticNeuralNet {
    type State = Array1<f64>;

    fn differentiate_into(&mut self, state: &Array1<f64>, derivative: &mut Array1<f64>) {
        // dx_i/dt = x_i - ∑_j J_ij tanh(g·x_j)
        let g = self.nonlinearity;
        self.temp_tanh.zip_mut_with(&state, |t, x| { *t = f64::tanh(g * *x); });

        // Calculate ∑_j J_ij · tanh(g · x_j)
        cblas::dgemv(cblas::Layout::RowMajor, // layout
            cblas::Transpose::None, // transa
            self.size as i32, // m
            self.size as i32, // n
            1.0, // alpha
            self.coupling.as_slice_memory_order().unwrap(), // a
            self.size as i32, // lda
            self.temp_tanh.as_slice_memory_order().unwrap(), // x
            1, // incx
            0.0, // beta
            derivative.as_slice_memory_order_mut().unwrap(), // y
            1, // incy
        );
        let mut derivative = derivative.view_mut();
        derivative -= state;
    }
}

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

#[bench]
fn chaotic_net(bench: &mut test::Bencher) {
    let size = 8192;
    let mean = 0.0;
    let nonlinearity = 1.5;
    let std_dev = 1.1;

    let dist = Normal::new(mean, std_dev / f64::sqrt(size as f64));
    let mut coupling = Array2::random([size,size], dist);
    coupling.diag_mut().map_inplace(|c| { *c = 0.0; });

    let temp_tanh = Array1::zeros(size);

    let chaotic_net = ChaoticNeuralNet {
        coupling: coupling,
        nonlinearity: nonlinearity,
        size: size,
        temp_tanh: temp_tanh,
    };

    let between = Range::new(-1.0, 1.0);

    let mut initial_state = Array1::random(size, between);

    let mut rk4 = RungeKutta4::new(chaotic_net, 0.1, &initial_state);

    bench.iter(|| { rk4.do_step(&mut initial_state); });
    let _x = black_box(rk4);
}
