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

struct NullOde<T> {
    temp_derivative: T
}

impl ODE for NullOde<Array1<f64>> {
    type State = Array1<f64>;

    fn differentiate_into(&mut self, _: &Array1<f64>, derivative: &mut Array1<f64>) {
        derivative.assign(&self.temp_derivative);
    }
}

impl ODE for NullOde<Vec<f64>> {
    type State = Vec<f64>;

    fn differentiate_into(&mut self, _: &Vec<f64>, derivative: &mut Vec<f64>) {
        derivative.clone_from(&self.temp_derivative);
    }
}

struct ChaoticNeuralNet {
    coupling: Array2<f64>,
    num_lyapunov_vectors: usize,
    nonlinearity: f64,
    size: usize,
    temp_prod: Array2<f64>,
    temp_tanh: Array1<f64>,
    temp_tanh2: Array1<f64>,
}

impl ODE for ChaoticNeuralNet {
    type State = Array2<f64>;

    fn differentiate_into(&mut self, state: &Array2<f64>, derivative: &mut Array2<f64>) {
        let g = self.nonlinearity;
        self.temp_tanh.zip_mut_with(&state.row(0), |t, x| { *t = f64::tanh(g * *x); });

        let derivative = derivative.view_mut();

        let (mut system_derivative, mut lyapunov_derivative) = derivative.split_at(Axis(0), 1);

        let mut system_derivative = system_derivative.row_mut(0);

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
            system_derivative.as_slice_memory_order_mut().unwrap(), // y
            1, // incy
        );

        system_derivative -= &state.row(0);

        {
        let temp_tanh = &self.temp_tanh;
        self.temp_tanh2.zip_mut_with(temp_tanh, |t2, t| { *t2 = 1.0 - t.powi(2); });
        }

        for i in 0..self.num_lyapunov_vectors {
            let mut t = self.temp_prod.row_mut(i);
            t.assign(&self.temp_tanh2);
            t *= &state.row(i+1);
        }

        cblas::dgemm(cblas::Layout::ColumnMajor, // layout
            cblas::Transpose::Ordinary, // transa
            cblas::Transpose::None, // transb
            self.size as i32, // m
            self.num_lyapunov_vectors as i32, // n
            self.size as i32, // k
            self.nonlinearity, // alpha
            self.coupling.as_slice_memory_order().unwrap(), // a
            self.size as i32, // lda
            self.temp_prod.as_slice_memory_order().unwrap(), // b
            self.size as i32, // ldb
            0.0, // beta
            lyapunov_derivative.as_slice_memory_order_mut().unwrap(), // y
            self.size as i32, // ldc
        );

        for i in 0..self.num_lyapunov_vectors {
            let mut v = lyapunov_derivative.row_mut(i);
            v -= &state.row(i+1);
        }
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
    let size = 512;
    let mean = 0.0;
    let nonlinearity = 1.5;
    let std_dev = 1.1;

    let num_lyapunov = size;

    let dist = Normal::new(mean, std_dev / f64::sqrt(size as f64));
    let mut coupling = Array2::random([size,size], dist);
    coupling.diag_mut().map_inplace(|c| { *c = 0.0; });

    let temp_prod = Array2::zeros((num_lyapunov,size));
    let temp_tanh = Array1::zeros(size);
    let temp_tanh2 = Array1::zeros(size);

    let chaotic_net = ChaoticNeuralNet {
        coupling: coupling,
        num_lyapunov_vectors: num_lyapunov,
        nonlinearity: nonlinearity,
        size: size,
        temp_prod: temp_prod,
        temp_tanh: temp_tanh,
        temp_tanh2: temp_tanh2,
    };

    let between = Range::new(-1.0, 1.0);
    let rows = 1 + num_lyapunov;

    let mut initial_state = Array2::random([rows, size], between);

    let mut rk4 = RungeKutta4::new(chaotic_net, 0.1, &initial_state);

    bench.iter(|| { rk4.do_step(&mut initial_state); });
    let _x = black_box(rk4);
}

#[bench]
fn rk4_array_speed(bench: &mut test::Bencher) {
    let size = 8192;

    let mut state = Array1::zeros(size);
    let system = NullOde {
        temp_derivative: state.clone(),
    };

    let rk4 = RungeKutta4::new(system, 0.1, &state);
    let mut rk4 = black_box(rk4);
    bench.iter(|| {rk4.do_step(&mut state); });
}

#[bench]
fn rk4_vec_speed(bench: &mut test::Bencher) {
    let size = 8192;

    let mut state = vec![0.0; size];
    let system = NullOde {
        temp_derivative: state.clone(),
    };

    let rk4 = RungeKutta4::new(system, 0.1, &state);
    let mut rk4 = black_box(rk4);
    bench.iter(|| {rk4.do_step(&mut state); });
}
