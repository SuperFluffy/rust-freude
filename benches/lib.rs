#![feature(test)]

extern crate blas;
extern crate freude;

#[macro_use(azip)]
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
extern crate test;

use blas::c as cblas;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;

use rand::distributions::IndependentSample;
use rand::distributions::{Normal, Range};
use std::f64;
use std::marker::PhantomData;

use freude::{Ode,RungeKutta4,Stepper};
use test::black_box;

struct NullSystem<T> {
    _phantom: PhantomData<T>,
}

impl Ode for NullSystem<Array1<f64>> {
    type State = Array1<f64>;

    fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
        derivative.assign(state);
    }
}

impl Ode for NullSystem<Vec<f64>> {
    type State = Vec<f64>;

    fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
        derivative.clone_from(state);
    }
}

struct Kuramoto {
    frequencies: Vec<f64>,
    size: usize,
    temp_sin: Vec<f64>,
    temp_cos: Vec<f64>,
}

impl Ode for Kuramoto {
    type State = Vec<f64>;

    fn differentiate_into(&mut self, state: &Vec<f64>, derivative: &mut Vec<f64>) {
        // dφ_i/dt =  ω_i + 1/N ∑_j { sin(φ_j - φ_i) }
        //         =  ω_i + 1/N ∑_j { sin(φ_j)cos(φ_i)  - cos(φ_j)sin(φ_i) }
        for (t, s) in self.temp_sin.iter_mut().zip(state.iter()) {
            *t = f64::sin(*s);
        }

        for (t, c) in self.temp_cos.iter_mut().zip(state.iter()) {
            *t = f64::cos(*c);
        }

        let mut sin_sum = self.temp_sin.iter().fold(0.0, |acc, s| { acc + s });
        let mut cos_sum = self.temp_cos.iter().fold(0.0, |acc, c| { acc + c });

        sin_sum /= self.size as f64;
        cos_sum /= self.size as f64;

        azip!(mut d (derivative), f (&self.frequencies), sin (&self.temp_sin), cos (&self.temp_cos) in {
            *d = f + sin_sum * cos - cos_sum * sin
        })
    }
}

struct ChaoticNeuralNet {
    coupling: Array2<f64>,
    nonlinearity: f64,
    size: usize,
    temp_tanh: Array1<f64>,
}

impl Ode for ChaoticNeuralNet {
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
    }

    impl Ode for SimpleODE {
        type State = f64;

        fn differentiate_into(&mut self, x: &f64, into: &mut f64) {
            *into = *x + x.sin();
        }
    }

    let mut x = 1.0;
    let mut sys = SimpleODE { a: 1. };

    let mut rk4 = RungeKutta4::new(&x, 0.1);
    bench.iter(|| { rk4.do_step(&mut sys, &mut x); });
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
fn chaotic_net_derivative(bench: &mut test::Bencher) {
    let size = 8192;
    let mean = 0.0;
    let nonlinearity = 1.5;
    let std_dev = 1.1;

    let dist = Normal::new(mean, std_dev / f64::sqrt(size as f64));
    let mut coupling = Array2::random([size,size], dist);
    coupling.diag_mut().map_inplace(|c| { *c = 0.0; });

    let temp_tanh = Array1::zeros(size);

    let mut chaotic_net = ChaoticNeuralNet {
        coupling: coupling,
        nonlinearity: nonlinearity,
        size: size,
        temp_tanh: temp_tanh,
    };

    let between = Range::new(-1.0, 1.0);

    let state = Array1::random(size, between);
    let derivative = Array1::zeros(size);
    let mut derivative = black_box(derivative);
    bench.iter(|| { chaotic_net.differentiate_into(&state, &mut derivative)});
}

#[bench]
fn chaotic_net_rk4(bench: &mut test::Bencher) {
    let size = 8192;
    let mean = 0.0;
    let nonlinearity = 1.5;
    let std_dev = 1.1;

    let dist = Normal::new(mean, std_dev / f64::sqrt(size as f64));
    let mut coupling = Array2::random([size,size], dist);
    coupling.diag_mut().map_inplace(|c| { *c = 0.0; });

    let temp_tanh = Array1::zeros(size);

    let mut chaotic_net = ChaoticNeuralNet {
        coupling: coupling,
        nonlinearity: nonlinearity,
        size: size,
        temp_tanh: temp_tanh,
    };

    let between = Range::new(-1.0, 1.0);

    let mut state = Array1::random(size, between);

    let rk4 = RungeKutta4::new(&state, 0.1);
    let mut rk4 = black_box(rk4);
    bench.iter(|| { rk4.do_step(&mut chaotic_net, &mut state); });
}

#[bench]
fn kuramoto_vec(bench: &mut test::Bencher) {
    let size = 8192;

    let mut rng = ::rand::thread_rng();

    let dist = Normal::new(0.0, 1.0);
    let frequencies: Vec<_> = ::std::iter::repeat(()).take(size).map(|_| {
        dist.ind_sample(&mut rng)
    }).collect();

    let temp_sin = vec![0.0; size];
    let temp_cos = vec![0.0; size];

    let mut kuramoto = Kuramoto {
        frequencies: frequencies,
        size: size,
        temp_sin: temp_sin,
        temp_cos: temp_cos,
    };

    let pi = ::std::f64::consts::PI;
    let between = Range::new(-pi, pi);

    let mut state: Vec<_> = ::std::iter::repeat(()).take(size).map(|_| {
        between.ind_sample(&mut rng)
    }).collect();

    let rk4 = RungeKutta4::new(&state, 0.1);
    let mut rk4 = black_box(rk4);
    bench.iter(|| { rk4.do_step(&mut kuramoto, &mut state); });
}

#[bench]
fn rk4_array_speed(bench: &mut test::Bencher) {
    let size = 8192;

    let mut state = Array1::zeros(size);
    let mut system = NullSystem { _phantom: PhantomData::<Array1<f64>> };

    let rk4 = RungeKutta4::new(&state, 0.1);
    let mut rk4 = black_box(rk4);
    bench.iter(|| {rk4.do_step(&mut system, &mut state); });
}

#[bench]
fn rk4_vec_speed(bench: &mut test::Bencher) {
    let size = 8192;

    let mut state = vec![0.0; size];
    let mut system = NullSystem { _phantom: PhantomData::<Vec<f64>> };

    let rk4 = RungeKutta4::new(&state, 0.1);
    let mut rk4 = black_box(rk4);
    bench.iter(|| {rk4.do_step(&mut system, &mut state); });
}
