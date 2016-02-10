#![feature(augmented_assignments)]

#[macro_use(izip)] extern crate itertools;
extern crate ndarray;

pub use traits::ODE;
pub use steppers::{Stepper, RungeKutta4};

mod steppers;
mod traits;
