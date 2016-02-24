#![feature(augmented_assignments)]

#[macro_use(izip)]
extern crate itertools;
extern crate ndarray;

// Re-exports
pub use integrators::Integrator;
pub use traits::{Observer, ODE};
pub use steppers::{Stepper, RungeKutta4};

mod integrators;
mod steppers;
mod traits;

pub mod utils;
