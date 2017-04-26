#![feature(box_syntax, test, fmt_internals)]

extern crate freude;
#[macro_use]
extern crate freude_derive;

use freude::{
    Integrator,
    Ode,
    Stepper,
    RungeKutta4,
};

#[derive(Integrator)]
#[freude(after_integration="Observer::after_integration", after_step="Observer::after_step")]
struct Observer<A,B,C>
    where A: Clone,
          B: Stepper<State=A>,
          C: Ode<State=A>,
{
    #[freude(state)]
    field_1: A,
    #[freude(stepper)]
    field_2: B,
    #[freude(system)]
    field_3: C,
}

struct Sys { }

impl Ode for Sys {
    type State = Vec<f64>;

    fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
        println!("differentiate_into called");
    }
}

impl<A,B,C> Observer<A,B,C>
    where A: Clone,
          B: Stepper<State=A>,
          C: Ode<State=A>,
{
    fn after_step(&mut self, dt: f64) {
        println!("after_step works: dt = {}", dt);
    }

    fn after_integration(&mut self, dt: f64, t: f64) {
        println!("after_integration works: dt = {}, t = {}", dt, t);
    }
}

#[test]
fn test_observer() {
    let state = vec![1.0,1.0,1.0];
    let stepper = RungeKutta4::new(&state, 0.1);
    let sys = Sys {};

    let obs = Observer {
        field_1: state,
        field_2: stepper,
        field_3: sys,
    };

    obs.integrate_time(1.0);
}
