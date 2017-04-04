extern crate freude;
extern crate tuple;

use freude::RungeKutta4;
use freude::Integrator;
use freude::observers::NullObserver;
use tuple::T3;

fn main() {
    let dt = 0.1;
    let time = 1000.0;

    let sigma = 10.0;
    let beta = 8.0/3.0;
    let rho = 28.0;

    let f: Box<Fn(T3<_,_,_>) -> T3<_,_,_>> = Box::new(
        move |T3(x,y,z): T3<f64,f64,f64>| {
            T3( sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z
            )
        }
    );

    let state = T3(0.5,1.0,1.5);
    let rk4 = RungeKutta4::new(f, dt, &state);
    let mut integrator = Integrator::new(rk4, NullObserver::new(), state);
    let (_, steps) = integrator.integrate_time(time);
    println!("Integrated Lorenz systems for {} steps", steps);
    println!("Final state: {:?}", integrator.state_ref());
}
