use freude::RungeKutta4;
use freude::Stepper;
use tuple::T3;

fn main() {
    let dt = 0.1;
    let time = 1000.0;

    let sigma = 10.0;
    let beta = 8.0 / 3.0;
    let rho = 28.0;

    let mut f: &dyn Fn(T3<_, _, _>) -> T3<_, _, _> = &move |T3(x, y, z): T3<f64, f64, f64>| {
        T3(sigma * (y - x), x * (rho - z) - y, x * y - beta * z)
    };

    let mut state = T3(0.5, 1.0, 1.5);
    let mut rk4 = RungeKutta4::new(&state, dt);
    let (_, steps) = rk4.integrate_time(&mut f, &mut state, time);
    println!("Integrated Lorenz systems for {} steps", steps);
    println!("Final state: {:?}", state);
}
