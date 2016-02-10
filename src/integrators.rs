use steppers::Stepper;

struct Integrator {
    stepper: Box<Stepper>,
}

impl Integrator {
    pub fn integrate_between(&mut self, t0: f64, tf: f64, dt: f64) -> (f64,u64) {
        let mut tacc = t0;
        let mut count = 0;

        while tacc <= tf {
            self.stepper.do_step(dt);
            tacc += dt;
            count += 1;
        }

        (tacc, count)
    }

    pub fn integrate_n_times(&mut self, n: u64, dt: f64) {
        for _i in 0..n {
            self.stepper.do_step(dt);
        }
    }
}
