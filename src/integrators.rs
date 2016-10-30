use observers::Observer;
use steppers::Stepper;

/// An integrator that takes ownership of stepper and observer.
pub struct Integrator<S, O>
{
    stepper: S,
    observer: O,
}

impl<S, O> Integrator<S, O>
    where S: Stepper,
          O: Observer<System = <S as Stepper>::System>
{
    pub fn new(stepper: S, observer: O) -> Self {
        Integrator {
            stepper: stepper,
            observer: observer,
        }
    }

    pub fn swap_observer<P: Observer<System = <S as Stepper>::System>>(self, observer: P) -> Integrator<S, P> {
        Integrator { stepper: self.stepper, observer: observer }
    }

    pub fn stepper_ref(&self) -> &S {
        &self.stepper
    }

    pub fn stepper_mut(&mut self) -> &mut S {
        &mut self.stepper
    }

    pub fn observer_ref(&self) -> &O {
        &self.observer
    }

    pub fn observer_mut(&mut self) -> &mut O {
        &mut self.observer
    }

    pub fn stepper_and_observer_mut(&mut self) -> (&mut S, &mut O) {
        (&mut self.stepper, &mut self.observer)
    }

    pub fn integrate_time(&mut self, t: f64) -> (f64, usize)
    {
        let mut tacc = 0f64;;
        let mut count = 0;

        let dt = self.stepper.timestep();

        // Ensure t_final is not exceeded
        while (tacc + dt) <= t {
            self.stepper.do_step();
            tacc += dt;
            count += 1;

            self.observer.observe(self.stepper.get_system_mut(), dt);
        }

        (tacc, count)
    }

    pub fn integrate_time_range<I: IntoIterator<Item=f64>>(&mut self, ts: I) -> (f64, usize)
    {
        let mut count = 0;
        let mut tacc = 0f64;

        for t in ts {
            let (totacc,tocount) = self.integrate_time(t);
            tacc += totacc;
            count += tocount;
        }
        (tacc, count)
    }

    pub fn integrate_n_steps(&mut self, n: usize) -> (f64,usize)
    {
        let dt = self.stepper.timestep();
        let mut tacc = 0f64;
        for _i in 0..n {
            self.stepper.do_step();
            tacc += dt;

            self.observer.observe(self.stepper.get_system_mut(), dt);
        }
        (tacc, n)
    }

    pub fn integrate_n_range<I: IntoIterator<Item=usize>>(&mut self, ns: I,) -> (f64, usize)
    {
        let mut count = 0;
        let mut tacc = 0f64;
        for n in ns.into_iter() {
            let (totacc,tocount) = self.integrate_n_steps(n);
            tacc += totacc;
            count += tocount; // Same as n
        }
        (tacc, count)
    }

    pub fn warmup_n_steps(&mut self, n: usize) {
        for _i in 0..n {
            self.stepper.do_step();
        }
    }

    pub fn warmup_time(&mut self, t: f64) -> (f64, usize) {
        let mut tacc = 0f64;;
        let mut count = 0;

        let dt = self.stepper.timestep();

        // Ensure t_final is not exceeded
        while (tacc + dt) <= t {
            self.stepper.do_step();
            tacc += dt;
            count += 1;
        }

        (tacc, count)
    }
}
