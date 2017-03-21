use std::marker::PhantomData;

pub trait Observer {
    type State;
    type System;
    // The observer should be allowed to affect the observable,
    // e.g. in the case of reading the Lyapunov exponents, or resetting
    // a phase to the range [0,2π)
    fn observe(&mut self, &Self::System, &mut Self::State, f64);

    fn after_run(&mut self, &Self::System, &mut Self::State) { }

    fn after_warmup(&mut self, &Self::System, &mut Self::State) { }
}

#[derive(Clone)]
pub struct NullObserver<St,Sy> {
    _phantom_state: PhantomData<St>,
    _phantom_system: PhantomData<Sy>,
}

impl<St,Sy> NullObserver<St,Sy> {
    pub fn new() -> Self {
        NullObserver {
            _phantom_state: PhantomData,
            _phantom_system: PhantomData,
        }
    }
}

impl<St,Sy> Observer for NullObserver<St,Sy> {
    type State = St;
    type System = Sy;

    fn observe(&mut self, _: &Sy, _: &mut St, _: f64) { }
}
