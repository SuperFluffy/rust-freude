use std::marker::PhantomData;

pub trait Observer {
    type System;
    // The observer should be allowed to affect the observable,
    // e.g. in the case of reading the Lyapunov exponents, or resetting
    // a phase to the range [0,2π)
    fn observe(&mut self, &mut Self::System, f64);

    fn finalize(&mut self, &mut Self::System);
}

#[derive(Clone)]
pub struct NullObserver<T> {
    _phantom: PhantomData<T>,
}

impl<T> NullObserver<T> {
    pub fn new() -> Self {
        NullObserver { _phantom: PhantomData }
    }
}

impl<T> Observer for NullObserver<T> {
    type System = T;

    fn observe(&mut self, _: &mut T, _: f64) { }

    fn finalize(&mut self, _: &mut T) { }
}
