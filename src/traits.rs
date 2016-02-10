pub trait ODE {
    type State;

    fn get_state(&self) -> &Self::State;

    fn differentiate(&self, &Self::State) -> Self::State;
    fn differentiate_into(&self, &Self::State, &mut Self::State);
    fn update_state(&mut self, &Self::State);
}

pub trait Observer<T> {
    // The observer should be allowed to affect the observable,
    // e.g. in the case of reading the Lyapunov exponents, or resetting
    // a phase to the range [0,2π)
    fn observe(&mut self, &mut T);
}
