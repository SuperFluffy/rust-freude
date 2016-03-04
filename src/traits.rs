use std::any::Any;

pub trait ODE {
    type State: Clone;

    fn as_any(&self) -> &Any;

    fn differentiate_into(&mut self, &Self::State, &mut Self::State);

    fn get_state(&self) -> &Self::State;
    fn get_state_mut(&mut self) -> &mut Self::State;

    fn update_state(&mut self, &Self::State);

    fn differentiate(&mut self, state: &Self::State) -> Self::State {
        let mut newstate = state.clone();
        self.differentiate_into(state, &mut newstate);
        newstate
    }
}

pub trait Observer<T: ?Sized> {
    // The observer should be allowed to affect the observable,
    // e.g. in the case of reading the Lyapunov exponents, or resetting
    // a phase to the range [0,2π)
    fn observe(&mut self, &mut T, usize);
}
