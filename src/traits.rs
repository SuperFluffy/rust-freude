pub trait ODE {
    type State: Clone;

    fn differentiate_into(&mut self, &Self::State, &mut Self::State);

    fn differentiate(&mut self, state: &Self::State) -> Self::State {
        let mut derivative = state.clone();
        self.differentiate_into(state, &mut derivative);
        derivative
    }

    fn update_state(&self, state: &mut Self::State, value: &Self::State) {
        state.clone_from(value);
    }
}
