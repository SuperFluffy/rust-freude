pub trait ODE: Clone {
    type State: Clone;

    fn differentiate_into(&mut self, &Self::State, &mut Self::State);

    fn get_state(&self) -> &Self::State;
    fn get_state_mut(&mut self) -> &mut Self::State;

    fn update_state(&mut self, &Self::State);

    fn differentiate(&self) -> Self::State {
        let mut sys = self.clone();
        let mut state = sys.get_state().clone();
        sys.differentiate_into(self.get_state(), &mut state);
        state
    }

    fn differentiate_at_state(&self, state: &Self::State) -> Self::State {
        let mut sys = self.clone();
        let mut newstate = state.clone();
        sys.differentiate_into(state, &mut newstate);
        newstate
    }
}
