pub trait ODE {
    type State;

    fn get_state(&self) -> &Self::State;

    fn differentiate(&self, &Self::State) -> Self::State;
    fn differentiate_into(&self, &Self::State, &mut Self::State);
    fn update_state(&mut self, &Self::State);
}
