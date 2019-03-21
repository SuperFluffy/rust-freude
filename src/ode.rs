#[cfg(feature = "tuple")]
mod tuple_impls;

#[cfg(feature = "tuple")]
pub use self::tuple_impls::*;

pub trait Ode {
    type State: Clone;

    fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State);

    fn differentiate(&mut self, state: &Self::State) -> Self::State {
        let mut derivative = state.clone();
        self.differentiate_into(state, &mut derivative);
        derivative
    }

    fn update_state(&self, state: &mut Self::State, value: &Self::State) {
        state.clone_from(value);
    }
}
