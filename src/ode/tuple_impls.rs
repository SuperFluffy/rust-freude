use super::Ode;

use tuple::*;

pub type UT1<T> = T1<T>;
pub type UT2<T> = T2<T, T>;
pub type UT3<T> = T3<T, T, T>;
pub type UT4<T> = T4<T, T, T, T>;
pub type UT5<T> = T5<T, T, T, T, T>;
pub type UT6<T> = T6<T, T, T, T, T, T>;
pub type UT7<T> = T7<T, T, T, T, T, T, T>;
pub type UT8<T> = T8<T, T, T, T, T, T, T, T>;
pub type UT9<T> = T9<T, T, T, T, T, T, T, T, T>;
pub type UT10<T> = T10<T, T, T, T, T, T, T, T, T, T>;
pub type UT11<T> = T11<T, T, T, T, T, T, T, T, T, T, T>;
pub type UT12<T> = T12<T, T, T, T, T, T, T, T, T, T, T, T>;

macro_rules! impl_ode_for_tuples {
    ( $tup:ty ) => {
        impl Ode for Box<Fn($tup) -> $tup>
        {
            type State = $tup;

            fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
                derivative.clone_from(&self(*state));
            }
        }

        impl<'a> Ode for &'a Fn($tup) -> $tup
        {
            type State = $tup;

            fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
                derivative.clone_from(&self(*state));
            }
        }

        impl Ode for Box<FnMut($tup) -> $tup>
        {
            type State = $tup;

            fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
                derivative.clone_from(&self(*state));
            }
        }

        impl<'a> Ode for &'a mut FnMut($tup) -> $tup
        {
            type State = $tup;

            fn differentiate_into(&mut self, state: &Self::State, derivative: &mut Self::State) {
                derivative.clone_from(&self(*state));
            }
        }
    };

    ( $( $tup:ty ),+ ) => {
        $(
            impl_ode_for_tuples!($tup);
        )+
    };
}

impl_ode_for_tuples!(
    UT1<f64>,
    UT2<f64>,
    UT3<f64>,
    UT4<f64>,
    UT5<f64>,
    UT6<f64>,
    UT7<f64>,
    UT8<f64>,
    UT9<f64>,
    UT10<f64>,
    UT11<f64>,
    UT12<f64>
);
