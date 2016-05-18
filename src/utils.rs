// Carbon-copied from rust-ndarray, until they are properly included.

use ndarray::{
    ArrayBase,
    ArrayView,
    DataMut,
    Dimension,
};
use std::cmp;

#[derive(Copy, Clone, Debug)]
pub enum LockstepError {
    NotSameLayout,
    NotSameShape,
}

macro_rules! define_fold {
    ($name:ident, $($arg:ident),+) => {
#[allow(non_snake_case)]
pub fn $name<A, $($arg,)+ Acc, Dim, Func>(a: ArrayView<A, Dim>,
    $($arg: ArrayView<$arg, Dim>,)+ mut init: Acc, mut f: Func)
    -> Result<Acc, LockstepError>
    where Dim: Dimension,
          Func: FnMut(Acc, &A, $(&$arg),+) -> Acc
{
    if $(a.shape() != $arg.shape() ||)+ false {
        return Err(LockstepError::NotSameShape);
    }
    if let Some(a_s) = a.as_slice() {
        let len = a_s.len();
        $(
            // extract the slice
            let $arg = if let Some(s) = $arg.as_slice() {
                s
            } else {
                return Err(LockstepError::NotSameLayout);
            };
            let len = cmp::min(len, $arg.len());
        )+
        let a_s = &a_s[..len];
        $(
            let $arg = &$arg[..len];
        )+
        for i in 0..len {
            init = f(init, &a_s[i], $(&$arg[i]),+);
        }

        return Ok(init);
    }
    // otherwise
    Err(LockstepError::NotSameLayout)
}
    }
}

macro_rules! define_zip {
    ($name:ident, $($arg:ident),+) => {
#[allow(non_snake_case)]
pub fn $name<A, $($arg,)+ Data, Dim, Func>(a: &mut ArrayBase<Data, Dim>,
    $($arg: ArrayView<$arg, Dim>,)+ mut f: Func)
    -> Result<(), LockstepError>
    where Data: DataMut<Elem=A>,
          Dim: Dimension,
          Func: FnMut(&mut A, $(&$arg),+)
{
    if $(a.shape() != $arg.shape() ||)+ false {
        return Err(LockstepError::NotSameShape);
    }
    if let Some(a_s) = a.as_slice_mut() {
        let len = a_s.len();
        $(
            // extract the slice
            let $arg = if let Some(s) = $arg.as_slice() {
                s
            } else {
                return Err(LockstepError::NotSameLayout);
            };
            let len = cmp::min(len, $arg.len());
        )+
        let a_s = &mut a_s[..len];
        $(
            let $arg = &$arg[..len];
        )+
        for i in 0..len {
            f(&mut a_s[i], $(&$arg[i]),+)
        }
        return Ok(());
    }
    // otherwise
    Err(LockstepError::NotSameLayout)
}
    }
}

define_fold!(fold_with_1,  B);
define_fold!(fold_with_2,  B, C);
define_fold!(fold_with_3,  B, C, D);
define_fold!(fold_with_4,  B, C, D, E);
define_fold!(fold_with_5,  B, C, D, E, F);
define_fold!(fold_with_6,  B, C, D, E, F, G);
define_fold!(fold_with_7,  B, C, D, E, F, G, H);
define_fold!(fold_with_8,  B, C, D, E, F, G, H, I);
define_fold!(fold_with_9,  B, C, D, E, F, G, H, I, J);
define_fold!(fold_with_10, B, C, D, E, F, G, H, I, J, K);
define_fold!(fold_with_11, B, C, D, E, F, G, H, I, J, K, L);
define_fold!(fold_with_12, B, C, D, E, F, G, H, I, J, K, L, M);

define_zip!(zip_mut_with_1,  B);
define_zip!(zip_mut_with_2,  B, C);
define_zip!(zip_mut_with_3,  B, C, D);
define_zip!(zip_mut_with_4,  B, C, D, E);
define_zip!(zip_mut_with_5,  B, C, D, E, F);
define_zip!(zip_mut_with_6,  B, C, D, E, F, G);
define_zip!(zip_mut_with_7,  B, C, D, E, F, G, H);
define_zip!(zip_mut_with_8,  B, C, D, E, F, G, H, I);
define_zip!(zip_mut_with_9,  B, C, D, E, F, G, H, I, J);
define_zip!(zip_mut_with_10, B, C, D, E, F, G, H, I, J, K);
define_zip!(zip_mut_with_11, B, C, D, E, F, G, H, I, J, K, L);
define_zip!(zip_mut_with_12, B, C, D, E, F, G, H, I, J, K, L, M);
