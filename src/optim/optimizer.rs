use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use super::param::ToParams;

pub trait Optimizer: Serialize + Deserialize<'static> + Debug + Clone {
    fn with(self, optimizable: &mut impl ToParams) -> Self;
    fn step(&mut self, optimizable: &mut impl ToParams);
}
