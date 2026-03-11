use std::{
    collections::HashMap,
    ops::{Index, IndexMut},
};

use serde::{Deserialize, Serialize};

use crate::Tensor;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Cache {
    map: HashMap<String, Tensor>,
}

impl Cache {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: Tensor) {
        self.map.insert(key.into(), value);
    }

    pub fn accumulate(&mut self, key: &str, value: Tensor) {
        let current = self.map.entry(key.into()).or_insert(Tensor::zeros(0));

        if current.shape() == &[0] {
            self.set(key, value)
        } else {
            *current += &value;
        }
    }

    pub fn clear(&mut self) -> HashMap<String, Tensor> {
        std::mem::replace(&mut self.map, HashMap::new())
    }
}

impl Index<&'static str> for Cache {
    type Output = Tensor;
    fn index(&self, idx: &'static str) -> &Tensor {
        &self.map.get(idx).unwrap()
    }
}

impl IndexMut<&'static str> for Cache {
    fn index_mut(&mut self, idx: &'static str) -> &mut Tensor {
        self.map.entry(idx.into()).or_insert(Tensor::zeros(0))
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}
