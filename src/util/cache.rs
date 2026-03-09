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
        if let Some(current) = self.map.get_mut(key) {
            *current += &value;
        } else {
            self.set(key, value)
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
        self.map.get_mut(idx).unwrap()
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}
