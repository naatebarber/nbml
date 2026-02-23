use std::{any::Any, collections::HashMap, fmt};

pub struct Cache(HashMap<String, Box<dyn Any>>);

impl Cache {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn set<T: Any>(&mut self, key: &str, item: T) {
        self.0.insert(key.to_string(), Box::new(item));
    }

    pub fn get<T: Any>(&self, key: &str) -> &T {
        let item = self.0.get(key).expect(format!("cannot get unknown cache value {key}").as_str());
        item.downcast_ref::<T>().unwrap()
    }

    pub fn get_mut<T: Any>(&mut self, key: &str) -> &mut T {
        self.0.get_mut(key).expect(format!("cannot get_mut unknown cache value {key}").as_str()).downcast_mut::<T>().unwrap()
    }

    pub fn take<T: Any>(&mut self, key: &str) -> T {
        let item = self.0.remove(key).expect(format!("cannot take unknown cache value {key}").as_str());
        *item.downcast::<T>().unwrap()
    }

    pub fn init<T: Any>(&mut self, key: &str, init: T) -> &mut T {
        if self.0.contains_key(key) {
            return self.get_mut(key) 
        }

        self.set(key, init);
        return self.get_mut(key)
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn snapshot(&mut self) -> Cache {
        std::mem::replace(self, Cache::new())
    }

    pub fn restore(&mut self, cache: Cache) {
        *self = cache;
    }
}

impl fmt::Debug for Cache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Cache({})", self.0.len())
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

impl Clone for Cache {
    fn clone(&self) -> Self {
        Self(HashMap::new())
    }
}
