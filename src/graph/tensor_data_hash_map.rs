use std::collections::HashMap;

use objc2::rc::Retained;
use objc2_foundation::NSDictionary;

use crate::Tensor;
use crate::TensorData;

pub type RetainedTensorDataHashMap = HashMap<Retained<Tensor>, Retained<TensorData>>;
pub type TensorDataHashMap<'a> = HashMap<&'a Tensor, &'a TensorData>;
pub type TensorDataDictionary = NSDictionary<Tensor, TensorData>;

pub trait ToTensorDataDictionary {
    fn to_dictionary(&self) -> Retained<TensorDataDictionary>;
}

impl<'a> ToTensorDataDictionary for HashMap<&'a Tensor, &'a TensorData> {
    fn to_dictionary(&self) -> Retained<TensorDataDictionary> {
        let keys: Vec<&Tensor> = self.keys().copied().collect();
        let values: Vec<&TensorData> = self.values().copied().collect();
        NSDictionary::from_slices(&keys, &values)
    }
}

pub trait TensorDataDictionaryExt {
    fn to_hashmap(&self) -> RetainedTensorDataHashMap;
}

impl TensorDataDictionaryExt for TensorDataDictionary {
    fn to_hashmap(&self) -> RetainedTensorDataHashMap {
        let (keys, values) = self.to_vecs();
        HashMap::from_iter(keys.into_iter().zip(values))
    }
}
