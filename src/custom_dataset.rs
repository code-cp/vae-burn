use burn::data;
use burn::{
    data::dataset::Dataset,
    prelude::*,
};
use std::fs;

#[derive(Clone, Debug)]
pub struct ImageDatasetItem {
    pixels: Vec<u8>, 
}

#[derive(Debug)]
pub struct CustomDataset {
    /// stores the file path of the images 
    dataset: Vec<String>, 
}

impl Dataset<ImageDatasetItem> for CustomDataset {
    fn get(&self, index: usize) -> Option<ImageDatasetItem> {
        if index >= self.len() {
            return None; 
        }

        let filename = self.dataset[index].clone(); 
        let img = image::open(filename).unwrap().into_luma8();

        Some(
            ImageDatasetItem {
                pixels: img.into_raw(), 
            }
        )
    }

    fn len(&self) -> usize {
        self.dataset.len() 
    }
}

impl CustomDataset {
    pub fn new(dataset_path: String) -> Self {
        let entries = fs::read_dir(dataset_path).expect("Dataset folder should exist"); 
        let mut dataset = Vec::new();

        for entry in entries {
            let entry = entry.expect("File should be valid"); 
            let path = entry.path(); 
            if path.is_file() {
                if let Some(filename) = path.file_name() {
                    dataset.push(filename.to_str().unwrap().to_owned()); 
                }
            }
        }

        CustomDataset {
            dataset, 
        }
    }
}