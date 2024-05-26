use burn::{
    data::dataset::Dataset,
};
use std::fs;

#[derive(Clone, Debug)]
pub struct ImageDatasetItem {
    pub pixels: Vec<u8>, 
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
    pub fn new(dataset_path: &str) -> Self {
        let entries = fs::read_dir(dataset_path).expect("Dataset folder should exist"); 
        let mut dataset = Vec::new();

        for entry in entries {
            let entry = entry.expect("File should be valid"); 
            let path = entry.path(); 
            if path.is_file() {
                dataset.push(path.to_str().unwrap().to_owned());
            }
        }

        CustomDataset {
            dataset, 
        }
    }
}

#[cfg(test)]
mod test {
    use super::*; 

    #[test]
    pub fn check_length() {
        let dataset = CustomDataset::new("/home/sean/workspace/vae-dataset/conan/processed_faces"); 
        assert_eq!(690, dataset.len());
    }
}