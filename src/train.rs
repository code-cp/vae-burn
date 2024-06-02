use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::{decay::WeightDecayConfig, AdamConfig},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            CudaMetric, LearningRateMetric, LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

#[cfg(feature = "custom")]
use crate::custom_batcher::ImageDatasetBatcher;
#[cfg(feature = "mnist")]
use crate::mnist_batcher::ImageDatasetBatcher;

use crate::model::Model;

#[derive(Config)]
pub struct TrainConfig {
    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

static ARTIFACT_DIR: &str = "./artifacts";

pub fn train<B: AutodiffBackend>(device: &B::Device) {
    let config_optim = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));

    // let num_epochs = 10000;
    // let batch_size = 64;
    let num_epochs = 1;
    let batch_size = 1;

    let config_train = TrainConfig::new(config_optim)
        .with_batch_size(batch_size)
        .with_num_epochs(num_epochs);
    B::seed(config_train.seed);

    let batcher_train = ImageDatasetBatcher::<B>::new(device.clone());
    let batcher_valid = ImageDatasetBatcher::<B::InnerBackend>::new(device.clone());

    #[cfg(feature = "custom")]
    let train_set_path = "/home/sean/workspace/vae-dataset/conan/faces_train";
    #[cfg(feature = "custom")]
    let test_set_path = "/home/sean/workspace/vae-dataset/conan/faces_test";
    #[cfg(feature = "custom")]
    let train_dataset = CustomDataset::new(train_set_path);
    #[cfg(feature = "custom")]
    let test_dataset = CustomDataset::new(test_set_path);

    #[cfg(feature = "mnist")]
    let train_dataset = MnistDataset::train();
    #[cfg(feature = "mnist")]
    let test_dataset = MnistDataset::test();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config_train.batch_size)
        .shuffle(config_train.seed)
        .num_workers(config_train.num_workers)
        .build(train_dataset);
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config_train.batch_size)
        .shuffle(config_train.seed)
        .num_workers(config_train.num_workers)
        .build(test_dataset);

    let model = Model::new(device);

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 20 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config_train.num_epochs)
        .summary()
        .build(model, config_train.optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config_train
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            // &NoStdTrainingRecorder::new(),
            &CompactRecorder::new(),
        )
        .expect("Failed to save trained model");
}
