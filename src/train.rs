use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::{decay::WeightDecayConfig, AdamConfig},
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

use crate::data::MnistBatcher;
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

pub fn train<B: AutodiffBackend>(device: B::Device) {
    let config_optim = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config_train = TrainConfig::new(config_optim);
    B::seed(config_train.seed);

    let batcher_train = MnistBatcher::<B>::new(device.clone());
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config_train.batch_size)
        .shuffle(config_train.seed)
        .num_workers(config_train.num_workers)
        .build(MnistDataset::train());
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config_train.batch_size)
        .shuffle(config_train.seed)
        .num_workers(config_train.num_workers)
        .build(MnistDataset::test());

    let model = Model::new();

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
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
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
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
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}