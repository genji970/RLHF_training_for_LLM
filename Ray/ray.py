import ray

from ray.train import (
    ScalingConfig,
)

from ray.train.torch import (
    TorchTrainer,
)

from config import AppConfig

from train.train import (
    DistributedTrainLoop,
)

from inference.inference import (
    VLLMInferenceWorker,
)


class RayOrchestrator:

    def __init__(
        self,
        config: AppConfig,
    ):

        self.config = config

        self.inference_worker = None

    def run(self):

        self._initialize_ray()

        self._check_resources()

        try:

            self._start_inference_worker()

            result = (
                self._start_training()
            )

            return result

        finally:

            self._shutdown_workers()

    def _initialize_ray(self):

        if self.config.ray.address:

            ray.init(
                address=(
                    self.config
                    .ray
                    .address
                )
            )

        else:

            ray.init()

        print(
            "Ray cluster resources:"
        )

        print(
            ray.cluster_resources()
        )

    def _check_resources(self):

        resources = (
            ray.cluster_resources()
        )

        available_gpus = int(
            resources.get(
                "GPU",
                0,
            )
        )

        required_gpus = (
            self.config
            .train
            .train_gpus

            +

            self.config
            .inference
            .inference_gpus
        )

        if (
            available_gpus
            < required_gpus
        ):
            raise RuntimeError(
                f"Need {required_gpus} GPUs, "
                f"but Ray cluster has "
                f"{available_gpus}."
            )

    def _start_inference_worker(
        self,
    ):

        RemoteInferenceWorker = (
            ray.remote(
                VLLMInferenceWorker
            )
        )

        self.inference_worker = (
            RemoteInferenceWorker
            .options(
                num_gpus=(
                    self.config
                    .inference
                    .inference_gpus
                ),

                num_cpus=(
                    self.config
                    .ray
                    .inference_cpus
                ),
            )
            .remote(
                model_config=(
                    self.config.model
                ),

                inference_config=(
                    self.config
                    .inference
                ),
            )
        )

        status = ray.get(
            self.inference_worker
            .ready
            .remote()
        )

        print(
            "Inference worker:",
            status,
        )

    def _start_training(self):

        scaling_config = (
            ScalingConfig(
                num_workers=(
                    self.config
                    .train
                    .train_gpus
                ),

                use_gpu=True,

                resources_per_worker={
                    "GPU": 1,

                    "CPU":
                        self.config
                        .ray
                        .cpus_per_train_worker,
                },

                placement_strategy=(
                    self.config
                    .ray
                    .placement_strategy
                ),
            )
        )

        trainer = TorchTrainer(

            train_loop_per_worker=(
                DistributedTrainLoop()
            ),

            train_loop_config={
                "config":
                    self.config,

                "inference_worker":
                    self.inference_worker,
            },

            scaling_config=(
                scaling_config
            ),
        )

        result = trainer.fit()

        print(
            "Training finished."
        )

        print(
            result.metrics
        )

        return result

    def _shutdown_workers(self):

        if (
            self.inference_worker
            is not None
        ):

            ray.kill(
                self.inference_worker
            )