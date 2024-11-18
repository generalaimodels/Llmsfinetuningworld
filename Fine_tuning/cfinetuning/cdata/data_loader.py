from datasets import load_dataset
from cfinetuning.cconfig import DataConfig



class DatasetLoader:
    """Class to load datasets using a data configuration."""

    def __init__(self, config: DataConfig):
        self.config = config

    def load(self):
        """Load a dataset based on the given configuration."""
        try:
            dataset = load_dataset(
                path=self.config.path,
                name=self.config.name,
                data_dir=self.config.data_dir,
                data_files=self.config.data_files,
                split=self.config.split,
                cache_dir=self.config.cache_dir,
                features=self.config.features,
                download_config=self.config.download_config,
                download_mode=self.config.download_mode,
                verification_mode=self.config.verification_mode,
                keep_in_memory=self.config.keep_in_memory,
                save_infos=self.config.save_infos,
                revision=self.config.revision,
                token=self.config.token,
                streaming=self.config.streaming,
                num_proc=self.config.num_proc,
                storage_options=self.config.storage_options,
                trust_remote_code=self.config.trust_remote_code,
                **self.config.config_kwargs
            )
            return dataset
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {str(e)}from  {self.config.path}") from e