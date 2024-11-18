from typing import Optional, Union, Sequence, Mapping, Dict, Any,List
from datasets import  Split
from datasets.features import Features
from datasets.download.download_config import DownloadConfig
from datasets.download.download_manager import DownloadMode
from datasets.download.download_config import DownloadConfig
from datasets.utils.info_utils import VerificationMode
from datasets.utils.version import Version
from dataclasses import dataclass,field
from typing import Optional


class DataConfig:
    """Class to hold configuration for dataset loading."""

    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, Sequence[str], Mapping[str, Union[str, Sequence[str]]]]] = None,
        split: Optional[Union[str, Split]] = None,
        cache_dir: Optional[str] = None,
        features: Optional[Features] = None,
        download_config: Optional[DownloadConfig] = None,
        download_mode: Optional[Union[DownloadMode, str]] = None,
        verification_mode: Optional[Union[VerificationMode, str]] = None,
        keep_in_memory: Optional[bool] = None,
        save_infos: bool = False,
        revision: Optional[Union[str, Version]] = None,
        token: Optional[Union[bool, str]] = None,
        streaming: bool = False,
        num_proc: Optional[int] = None,
        storage_options: Optional[Dict] = None,
        trust_remote_code: Optional[bool] = None,
        **config_kwargs: Any
    ):
        self.path = path
        self.name = name
        self.data_dir = data_dir
        self.data_files = data_files
        self.split = split
        self.cache_dir = cache_dir
        self.features = features
        self.download_config = download_config
        self.download_mode = download_mode
        self.verification_mode = verification_mode
        self.keep_in_memory = keep_in_memory
        self.save_infos = save_infos
        self.revision = revision
        self.token = token
        self.streaming = streaming
        self.num_proc = num_proc
        self.storage_options = storage_options
        self.trust_remote_code = trust_remote_code
        self.config_kwargs = config_kwargs







class ModelConfig:
    """Configuration class for managing model parameters."""
    
    def __init__(self, pretrained_model_name_or_path: str, *inputs: Any, **kwargs: Dict[str, Any]) -> None:
        """
        Initialize ModelConfig with model details.

        :param pretrained_model_name_or_path: The name or path of the pretrained model.
        :param inputs: Additional positional arguments.
        :param kwargs: Additional keyword arguments for flexibility.
        """
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.inputs = inputs
        self.kwargs = kwargs



@dataclass
class DatasetConfig:
    split: str
    train_ratio: float = 0.8
    eval_ratio: float = 0.1
    test_ratio: float = 0.1
    input_columns: List[str] = None
    target_column: str = None
    max_length: int = 512
    batch_size: int = 32



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    config_overrides: Optional[str] = field(default=None)
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    token: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=False)
    torch_dtype: Optional[str] = field(default=None)
    low_cpu_mem_usage: bool = field(default=False)

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    streaming: bool = field(default=False)
    block_size: Optional[int] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    preprocessing_num_workers: Optional[int] = field(default=None)
    keep_linebreaks: bool = field(default=True)
    input_column_names: List[str] = field(default_factory=list)  # List of input column names
    target_column_name: Optional[str] = field(default=None)       # Name of the target column for labels

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json, or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json, or a txt file."

        if not self.input_column_names:
            raise ValueError("`input_column_names` must be a non-empty list specifying the input columns.")
        if self.target_column_name is None:
            raise ValueError("`target_column_name` must be specified.")
        
@dataclass
class CTrainingArguments:
    """Training arguments pertaining to the training loop itself."""

    output_dir: str = field(
        metadata={
            "help": (
                "The output directory where the model predictions, checkpoints, and other output artifacts will be "
                "written. This path will be used to save all outputs generated during training and evaluation, such "
                "as trained model files and evaluation logs."
            )
        }
    )
    do_train: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to run the training loop. Set this to `True` to enable the training phase, where the model "
                "learns from the training dataset."
            )
        },
    )
    do_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to run evaluation on the validation set. Set to `True` to validate model performance during "
                "or after training."
            )
        },
    )
    do_predict: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to run inference on the test/inference set. Set this to `True` to generate predictions after "
                "training is complete."
            )
        },
    )
    seed: Optional[int] = field(
        default=42,
        metadata={
            "help": (
                "The random seed value to ensure reproducibility of results. Setting a fixed seed helps in making "
                "training results consistent across multiple runs."
            )
        },
    )
    per_device_train_batch_size: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "The number of training samples per batch per device (GPU/TPU/CPU). Adjusting this value impacts the "
                "memory usage and training time."
            )
        },
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "The number of evaluation samples per batch per device (GPU/TPU/CPU). A larger batch size can speed "
                "up evaluation but may require more memory."
            )
        },
    )
    weight_decay: Optional[float] = field(
        default=0.0,
        metadata={
            "help": (
                "The weight decay (L2 regularization) factor applied to the model's weights, excluding biases and "
                "LayerNorm weights. A non-zero value helps prevent overfitting by penalizing large weights."
            )
        },
    )
    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={
            "help": (
                "The initial learning rate for the optimizer. This value determines the step size at each iteration "
                "while moving toward a minimum of the loss function."
            )
        },
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "The number of gradient accumulation steps before a backward pass. This effectively increases the "
                "batch size without using additional memory, as gradients are accumulated over multiple steps."
            )
        },
    )
    max_steps: Optional[int] = field(
        default=-1,
        metadata={
            "help": (
                "The maximum number of training steps to execute. If set to a positive value, this overrides the "
                "number of epochs and stops training once the specified steps are completed."
            )
        },
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={
            "help": (
                "The type of learning rate scheduler to use during training, such as 'linear', 'cosine', or 'constant'. "
                "This controls how the learning rate changes over time."
            )
        },
    )
    warmup_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "The number of warmup steps to perform at the start of training. During warmup, the learning rate "
                "linearly increases from zero to the set learning rate, which helps stabilize the training process."
            )
        },
    )
    eval_strategy: Optional[str] = field(
        default="no",
        metadata={
            "help": (
                "The evaluation strategy during training, specifying when to perform evaluations. Options are: "
                "'no' (never), 'steps' (after a certain number of steps), or 'epoch' (at the end of each epoch)."
            )
        },
    )
    eval_steps: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "The frequency of evaluations when `eval_strategy` is set to 'steps'. This specifies the number of "
                "update steps between two consecutive evaluations."
            )
        },
    )
    eval_metric: Optional[str] = field(
        default="accuracy",
        metadata={
            "help": (
                "The evaluation metric used to assess the model's performance on the validation set, such as 'accuracy', "
                "'f1', or 'loss'."
            )
        },
    )
    keep_checkpoint_max: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "The maximum number of best checkpoint files to retain during training. Checkpoints store the model's "
                "state and can be used to resume or analyze training later."
            )
        },
    )
    early_stopping_patience: Optional[int] = field(
        default=10,
        metadata={
            "help": (
                "The patience for early stopping, specifying how many consecutive evaluations with no improvement are "
                "allowed before stopping the training process."
            )
        },
    )
    early_stopping_threshold: Optional[float] = field(
        default=0.0,
        metadata={
            "help": (
                "The minimum amount by which the evaluation metric must improve to reset early stopping. Smaller "
                "thresholds make early stopping more sensitive to changes in the metric."
            )
        },
    )
    pad_to_max_length :Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "pad_to_max_length"
            )
        },
    )
    num_train_epochs :Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "Number of training epochs"
            )
        },
    )
    is_regression :Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Number of training epochs"
            )
        },
    )
    
    
    
