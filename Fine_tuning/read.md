# Advanced Fine-Tuning Framework

This repository provides a streamlined approach to fine-tuning models using a YAML-based configuration system. With minimal setup, you can fine-tune models, preprocess datasets, and visualize training progress effortlessly. Built with PyTorch, Hugging Face Transformers, and Plotly for robust performance and user-friendly visualization.

## Features

- YAML-based configuration for easy customization.
- Dynamic dataset processing with prompt-based templates.
- Integration with Hugging Face Transformers for seamless model handling.
- Advanced training features like dynamic padding and evaluation callbacks.
- Automatic plotting of training progress using Plotly.

## Getting Started

### Prerequisites

Ensure you have Python installed and the necessary libraries:
- `transformers`
- `datasets`
- `evaluate`
- `plotly`
- `pyyaml`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration

Update the YAML file with your dataset and model details. Example:

```yaml
data_config:
  train_file: "path/to/train.csv"
  validation_file: "path/to/validation.csv"
  test_file: "path/to/test.csv"
  text_column: "text"
  label_column: "label"

model_config:
  model_name_or_path: "bert-base-uncased"
  num_labels: 2

prompt_template:
  template: "Classify the following text: {text}\nLabel:"

training_args:
  output_dir: "./results"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  logging_dir: "./logs"
  evaluation_strategy: "steps"
  eval_steps: 500
  save_steps: 1000
  load_best_model_at_end: true

hub_repo_name: "your-username/your-model-name"
```

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/generalaimodels/Llmsfinetuningworld.git
   cd Llmsfinetuningworld
   ```



3. Visualize training progress in `training_progress.html`.

### Output
- **Best model weights**: Saved automatically in `best_weights`.
- **Training logs**: Available in `logs`.
- **Evaluation results**: Stored in `output`.

## Visualization

Training progress is plotted dynamically using Plotly. Metrics like training loss, evaluation loss, and accuracy are visualized for detailed insights.

## License

This project is licensed under the MIT License. Feel free to contribute and share!

---

Happy fine-tuning! 🚀