# NutriKids AI: Empowering Early Pediatric Malnutrition Diagnosis for a Brighter Future

Welcome to **NutriKids AI** – an innovative, AI-driven solution designed to revolutionize the early detection of pediatric malnutrition. In pediatric healthcare, early and accurate diagnosis is critical for ensuring timely intervention and improving long-term outcomes. NutriKids AI leverages state-of-the-art machine learning techniques and natural language processing to analyze clinical data, helping healthcare professionals quickly identify children at risk of malnutrition.

<img src="nutrikid_ai.png" alt="WordCloud" style="height: 50%;">

## Console Interface

NutriKids AI provides a powerful command-line console interface that enables healthcare professionals and researchers to interact with the system efficiently. The console offers a comprehensive set of commands organized into several functional categories:

### File Operations
Navigate and manage your data files with ease using standard commands:
- `ls`: List directory contents with clear formatting
- `pwd`: Print current working directory with path validation
- `cd`: Change directory with robust error handling
- `cat`: Display file contents with line numbers and pagination support

### Model Training
Train and fine-tune various machine learning models on your clinical data:
- `tunetextcnn`: Fine-tune TextCNN models for natural language processing tasks
- `traintextcnn`: Train new TextCNN models for text classification
- `traintabpfn`: Train tabular probabilistic functional networks on structured data
- `tunexgb`: Fine-tune XGBoost models for optimal performance
- `trainxgb`: Train new XGBoost models for regression or classification tasks
- `llmtrain`: Train or fine-tune large language models for specialized medical text analysis

### Model Evaluation
Assess model performance with comprehensive evaluation tools:
- `evaluatetextcnn`: Evaluate TextCNN model performance metrics
- `evaluatetabpfn`: Evaluate TabPFN model accuracy and effectiveness
- `evaluatexgb`: Generate detailed performance reports for XGBoost models

### Model Inference
Deploy trained models to make predictions on new data:
- `textcnnpredict`: Make predictions using trained TextCNN models
- `predicttabpfn`: Generate predictions from TabPFN models
- `predictxgb`: Use XGBoost models for inference on new data
- `llminference`: Run inference using fine-tuned large language models
- `ollamaserve`: Deploy Ollama models as inference services

### System Operations
Manage your console environment:
- `clear`: Clear the terminal screen for better visibility
- `quit`: Exit the application safely
- `help`: Access the detailed help system with command documentation

## Getting Started

To begin using NutriKids AI, simply launch the console application and use the `help` command to explore available functionality. For detailed information about specific commands, use `help <command>`.


## Development Status

⚠️ **This project is currently under active development** ⚠️

Please note that this is a research project in development. Features, APIs, and documentation may change significantly. Use with caution in production environments.


**NutriKids AI** – Because every child deserves a healthy start in life.