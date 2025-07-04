# Image-Prompt Evaluator

A sophisticated tool for evaluating how well AI-generated images match their text prompts using OpenAI's CLIP model. This project provides both a command-line interface and a web-based UI for analyzing image-prompt similarity with semantic understanding and contradiction detection.

## Features

- **CLIP-based Similarity Scoring**: Uses OpenAI's CLIP (ViT-B/32) model for robust image-text similarity evaluation
- **Prompt Variation Testing**: Tests multiple variations of prompts for better matching accuracy
- **Semantic Contradiction Detection**: Identifies and penalizes contradictory elements in image-prompt pairs
- **Intelligent Score Normalization**: Converts raw CLIP scores to intuitive 0-100% percentages
- **Keyword Analysis**: Breaks down prompt into keywords and evaluates individual component presence
- **Quality Classification**: Categorizes matches as Excellent, Good, Fair, or Poor
- **Web Interface**: User-friendly Gradio-based web UI for easy image uploads and evaluation
- **Command Line Interface**: Full-featured CLI for batch processing and automation

## Installation

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, but recommended for faster processing)

### Install Dependencies

```bash
pip install -r requirements_evaluator.txt
```

### Manual Installation

If you prefer to install dependencies manually:

```bash
pip install torch>=1.7.1 torchvision>=0.8.2
pip install git+https://github.com/openai/CLIP.git
pip install Pillow>=8.0.0 numpy>=1.19.0 gradio>=3.0.0
```

## Usage

### Web Interface (Recommended)

Launch the Gradio web interface for interactive evaluation:

```bash
python image_prompt_evaluator_ui.py
```

This will start a web server (typically at `http://localhost:7860`) where you can:
- Upload images directly from your browser
- Enter text prompts
- Adjust similarity thresholds
- View detailed evaluation results

### Command Line Interface

For command-line usage and automation:

```bash
python image_prompt_evaluator.py --image path/to/image.jpg --prompt "your text prompt"
```

#### CLI Options

- `--image`: Path to the image file (required)
- `--prompt`: Text prompt to evaluate against (required)
- `--threshold`: Similarity threshold (default: 0.22)
- `--verbose`: Show detailed analysis including all variation scores

#### Examples

Basic evaluation:
```bash
python image_prompt_evaluator.py --image "generated_cat.jpg" --prompt "a fluffy orange cat sitting on a windowsill"
```

Verbose output with custom threshold:
```bash
python image_prompt_evaluator.py --image "landscape.jpg" --prompt "mountain sunset with purple sky" --threshold 0.25 --verbose
```

### Programmatic Usage

```python
from image_prompt_evaluator import ImagePromptEvaluator

# Initialize the evaluator
evaluator = ImagePromptEvaluator()

# Evaluate an image-prompt pair
results = evaluator.evaluate_image("path/to/image.jpg", "your prompt text")

print(f"Match percentage: {results['percentage_match']}")
print(f"Quality: {results['quality']}")
print(f"Feedback: {results['feedback']}")
```

## How It Works

### CLIP Model Integration

The evaluator uses OpenAI's CLIP (Contrastive Language-Image Pre-Training) model, which was trained on millions of image-text pairs and can understand the relationship between visual and textual content.

### Evaluation Process

1. **Image Preprocessing**: Images are preprocessed using CLIP's standard transformations
2. **Prompt Variations**: Multiple variations of the prompt are generated:
   - Original prompt
   - Cleaned version (special characters removed)
   - "A photo of [prompt]"
   - "An image showing [prompt]"
   - "A picture of [prompt]"
3. **Feature Extraction**: Both image and text variations are encoded into high-dimensional feature vectors
4. **Similarity Calculation**: Cosine similarity is computed between image and text features
5. **Semantic Analysis**: The system checks for contradictory elements (e.g., "bear" vs "eagle")
6. **Score Normalization**: Raw CLIP scores are mapped to intuitive 0-100% ranges

### Semantic Contradiction Detection

The system includes sophisticated logic to detect semantic contradictions:

- **Animal Contradictions**: Bear vs bird features, cat vs dog characteristics
- **Object Contradictions**: Car vs airplane, terrestrial vs aerial objects
- **Penalty Application**: Scores are reduced when contradictory elements are detected
- **Threshold-based**: Only applies penalties for significant contradictions

### Score Interpretation

- **80-100%**: Excellent match - Image closely represents the prompt
- **50-80%**: Good match - Image generally matches with minor discrepancies
- **30-50%**: Fair match - Some elements match but improvements needed
- **0-30%**: Poor match - Image doesn't represent the prompt well

## Output Format

### CLI Output Example

```
==================================================
Image-Prompt Evaluation Results
==================================================
Prompt: "a fluffy orange cat sitting on a windowsill"
Overall Match: 87.34% (Excellent)
Raw CLIP Score: 31.24%
Feedback: The image closely matches the prompt.

Keyword Analysis:
  ✓ fluffy - 78.2% (raw: 0.234)
  ✓ orange - 82.1% (raw: 0.267)
  ✓ cat - 91.5% (raw: 0.312)
  ✓ sitting - 65.7% (raw: 0.198)
  ✓ windowsill - 45.3% (raw: 0.156)
==================================================
```

### API Response Structure

```python
{
    "overall_score": 0.287,
    "percentage_match": "87.34%",
    "raw_percentage": "28.70%",
    "quality": "Excellent",
    "feedback": "The image closely matches the prompt.",
    "prompt": "a fluffy orange cat sitting on a windowsill",
    "keyword_analysis": [
        {
            "keyword": "fluffy",
            "present": True,
            "confidence": "78.2%",
            "raw_score": 0.234
        }
        # ... more keywords
    ],
    "meets_threshold": True,
    "detailed_metrics": {
        "raw_score": 0.287,
        "penalized_score": 0.287,
        "average_score": 0.251,
        "normalized_score": 87.34,
        "all_scores": [0.287, 0.251, 0.298, 0.243, 0.267]
    },
    "contradiction_warning": ""
}
```

## Technical Details

### Model Requirements

- **GPU Memory**: ~1GB VRAM for CLIP ViT-B/32 model
- **RAM**: ~2GB system RAM minimum
- **Processing Time**: ~1-3 seconds per evaluation (GPU), ~5-10 seconds (CPU)

### Supported Image Formats

- JPEG, PNG, BMP, TIFF, WebP
- RGB and RGBA color modes
- Automatic resizing to 224x224 for CLIP processing

### Limitations

- Model performance depends on CLIP's training data (pre-2021 internet images)
- May struggle with very recent concepts or extremely specific technical terms
- Evaluation is based on visual similarity, not factual accuracy
- Processing time increases with longer prompts

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU mode
2. **Import Errors**: Ensure all dependencies are installed correctly
3. **Image Loading Errors**: Check image file format and corruption
4. **Low Scores for Good Matches**: Try different prompt variations or check for semantic contradictions

### Performance Optimization

- Use GPU acceleration when available
- Process images in batches for multiple evaluations
- Consider using smaller CLIP models for faster inference

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check the repository for license details.

## Acknowledgments

- OpenAI for the CLIP model
- Gradio team for the web interface framework
- PyTorch and torchvision teams for the ML framework

## Version History

- **v1.0**: Initial release with basic CLIP evaluation
- **v1.1**: Added semantic contradiction detection
- **v1.2**: Improved score normalization and keyword analysis
- **v1.3**: Added Gradio web interface and enhanced CLI