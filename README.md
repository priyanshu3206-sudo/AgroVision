# AgroVision — Integrated Streamlit + PyTorch prototype

## Setup (Windows / VS Code)

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501. Model weights are included in `models/`.

## Architecture

- `app.py`: Streamlit frontend and upload workflow
- `src/predict.py`: MobileNetV2 inference (loads `.pth`, normalizes input, maps classes)
- `disease_info.py`: educational condition descriptions
- `models/`: Member 2's checkpoint and class labels
- `results/`: Member 2's evaluation figures

The checkpoint metadata specifies 224×224 RGB input, ImageNet normalization, 10 class names, and 96.55% reported test accuracy. The evaluation dataset is **not** included, so this project cannot independently reproduce that score from this ZIP alone. The training and evaluation figures are artifacts provided by Member 2.

## Integration notes

`predict_image(PIL.Image)` returns `{'class_name': str, 'confidence': float, 'probabilities': [...]}`. Inference runs locally in the Streamlit Python process; no REST API is needed.

## Safety

Predictions are for academic demonstration and are not confirmed plant diagnoses. General precautions are educational only.
