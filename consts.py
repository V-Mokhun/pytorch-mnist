from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True
                 )

MODEL_NAME = "mnist_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
