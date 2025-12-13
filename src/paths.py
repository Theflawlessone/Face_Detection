from pathlib import Path

THIS_FILE = Path(__file__).resolve()
BASE_DIR = THIS_FILE.parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CROPPED_DIR = DATA_DIR / "cropped"
PICKLE_DIR = DATA_DIR / "pickle_files"

# Models directory
MODEL_DIR = BASE_DIR / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CROPPED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("CROPPED_DIR:", CROPPED_DIR)
    print("MODEL_DIR:", MODEL_DIR)