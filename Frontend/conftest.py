import sys
from pathlib import Path

# app.py imports `mongo_store` and `from ML import Multivariable`, both of which
# live at the repo root, one level up from Frontend/. Make sure that's on
# sys.path regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
