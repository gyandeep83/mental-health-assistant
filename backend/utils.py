import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent

def get_db_path(db_name):
    """Get the absolute path to a database file"""
    return os.path.join(BASE_DIR, db_name)