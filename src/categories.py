"""Single source of truth for the photo categories.

Every other file (the inspect script, the DataLoader, the model head, the
Streamlit app) imports CATEGORIES from here. Define your classes in ONE place
so they can never drift out of sync.

IMPORTANT: the order of this list IS the label order the model learns.
Index 0 -> CATEGORIES[0], index 1 -> CATEGORIES[1], and so on. Once you start
training, do NOT reorder this list or insert items in the middle, or a saved
model's outputs will point at the wrong names. Append-only after training.
"""

# Your 7 predefined categories.
# Each name here must match a folder under data/raw/<name>/ exactly.
CATEGORIES = [
    "people",       # photos of other people (friends, family, groups)
    "documents",    # documents AND screenshots (receipts, forms, phone screenshots)
    "landscape",    # outdoor / nature / scenery
    "adi",          # you, Adi — photos that are mainly of yourself
    "home",         # interiors, rooms, furniture, your living space
    "food",         # meals, drinks, snacks
    "lab",          # devices, work gear, equipment, electronics
]

# Convenience lookups (built once, used everywhere).
NUM_CLASSES = len(CATEGORIES)
NAME_TO_INDEX = {name: i for i, name in enumerate(CATEGORIES)}
INDEX_TO_NAME = {i: name for i, name in enumerate(CATEGORIES)}
