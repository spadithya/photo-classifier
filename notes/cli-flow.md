# CLI flow (the algorithm)

`organize.py` — sort a local folder of photos into per-category subfolders on
disk. The **workhorse**: streams from disk one image at a time (constant memory),
so it scales to thousands of files. Reuses the same `predict()` as the app.

---

## Inputs (command-line args, via argparse)

```
--input      folder of photos to organize            (required)
--output     where sorted subfolders are created     (required)
--mode       copy | move        (default: copy — keeps originals)
--threshold  float              (default: 0.7 — below this → unsure/)
--model      path to weights    (default: models/photo_classifier.pt)
```

---

## Flow

```
parse args
   │
   ▼
validate: input folder exists?  model file exists?      (else print + exit)
   │
   ▼
gather_images(input)                # rglob("*") → every image file, recursively
   │                                # (skip non-image extensions)
   ▼
load_model(model_path)              # build architecture + load weights + eval()
build_eval_transform()              # build ONCE, reuse for every image
   │
   ▼
for each path in tqdm(paths):       # progress bar + ETA, one image at a time
   │
   ├─ image = load_image(path)      # open → RGB (None if unreadable)
   │
   ├─ if image is None:             # don't silently lose it
   │      place(path, output/"unreadable", mode)
   │      counts["unreadable"] += 1
   │      continue
   │
   ├─ (name, prob) = predict(model, image, device, transform)[0]   # top guess
   │
   ├─ folder = name  if prob >= threshold  else "unsure"           # threshold gate
   │
   ├─ place(path, output/folder, mode)      # copy or move, collision-safe
   └─ counts[folder] += 1
   │
   ▼
print summary (counts per folder)
```

---

## Key sub-steps

### `gather_images(input_dir)`
- `input_dir.rglob("*")` → **recursive** walk (nested subfolders included).
- Keep files whose suffix is a known image extension. Sorted for stable order.

### `place(src, folder, mode)`
```
folder.mkdir(parents=True, exist_ok=True)   # create category dir if needed
dest = unique_dest(folder, src.name)         # avoid overwriting same-named files
if mode == "copy": shutil.copy2(src, dest)   # copy2 keeps timestamps/metadata
else:              shutil.move(src, dest)     # relocate (destructive)
```

### `unique_dest(folder, name)`
- If `folder/name` is free → use it.
- Else append `_1`, `_2`, … (`IMG_1234.jpg` → `IMG_1234_1.jpg`) so two source
  files with the same name don't clobber each other.

---

## Design choices (why it's built this way)

- **Constant memory:** one image decoded at a time → 50 photos and 8,000 photos
  use the same RAM. (Contrast with the app, which holds everything in memory —
  fine for a small demo, not a full library.)
- **Same brain as the app:** imports `load_model` / `predict` from
  `src/inference.py`, so CLI and web app predict identically.
- **Transform built once**, outside the loop — not rebuilt per image.
- **Nothing lost:** unreadable files → `unreadable/`; low-confidence →
  `unsure/`. Everything ends up somewhere.
- **`copy` is safe / default**; `move` is destructive — only use once trusted.

---

## Example

```bash
python organize.py \
    --input  "C:\Users\Shady\Pictures\unsorted" \
    --output "C:\Users\Shady\Pictures\sorted" \
    --mode   copy \
    --threshold 0.7
```

Produces:
```
sorted/
├── people/      ├── documents/   ├── landscape/
├── adi/         ├── food/        ├── lab/
├── unsure/      └── unreadable/   (only if any landed there)
```
