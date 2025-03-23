# face-attendance-check

This project is intended for internal use of PTNK 2024 - 2027 Interdisciplinary Informatics class.

## How to use?

### Capturing

```sh
python src/capture.py
```

### Training

The following script will train the model and export the results to the `trained.pt` file in the current working directory.

```sh
python src/train.py <path_to_dataset> <img_sz>
```

### Detecting

The `trained.pt` file should be prepared inside the current working directory.

```sh
python src/main.py
```
