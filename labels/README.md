# Labels Directory

CSV files for train/val/test splits.

## Required Files

- `final_labels2_train.csv` - Training set
- `final_labels2_val.csv` - Validation set
- `final_labels2_test.csv` - Test set

## CSV Format

```csv
filename,artist
MZ0001.jpg,kue00015
MZ0002.jpg,kue00022
```

Columns:

- `filename`: Image filename (must exist in image directory)
- `artist`: Artist/trademark identifier
