# Changelog
## [Past] - Custom Train Pipeline with Word Data
### Changed


## [27-07-2025] - Custom Inference Pipeline with Word Data

### 1. Changed `inference.py`
- `main()` function:  
    - This inserts the `word_data` directly from folder of json files.  The `words_dir` need a path to words folder.
    ```python
    # main() function, line 1011
    words_dir = r"D:\MyWorking\dataset\FinTabNet.c\FinTabNet.c-Structure\words"
    words_path = os.path.join(words_dir, img_file.replace(".jpg", "_words.json"))
    with open(words_path, 'r') as f:
        word_data = json.load(f)
    ```  
    - `word_data` was run through the `recognize()` function of class `TableExtractionPipeline`
    ```python
    # main() function, line 1052
    if args.mode == 'recognize':
        extracted_table = pipe.recognize(img, word_data, tokens, out_objects=args.objects, out_cells=args.csv, out_html=args.html, out_csv=args.csv)
    ```
- `recognize()`function, line 859: add `word_data=None` into the parameters
    ```python
    # recognize() function, line 857
    def recognize(self, img, word_data, tokens=None, out_objects=False, out_cells=False, out_html=False, out_csv=False):
    ```
    ```python
    outputs = self.str_model(img, word_data)
    ```
- `build_model()` function, line 51: replaced `DefaultPredictor` with `CustomPredictor` to allow the `word_data` during inference. (the `CustomPredictor` will be specified below.)



### 2. Added `custom_predictor.py`
- Created `custom_predictor.py`:
  - Accepts `image` and `word_data` in `__call__`.
  - Inserts `word_data` into model inputs like in training.


### Problem
The model is running okay, but the predictions are not made. You can fix this threshold to observe the output of the model and predictions, but still no 
```python
# inference.py, line 48
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
```
