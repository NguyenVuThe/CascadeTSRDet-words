# Configs Setting
## Path
### 1. `model_config/fintabnet.yaml`
### 2. `layout_trainer.py` line 110
### 3. `inference.py` line 1011
### 4. `config.yaml`
### 5. `cascade_rcnn.py`

# Changelog
## [Past 1] - Setting some defaults
### 1. `MyGeneralizedRCNN` instead of `GeneralizedRCNN`
- Original, the model uses GeneralizedRCNN in `rcnn.py` of `detectron2`, but we should use the `rcnn.py` in folder `custom` to custom.
- We matched the function of the custom's `MyGeneralizedRCNN` the detectron2's `GeneralizedRCNN`  
```python
# Replace
images = [x["image"].to(self.device) for x in batched_inputs]
 |
 V
images = [self._move_to_current_device(x["image"]) for x in batched_inputs]

# Add this 
def _move_to_current_device(self, x):
    return move_device_like(x, self.pixel_mean)

# In fintabnet.yaml
Change GeneralizedRCNN to MyGeneralizedRCNN
```

## [Past 2] - Custom Train Pipeline with Word Data
### 1. Initial the words into `layout_trainer.py`
```python
# layout_trainer.py, line 110
words_dir = r"D:\MyWorking\dataset\FinTabNet.c\FinTabNec-Structure\words"
file_path = dataset_dict["file_name"]
file_name = os.path.basename(file_path)
word_json_name = file_name.replace('.jpg', '_words.json')
word_json_path = os.path.join(words_dir, word_json_name)
with open(word_json_path, 'r') as f:
    word_data = json.load(f)
dataset_dict['word_data'] = word_data
```
### 2. Input the word data to `rcnn.py`
```python
# line 155, 213
word_data_list = [x["word_data"] for x in batched_inputs]
# foward()
_, detector_losses = self.roi_heads(images, features, proposals, gt_instances, word_data_list=word_data_list)
#inference()
results, _ = self.roi_heads(images, features, proposals, None, word_data_list=word_data_list)
```
### 3. Input the word into ROIHeads in `cascade_rcnn.py`, the concat logic is also located here.
- Notice with some main functions of MyCascadeROIHeads, should add the parameter `word_data_list`
- The main function responsibles for concat the visual and text is `_run_stage()`

```python
def forward(self, images, features, proposals, targets=None, word_data_list=None)
def _forward_box(self, features, proposals, targets=None, word_data_list=None):
def _run_stage(self, features, proposals, stage, word_data_list=None):
```
- The text embedding and concat are only used in the last stage (Stage 2)
- Create `get_words_in_box()` function
```python
def get_words_in_box(self, box, word_data):
        x1, y1, x2, y2 = box
        return [word["text"] for word in word_data if x1 <= word["bbox"][0] and y1 <= word["bbox"][1]
                and x2 >= word["bbox"][2] and y2 >= word["bbox"][3]]
```
### 4. BERT and FFN
- We are currently using FinBERT, the output dimesion is 768

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


### 3. Problem
#### a. No predictions
The model is running okay, but the predictions are not made. You can fix this threshold to observe the output of the model and predictions. We have tried Threshold = 0.05, and it comes out with class and bbox but it still makes no prediction. 
```python
# inference.py, line 48
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
```
#### b. The parameter of BERT is too large
Maybe we should change the BERT, and find a reliable reason the right BERT.
```python
# BERT
Total parameters: 109,482,240
Trainable parameters: 109,482,240

# Original
Total parameters: 86,654,917
Trainable parameters: 86,401,359

# This model
Total parameters: 199,022,789
Trainable parameters: 198,769,231
```

#### c. What is the right learning rate?
Original
```python
STEPS: (84375, )
MAX_ITER: 112500
IMS_PER_BATCH: 16
BASE_LR: 0.02
CHECKPOINT_PERIOD: 10000
```
Schedule
```python
STEPS: (30000, 40000)
MAX_ITER: 50000
IMS_PER_BATCH: 2
BASE_LR: 0.0025
CHECKPOINT_PERIOD: 5000
```


#### d. What is the logic of training?
- Foward Pass (main things we are doing)
- Compute Loss
- Backproagation
- Optimizer
#### e. How the postprocess works?
Firstly, from the outputs of the model, we wil performance via objects.
```python
Sample boxes: tensor([[243.9738,   2.2211, 294.8190,  79.1781],
        [  2.0276,   2.2764, 294.9039,  79.2712],
        [181.9968,   2.1392, 243.9308,  79.1556]])

Sample labels: tensor([1, 0, 1])

Sample scores: tensor([0.9992, 0.9990, 0.9990])
``` 
To
```python
Objects: table column, score: 0.9992096424102783, bbox: [243.9738006591797, 2.221088171005249, 294.81903076171875, 79.17813873291016]
```
From this, we perform `refined_rows` and `refined columns` with NMS
- Sort the highest confidence score
- Remove the ovelap

#### f. Try to run Google Colab.
ok

#### g. Choices
- What BERT should I use?
- How should I concat text embeddings and visual features
#### What is the point of this thesis?
- The stages?
- The BERTs?
- The postprocess logic(most reasonable)

## [14-08-2025] - Run with Colab
### Settings
```python
STEPS: (30000, 40000)
MAX_ITER: 50000

IMS_PER_BATCH: 4
BASE_LR: 0.005

CHECKPOINT_PERIOD: 1000
EVAL_PERIOD: 5000
```
### Observation
Set up - 5m
19 iter - 12m
39 iter- 20m

## [23-08-2025] - Use Precompute Embeddings
### 1. Create `words_to_embed.py`
This script creates two files:
- `embeddings.npy`
- `token2id`

### 2. Add `text_embedding.py` 
### 3. Change the `run_stage` function (stage 2) to precompute
- The path is in line 99
```python
self.text_lookup = TextEmbeddingLookup(
            emb_path="embeddings/embeddings.npy",
            vocab_path="embeddings/token2id.json"
        )
```
- Change the `run_stage` function