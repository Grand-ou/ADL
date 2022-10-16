
## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## reproduce
```shell
python train_intent.py
python train_slot.py
```

## inference
```shell
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
```
