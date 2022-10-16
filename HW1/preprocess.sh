wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
unzip glove.840B.300d.zip
python preprocess_intent.py
python preprocess_slot.py