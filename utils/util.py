from joblib import dump, load
import spacy
from spacy.matcher import Matcher
import re
from datetime import datetime

nlp = spacy.load("en_core_web_sm")
dump(nlp, './nlp.joblib')