from src.components import exception
from src.components import logger
import pandas as pd
import numpy as np
from src.utils import load_obj

preprocessor=load_obj('./artifacts/preprocessor.pkl')
model=load_obj('./artifacts/model.pkl')
