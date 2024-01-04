import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


from sklearn.impute import SimpleImputer # Handling missing values
from sklearn.preprocessing import StandardScaler # Handing feature scaling
from sklearn.preprocessing import OneHotEncoder # for encoding

#pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



