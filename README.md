# Weather Prediction Using Machine Learning

This project is a **weather prediction** system built using Python and Jupyter Notebook. It uses the **Szeged Weather Dataset (2006–2016)** from Kaggle to train various regression models to predict weather-related features such as temperature.

##  Dataset

**Weather in Szeged 2006-2016**  
This dataset contains hourly/daily weather data from Szeged, Hungary, including:
- Temperature
- Pressure
- Wind Speed
- Humidity  
And more.

Dataset link: [Weather in Szeged - Kaggle](https://www.kaggle.com/datasets/budincsevity/szeged-weather)

---

##  Libraries Used

The following Python libraries are used in this project:

```python
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  
from sklearn.preprocessing import MinMaxScaler  
from sklearn.linear_model import LinearRegression  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.neighbors import KNeighborsRegressor  
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  
import matplotlib.pyplot as plt  
from matplotlib.pyplot import rcParams  
import joblib  
