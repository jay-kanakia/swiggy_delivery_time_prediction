import numpy as np
import pandas as pd
import logging

from joblib import Path

# create logger
logger = logging.getLogger('data_cleaning')
logger.setLevel(logging.INFO)

## create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# create formatter
formater = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add fromater to handler
console_handler.setFormatter(formater)

# add handler to logger
logger.addHandler(console_handler)

columns_to_drop =  ['rider_id',
                    'restaurant_latitude',
                    'restaurant_longitude',
                    'delivery_latitude',
                    'delivery_longitude',
                    'order_date',
                    "order_time_hour",
                    "order_day",
                    "city_name",
                    "order_day_of_week",
                    "order_month"]

