import random
from datetime import datetime, timedelta


class DataGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
    
    def generate_dates(self, count, start_date=None, 
                       end_date=None):
        if start_date is None:
            start_date = datetime(2020, 1, 1)
        if end_date is None:
            end_date = datetime(2026, 12, 31)
        
        time_delta = (end_date - start_date).total_seconds()
        
        for _ in range(count):
            random_seconds = random.uniform(0, time_delta)
            random_date = start_date + timedelta(seconds=random_seconds)
            yield random_date.strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_unique_dates(self, count):
        generated = set()
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2026, 12, 31)
        time_delta = (end_date - start_date).total_seconds()
        
        while len(generated) < count:
            random_seconds = random.uniform(0, time_delta)
            random_date = start_date + timedelta(seconds=random_seconds)
            date_str = random_date.strftime("%Y-%m-%d %H:%M:%S")
            
            if date_str not in generated:
                generated.add(date_str)
                yield date_str
    
    def generate_stream_with_duplicates(self, unique_count, 
                                       total_count):
        unique_items = list(self.generate_unique_dates(unique_count))
        
        for _ in range(total_count):
            yield random.choice(unique_items)
