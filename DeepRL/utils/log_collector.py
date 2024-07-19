import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_all_scalars(event_file):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']
    
    data = {}
    steps = set()
    
    for tag in tags:
        scalar_events = event_acc.Scalars(tag)
        tag_data = {event.step: event.value for event in scalar_events}
        steps.update(tag_data.keys())
        data[tag] = tag_data
    
    steps = sorted(steps)
    aligned_data = {'step': steps}
    
    for tag in tags:
        tag_values = []
        for step in steps:
            if step in data[tag]:
                tag_values.append(data[tag][step])
            else:
                tag_values.append(None)  # Use None for missing values
        aligned_data[tag] = tag_values
    
    return pd.DataFrame(aligned_data)

def convert_events_to_csv(env_path):
    for model_run in os.listdir(env_path):
        model_run_path = os.path.join(env_path, model_run)
        if os.path.isdir(model_run_path):
            event_file = None
            for file_name in os.listdir(model_run_path):
                if file_name.startswith('events.out.tfevents'):
                    event_file = os.path.join(model_run_path, file_name)
                    break
            if event_file:
                df = extract_all_scalars(event_file)
                csv_file = f"{model_run.replace('_', '')}.csv"
                df.to_csv(os.path.join(env_path, csv_file), index=False)

# Example usage
#env_path = 'racetrack'
#convert_events_to_csv(env_path)
