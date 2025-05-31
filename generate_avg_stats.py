import json
import os
from pathlib import Path
import numpy as np

def calculate_average_stats():
    # Get all evaluation json files
    eval_dir = Path("evaluation_results")
    eval_files = list(eval_dir.glob("evaluation_*.json"))
    
    if not eval_files:
        print("No evaluation files found!")
        return
        
    # Initialize aggregates
    all_scores = {
        "chord_alignment": [],
        "timing_accuracy": [],
        "note_density": [],
        "overall": []
    }
    
    all_melody_stats = {
        "num_notes": [],
        "duration": [],
        "pitch_range": [],
        "avg_note_duration": []
    }
    
    # Collect stats from all files
    for eval_file in eval_files:
        with open(eval_file) as f:
            data = json.load(f)
            
        # Collect scores
        for metric, value in data["scores"].items():
            all_scores[metric].append(value)
            
        # Collect melody stats    
        for stat, value in data["melody_stats"].items():
            all_melody_stats[stat].append(value)
            
    # Calculate averages
    avg_stats = {
        "scores": {
            metric: float(np.mean(values)) 
            for metric, values in all_scores.items()
        },
        "melody_stats": {
            stat: float(np.mean(values))
            for stat, values in all_melody_stats.items()
        }
    }
    
    # Add number of songs analyzed
    avg_stats["num_songs_analyzed"] = len(eval_files)
    
    # Save to file
    output_path = eval_dir / "avg_stats.json"
    with open(output_path, 'w') as f:
        json.dump(avg_stats, f, indent=2)
        
    print(f"Average stats saved to {output_path}")

if __name__ == "__main__":
    calculate_average_stats()
