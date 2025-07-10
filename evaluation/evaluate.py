from collections import defaultdict
import json

import numpy as np
from evaluation.evaluation_config import EvaluationConfig

def compute_mean_uncertainty(mc_results: dict):
    results = {}
    for style, style_data in mc_results.items():
        results[style] = {}
        for prompt_id, prompt_data in style_data.items():
            score_collections = defaultdict(list)
            for image_id, image_data in prompt_data.items():
                for score_type, score in image_data.items():
                    score_collections[score_type].append(score)

            results[style][prompt_id] = {}
            for score_type, scores in score_collections.items():
                results[style][prompt_id][score_type] = np.mean(scores)
    
    return results

def main() -> None:
    json_file = "output.json"
    with open(json_file, "r") as f:
        mc_results = json.load(
            f,
            object_hook=lambda d: {
                int(k) if k.lstrip("-").isdigit() else k: v for k, v in d.items()
            },
        )
    results = compute_mean_uncertainty(mc_results)
    output_json_file = "output_mean_uncertainty_scores.json"
    with open(output_json_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
