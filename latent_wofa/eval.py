from my_model_library import EvaluationMetrics

def evaluate_attacks(model, images):
    results = {}
    
    # Various attack scenarios
    attacks = ['crop', 'rotate', 'jpeg']
    for attack in attacks:
        modified_images = apply_attack(images, attack)
        metrics = EvaluationMetrics()
        score = metrics.calculate(modified_images)
        results[attack] = score
    
    return results

if __name__ == '__main__':
    # Assume 'original_images' is pre-loaded
    scores = evaluate_attacks(model, original_images)
    for attack, score in scores.items():
        print(f'{attack} score: {score}')