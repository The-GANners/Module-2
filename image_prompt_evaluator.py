import torch
from PIL import Image
import clip
import os
import numpy as np
import re

class ImagePromptEvaluator:
    def __init__(self):
        # Load the CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"Model loaded on {self.device}")
        
        # Calibration scores for normalization
        self.baseline_scores = self._get_baseline_scores()

    def _get_baseline_scores(self):
        """Get baseline scores for normalization"""
        return {
            "random_text": 0.1,  # Random text vs random image
            "generic_match": 0.2,  # Generic descriptions
            "good_match": 0.3,    # Well-matching content
            "perfect_match": 0.35  # Very specific matches
        }

    def _preprocess_prompt(self, prompt):
        """Clean and enhance the prompt for better matching"""
        # Remove special characters and extra spaces
        cleaned = re.sub(r'[^\w\s]', ' ', prompt.lower())
        cleaned = ' '.join(cleaned.split())
        
        # Generate variations of the prompt
        variations = [
            prompt,  # Original
            cleaned,  # Cleaned
            f"a photo of {cleaned}",  # Photo format
            f"an image showing {cleaned}",  # Image format
            f"a picture of {cleaned}",  # Picture format
        ]
        
        return variations

    def calculate_similarity(self, image_path, prompt):
        """Calculate similarity score between an image and a text prompt with improvements"""
        # Load and preprocess the image
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Get prompt variations
        prompt_variations = self._preprocess_prompt(prompt)
        similarities = []
        
        # Calculate features for image once
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Test all prompt variations
            for variant in prompt_variations:
                text_input = clip.tokenize([variant], truncate=True).to(self.device)
                text_features = self.model.encode_text(text_input)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity (cosine similarity)
                similarity = (image_features @ text_features.T).item()
                similarities.append(similarity)
        
        # Use the best similarity score
        max_similarity = max(similarities)
        avg_similarity = np.mean(similarities)
        
        # Apply semantic contradiction penalty
        penalized_score = self._apply_semantic_penalty(image_path, prompt, max_similarity)
        
        # Normalize the score to a 0-100% range with better scaling
        normalized_score = self._normalize_score(penalized_score)
        
        return penalized_score, {
            "raw_score": max_similarity,
            "penalized_score": penalized_score,
            "average_score": avg_similarity,
            "normalized_score": normalized_score,
            "percentage": f"{normalized_score:.2f}%",
            "all_scores": similarities
        }
    
    def _apply_semantic_penalty(self, image_path, prompt, similarity_score):
        """Apply penalty for semantic contradictions - fixed logic"""
        # Load and preprocess the image
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Define contradictory animal pairs and other objects
        contradictions = {
            'bear': ['eagle', 'bird', 'flying', 'wings', 'beak'],
            'eagle': ['bear', 'mammal', 'fur', 'paws', 'walking'],
            'cat': ['dog', 'barking', 'canine'],
            'dog': ['cat', 'meowing', 'feline'],
            'car': ['airplane', 'flying', 'wings'],
            'airplane': ['car', 'road', 'wheels', 'driving']
        }
        
        prompt_lower = prompt.lower()
        main_concept = None
        
        # Find the main concept in the prompt
        for concept in contradictions.keys():
            if concept in prompt_lower:
                main_concept = concept
                break
        
        if main_concept:
            # Test for contradictory concepts in the image
            contradictory_terms = contradictions[main_concept]
            contradiction_scores = []
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                for term in contradictory_terms:
                    # Test if the contradictory concept is present in the image
                    test_prompt = f"a photo of {term}"
                    text_input = clip.tokenize([test_prompt], truncate=True).to(self.device)
                    text_features = self.model.encode_text(text_input)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    contradiction_sim = (image_features @ text_features.T).item()
                    contradiction_scores.append(contradiction_sim)
            
            # Only apply penalty if contradictory concepts score SIGNIFICANTLY higher
            max_contradiction = max(contradiction_scores) if contradiction_scores else 0
            
            # More restrictive penalty conditions - only for clear contradictions
            if max_contradiction > similarity_score + 0.05:  # Contradiction must be 0.05 higher
                # Strong contradiction detected - apply severe penalty
                penalty_factor = 0.4
                penalized_score = similarity_score * penalty_factor
                return penalized_score
            elif max_contradiction > similarity_score + 0.02 and max_contradiction > 0.25:
                # Moderate contradiction - apply light penalty
                penalty_factor = 0.8
                penalized_score = similarity_score * penalty_factor
                return penalized_score
        
        return similarity_score
    
    def _normalize_score(self, raw_score):
        """Normalize CLIP scores to a more intuitive 0-100% range - improved"""
        # Better normalization that doesn't penalize good matches
        
        if raw_score < 0.15:
            # Poor match
            return max(0, raw_score * 200)  # 0-30%
        elif raw_score < 0.22:
            # Fair match  
            return 30 + (raw_score - 0.15) * 285  # 30-50%
        elif raw_score < 0.30:
            # Good match
            return 50 + (raw_score - 0.22) * 375  # 50-80%
        else:
            # Excellent match
            return 80 + min(20, (raw_score - 0.30) * 200)  # 80-100%
    
    def evaluate_image(self, image_path, prompt, threshold=0.20):
        """Evaluate if the generated image matches the prompt with better thresholds"""
        similarity, metrics = self.calculate_similarity(image_path, prompt)
        normalized_score = metrics["normalized_score"]
        
        # Better quality thresholds based on actual CLIP behavior
        if similarity > 0.28:
            quality = "Excellent"
            feedback = "The image closely matches the prompt."
        elif similarity > 0.22:
            quality = "Good"
            feedback = "The image generally matches the prompt."
        elif similarity > 0.18:
            quality = "Fair"
            feedback = "The image somewhat matches the prompt but could be improved."
        else:
            quality = "Poor"
            feedback = "The image doesn't seem to match the prompt well."
        
        # Fixed keyword analysis - no recursive calls
        keywords = [word for word in prompt.lower().split() if len(word) > 3]
        keyword_checks = []
        
        # Load image once for keyword analysis
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for keyword in keywords:
                test_prompts = [
                    f"an image containing {keyword}",
                    f"a photo of {keyword}",
                    f"{keyword}"
                ]
                
                kw_similarities = []
                for test_prompt in test_prompts:
                    text_input = clip.tokenize([test_prompt], truncate=True).to(self.device)
                    text_features = self.model.encode_text(text_input)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    kw_similarity = (image_features @ text_features.T).item()
                    kw_similarities.append(kw_similarity)
                
                best_kw_sim = max(kw_similarities)
                keyword_checks.append({
                    "keyword": keyword,
                    "present": best_kw_sim > 0.20,  # More conservative threshold
                    "confidence": f"{self._normalize_score(best_kw_sim):.1f}%",
                    "raw_score": best_kw_sim
                })
        
        # More accurate contradiction warning
        contradiction_warning = ""
        if "penalized_score" in metrics and metrics["penalized_score"] < metrics["raw_score"] * 0.95:
            contradiction_warning = "⚠️  Semantic contradiction detected - score adjusted downward."
        
        return {
            "overall_score": similarity,
            "percentage_match": f"{normalized_score:.2f}%",
            "raw_percentage": f"{similarity * 100:.2f}%",
            "quality": quality,
            "feedback": feedback,
            "prompt": prompt,
            "keyword_analysis": keyword_checks,
            "meets_threshold": similarity > threshold,
            "detailed_metrics": metrics,
            "contradiction_warning": contradiction_warning
        }

def main():
    """Command line interface for the image-prompt evaluator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate how well an image matches a text prompt")
    parser.add_argument("--image", required=True, help="Path to the generated image")
    parser.add_argument("--prompt", required=True, help="Text prompt used to generate the image")
    parser.add_argument("--threshold", type=float, default=0.22, help="Similarity threshold (default: 0.22)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    evaluator = ImagePromptEvaluator()
    results = evaluator.evaluate_image(args.image, args.prompt, args.threshold)
    
    print("\n" + "="*50)
    print(f"Image-Prompt Evaluation Results")
    print("="*50)
    print(f"Prompt: \"{results['prompt']}\"")
    print(f"Overall Match: {results['percentage_match']} ({results['quality']})")
    print(f"Raw CLIP Score: {results['raw_percentage']}")
    print(f"Feedback: {results['feedback']}")
    
    if results['contradiction_warning']:
        print(f"\n{results['contradiction_warning']}")
    
    if args.verbose:
        print(f"\nDetailed Metrics:")
        metrics = results['detailed_metrics']
        print(f"  Raw Score: {metrics['raw_score']:.4f}")
        if 'penalized_score' in metrics:
            print(f"  Penalized Score: {metrics['penalized_score']:.4f}")
        print(f"  Average Score: {metrics['average_score']:.4f}")
        print(f"  All Variation Scores: {[f'{s:.3f}' for s in metrics['all_scores']]}")
    
    print("\nKeyword Analysis:")
    for kw in results['keyword_analysis']:
        status = "✓" if kw['present'] else "✗"
        print(f"  {status} {kw['keyword']} - {kw['confidence']} (raw: {kw['raw_score']:.3f})")
    
    print("="*50)
    
if __name__ == "__main__":
    main()
