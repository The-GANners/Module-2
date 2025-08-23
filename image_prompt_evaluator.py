import torch
from PIL import Image
import clip
import os
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class ImagePromptEvaluator:
    def __init__(self):
        # Load the CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"Model loaded on {self.device}")
        
        # Download NLTK data if not already present
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords data...")
            nltk.download('stopwords', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer data...")
            nltk.download('punkt', quiet=True)
        
        # Get English stop words from NLTK
        self.stop_words = set(stopwords.words('english'))
        
        # Add some domain-specific stop words for image prompts
        self.stop_words.update({
            'image', 'photo', 'picture', 'showing', 'contains', 'featuring',
            'depicts', 'displays', 'includes', 'scene', 'view'
        })
        
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
    
    def _extract_important_keywords(self, prompt):
        """Extract the most important keywords from prompt using NLTK POS tagging for generalization"""
        # Tokenize and POS tag the prompt
        tokens = word_tokenize(prompt.lower())
        tagged = nltk.pos_tag(tokens)
        # Nouns, adjectives, and some verbs (for actions)
        important_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS', 'VB', 'VBG', 'VBN'}
        important_words = []
        for word, pos in tagged:
            clean_word = re.sub(r'[^\w]', '', word)
            if (len(clean_word) > 2 and
                clean_word not in self.stop_words and
                not clean_word.isdigit() and
                pos in important_pos):
                important_words.append(clean_word)
        # Fallback: if nothing found, use original logic
        if not important_words:
            for word in prompt.lower().split():
                clean_word = re.sub(r'[^\w]', '', word)
                if (len(clean_word) > 2 and 
                    clean_word not in self.stop_words and
                    not clean_word.isdigit()):
                    important_words.append(clean_word)
        return important_words[:8]  # Limit to most important 8 keywords
    
    def _get_feature_status(self, feature_score, overall_score):
        """Determine status of a feature relative to overall image match"""
        if feature_score < 0.15:
            return "missing"
        elif feature_score < 0.22:
            return "weak"
        elif feature_score < overall_score - 0.05:
            return "below_average"
        else:
            return "present"
    
    def _analyze_missing_features(self, image_path, prompt, overall_similarity):
        """Enhanced analysis of missing features with better feedback"""
        
        # Extract important keywords (nouns, adjectives, key descriptors)
        important_keywords = self._extract_important_keywords(prompt)
        
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        missing_features = []
        present_features = []
        weak_features = []
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for keyword in important_keywords:
                # Test multiple variations for better detection
                test_prompts = [
                    f"an image containing {keyword}",
                    f"a photo showing {keyword}",
                    f"{keyword}",
                    f"featuring {keyword}",
                    f"with {keyword}"
                ]
                
                kw_similarities = []
                for test_prompt in test_prompts:
                    text_input = clip.tokenize([test_prompt], truncate=True).to(self.device)
                    text_features = self.model.encode_text(text_input)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    kw_similarity = (image_features @ text_features.T).item()
                    kw_similarities.append(kw_similarity)
                
                best_score = max(kw_similarities)
                confidence = self._normalize_score(best_score)
                
                # Categorize features based on confidence
                feature_info = {
                    "keyword": keyword,
                    "confidence": confidence,
                    "raw_score": best_score,
                    "status": self._get_feature_status(best_score, overall_similarity)
                }
                
                if best_score < 0.15:  # Clearly missing
                    missing_features.append(feature_info)
                elif best_score < 0.22:  # Weakly present
                    weak_features.append(feature_info)
                else:  # Clearly present
                    present_features.append(feature_info)
        
        return {
            "missing_features": missing_features,
            "weak_features": weak_features,
            "present_features": present_features,
            "missing_count": len(missing_features),
            "weak_count": len(weak_features),
            "total_features": len(important_keywords)
        }
    
    def _calculate_missing_feature_penalty(self, missing_analysis, overall_similarity):
        """Apply penalty based on missing critical features"""
        missing_count = missing_analysis["missing_count"]
        weak_count = missing_analysis["weak_count"]
        total_count = missing_analysis["total_features"]
        
        if total_count == 0:
            return overall_similarity
        
        missing_ratio = missing_count / total_count
        weak_ratio = weak_count / total_count
        
        # Apply graduated penalty based on missing feature ratio
        penalty = 0
        if missing_ratio > 0.6:  # More than 60% missing
            penalty = 0.3
        elif missing_ratio > 0.4:  # More than 40% missing
            penalty = 0.2
        elif missing_ratio > 0.2:  # More than 20% missing
            penalty = 0.1
        
        # Additional penalty for weak features
        if weak_ratio > 0.4:
            penalty += 0.1
        elif weak_ratio > 0.2:
            penalty += 0.05
        
        penalty = min(penalty, 0.4)  # Cap maximum penalty at 40%
        return overall_similarity * (1 - penalty)
    
    def _generate_missing_feature_feedback(self, missing_analysis):
        """Generate human-readable feedback about missing features"""
        missing = missing_analysis["missing_features"]
        weak = missing_analysis["weak_features"]
        present = missing_analysis["present_features"]
        
        feedback = []
        
        if missing:
            missing_names = [f["keyword"] for f in missing]
            feedback.append(f"❌ Missing features: {', '.join(missing_names[:4])}")
        
        if weak:
            weak_names = [f["keyword"] for f in weak]
            feedback.append(f"⚠️  Weak features: {', '.join(weak_names[:3])}")
        
        if len(missing) + len(weak) == 0:
            feedback.append("✅ All key features are well represented")
        elif len(present) > len(missing) + len(weak):
            feedback.append("✅ Most features are present")
        
        # Specific improvement suggestions
        if missing:
            feedback.append(f"💡 Consider regenerating to include: {', '.join(missing_names[:3])}")
        
        return " | ".join(feedback)
    
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
        """Evaluate if the generated image matches the prompt with enhanced missing feature analysis"""
        similarity, metrics = self.calculate_similarity(image_path, prompt)
        
        # Analyze missing features
        missing_analysis = self._analyze_missing_features(image_path, prompt, similarity)
        
        # Apply missing feature penalty
        adjusted_similarity = self._calculate_missing_feature_penalty(missing_analysis, similarity)
        adjusted_normalized = self._normalize_score(adjusted_similarity)
        
        # Generate missing feature feedback
        missing_feedback = self._generate_missing_feature_feedback(missing_analysis)
        
        # Calculate penalty applied
        missing_penalty = similarity - adjusted_similarity
        
        # Better quality thresholds based on adjusted similarity
        if adjusted_similarity > 0.28:
            quality = "Excellent"
            feedback = "The image closely matches the prompt."
        elif adjusted_similarity > 0.22:
            quality = "Good"
            feedback = "The image generally matches the prompt."
        elif adjusted_similarity > 0.18:
            quality = "Fair"
            feedback = "The image somewhat matches the prompt but could be improved."
        else:
            quality = "Poor"
            feedback = "The image doesn't seem to match the prompt well."
        
        # Enhanced keyword analysis using the new system
        enhanced_keyword_checks = []
        
        # Add present features
        for feature in missing_analysis["present_features"]:
            enhanced_keyword_checks.append({
                "keyword": feature["keyword"],
                "present": True,
                "confidence": f"{feature['confidence']:.1f}%",
                "raw_score": feature["raw_score"],
                "status": "✅ present",
                "status_type": "present"
            })
        
        # Add weak features
        for feature in missing_analysis["weak_features"]:
            enhanced_keyword_checks.append({
                "keyword": feature["keyword"],
                "present": False,
                "confidence": f"{feature['confidence']:.1f}%",
                "raw_score": feature["raw_score"],
                "status": "⚠️  weak",
                "status_type": "weak"
            })
        
        # Add missing features
        for feature in missing_analysis["missing_features"]:
            enhanced_keyword_checks.append({
                "keyword": feature["keyword"],
                "present": False,
                "confidence": f"{feature['confidence']:.1f}%",
                "raw_score": feature["raw_score"],
                "status": "❌ missing",
                "status_type": "missing"
            })
        
        # Fallback to original keyword analysis for any remaining words
        # Use the same NLTK stop words from __init__
        
        # Get remaining keywords, filtering out stop words
        original_keywords = []
        for word in prompt.lower().split():
            clean_word = re.sub(r'[^\w]', '', word)
            if (len(clean_word) > 3 and 
                clean_word not in self.stop_words and 
                not clean_word.isdigit()):
                original_keywords.append(clean_word)
        
        analyzed_keywords = [f["keyword"] for f in missing_analysis["present_features"] + 
                           missing_analysis["weak_features"] + missing_analysis["missing_features"]]
        
        remaining_keywords = [kw for kw in original_keywords if kw not in analyzed_keywords]
        
        if remaining_keywords:
            # Load image once for remaining keyword analysis
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                for keyword in remaining_keywords:
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
                    enhanced_keyword_checks.append({
                        "keyword": keyword,
                        "present": best_kw_sim > 0.20,
                        "confidence": f"{self._normalize_score(best_kw_sim):.1f}%",
                        "raw_score": best_kw_sim,
                        "status": "✓" if best_kw_sim > 0.20 else "✗",
                        "status_type": "present" if best_kw_sim > 0.20 else "missing"
                    })
        
        # More accurate contradiction warning
        contradiction_warning = ""
        if "penalized_score" in metrics and metrics["penalized_score"] < metrics["raw_score"] * 0.95:
            contradiction_warning = "⚠️  Semantic contradiction detected - score adjusted downward."
        
        return {
            "overall_score": adjusted_similarity,
            "original_score": similarity,
            "percentage_match": f"{adjusted_normalized:.2f}%",
            "original_percentage": f"{self._normalize_score(similarity):.2f}%",
            "raw_percentage": f"{similarity * 100:.2f}%",
            "quality": quality,
            "feedback": feedback,
            "prompt": prompt,
            "keyword_analysis": enhanced_keyword_checks,
            "meets_threshold": adjusted_similarity > threshold,
            "detailed_metrics": metrics,
            "contradiction_warning": contradiction_warning,
            "missing_feature_analysis": missing_analysis,
            "missing_feature_feedback": missing_feedback,
            "missing_feature_penalty": missing_penalty,
            "penalty_percentage": f"{(missing_penalty / similarity * 100):.1f}%" if similarity > 0 else "0.0%"
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
    
    # Show penalty information if applied
    if results['missing_feature_penalty'] > 0:
        print(f"Original Score: {results['original_percentage']} → Final Score: {results['percentage_match']}")
        print(f"Missing Feature Penalty: -{results['penalty_percentage']} applied")
    
    print(f"Raw CLIP Score: {results['raw_percentage']}")
    print(f"Feedback: {results['feedback']}")
    
    # Enhanced missing feature feedback
    if results['missing_feature_feedback']:
        print(f"\nFeature Analysis:")
        print(f"  {results['missing_feature_feedback']}")
    
    # Feature summary
    missing_analysis = results['missing_feature_analysis']
    total_features = missing_analysis['total_features']
    present_count = len(missing_analysis['present_features'])
    weak_count = len(missing_analysis['weak_features'])
    missing_count = len(missing_analysis['missing_features'])
    
    if total_features > 0:
        print(f"\nFeature Summary ({total_features} total features):")
        print(f"  ✅ Present: {present_count}")
        if weak_count > 0:
            print(f"  ⚠️  Weak: {weak_count}")
        if missing_count > 0:
            print(f"  ❌ Missing: {missing_count}")
    
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
        
        # Show detailed feature breakdown
        if missing_analysis['present_features']:
            print(f"\n  Present Features:")
            for feature in missing_analysis['present_features']:
                print(f"    ✅ {feature['keyword']} - {feature['confidence']:.1f}%")
        
        if missing_analysis['weak_features']:
            print(f"\n  Weak Features:")
            for feature in missing_analysis['weak_features']:
                print(f"    ⚠️  {feature['keyword']} - {feature['confidence']:.1f}%")
        
        if missing_analysis['missing_features']:
            print(f"\n  Missing Features:")
            for feature in missing_analysis['missing_features']:
                print(f"    ❌ {feature['keyword']} - {feature['confidence']:.1f}%")
    
    print("\nDetailed Keyword Analysis:")
    for kw in results['keyword_analysis']:
        if 'status' in kw:
            print(f"  {kw['status']} {kw['keyword']} - {kw['confidence']} (raw: {kw['raw_score']:.3f})")
        else:
            status = "✓" if kw['present'] else "✗"
            print(f"  {status} {kw['keyword']} - {kw['confidence']} (raw: {kw['raw_score']:.3f})")
    
    print("="*50)
    
if __name__ == "__main__":
    main()
