from typing import Dict, Tuple

class RiskScorer:
    """Class that computes risk scores based on detected suspicious patterns"""
    
    def __init__(self):
        """Initialize risk weights and thresholds for different patterns"""
        # Weights represent suspiciousness of each pattern (higher values = more suspicious)
        self.weights = {
            'chain_hopping': 0.95,      # Highest weight - cross-chain laundering
            'structuring': 0.9,         # Breaking large amounts into small deposits
            'layering': 0.85,           # Complex multi-hop obfuscation
            'peel_chain': 0.8,          # Sequential transfers with decreasing amounts
            'rapid_movement': 0.7,      # Fast sequential transactions
            'coinjoin': 0.6,            # Privacy mixing (less suspicious)
            'tf_crowdfunding': 0.6      # Terrorist financing through micro-donations
        }
        
        # Thresholds to categorize final risk score into risk levels
        self.thresholds = {
            'LOW': 0.4,                 # Risk scores up to 0.4
            'MEDIUM': 0.7,              # Risk scores 0.4 to 0.7
            'HIGH': 1.0                 # Risk scores above 0.7
        }

    def compute_risk_score(self, detection_results: Dict) -> Tuple[float, str, Dict]:
        """
        Aggregate pattern confidences into overall risk score using weighted maximum approach
        
        Args:
            detection_results: Dictionary of pattern detection results with confidence scores
            
        Returns:
            Tuple containing (risk_score, risk_level, detailed_summary)
        """
        weighted_scores = []        # List of weighted confidence scores
        active_patterns = []        # List of patterns with positive detections

        # Process each pattern detected to calculate weighted scores
        for pattern, result in detection_results.items():
            if pattern in self.weights and result.get('confidence', 0) > 0:
                confidence = result['confidence']
                # Multiply confidence by pattern weight to get weighted score
                weighted_score = confidence * self.weights[pattern]
                weighted_scores.append(weighted_score)
                
                # Store pattern details for reporting
                active_patterns.append({
                    'pattern': pattern,
                    'confidence': confidence,
                    'weighted_score': weighted_score,
                    'explanation': result.get('explanation', '')
                })

        # Final risk score is the highest weighted score (most suspicious pattern dominates)
        final_score = max(weighted_scores) if weighted_scores else 0.0

        # Determine risk level based on predefined thresholds
        if final_score <= self.thresholds['LOW']:
            risk_level = 'LOW'
        elif final_score <= self.thresholds['MEDIUM']:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'

        # Create comprehensive summary of risk assessment
        summary = {
            'overall_risk_score': final_score,          # Final computed risk score
            'risk_level': risk_level,                   # Categorical risk level
            'active_patterns': active_patterns,         # List of detected patterns
            'pattern_count': len(active_patterns),      # Number of active patterns
            'weighted_scores': weighted_scores,         # All weighted scores
            'max_weighted_score': final_score           # Highest weighted score
        }

        return final_score, risk_level, summary

    def get_pattern_weight(self, pattern: str) -> float:
        """
        Get the weight assigned to a specific pattern
        
        Args:
            pattern: Name of the pattern to get weight for
            
        Returns:
            Weight value for the pattern (0.0 if pattern not found)
        """
        return self.weights.get(pattern, 0.0)

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update weights for specified patterns with validation
        
        Args:
            new_weights: Dictionary mapping pattern names to new weight values
        """
        for pattern, weight in new_weights.items():
            if pattern in self.weights:
                # Ensure weight stays within valid range [0.0, 1.0]
                self.weights[pattern] = max(0.0, min(1.0, weight))
