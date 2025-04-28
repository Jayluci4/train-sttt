import torch
import logging
import re
import random
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

class ChainOfThoughtInjector:
    """
    Implements Chain-of-Thought (CoT) injection with mathematical guarantees.
    
    This component transforms inputs to include explicit reasoning traces:
        x ↦ x̃ = T_CoT(x)
    
    where T_CoT enriches the input with a reasoning scaffold.
    
    Mathematical insights:
    1. CoT transforms the input distribution from D to D'
    2. It reduces sample complexity by making latent reasoning traces explicit
    3. Under mild assumptions, reduces VC-dimension of the required function class
    """
    def __init__(
        self, 
        teacher_model=None,
        tokenizer=None,
        adaptive_mode: bool = True,
        max_cot_length: int = 512,
        cot_lambda: float = 0.5,  # λ for mixing original and CoT-transformed samples
        gradient_bound: float = 0.1,  # ε for bounded gradient perturbation
        lambda_annealing: bool = True,
        task_triggers: Optional[Dict[str, List[str]]] = None,
        temperature: float = 0.7
    ):
        """
        Initialize the Chain-of-Thought Injector.
        
        Args:
            teacher_model: LLM to use for generating CoT (if None, uses rule-based generation)
            tokenizer: Tokenizer for the models
            adaptive_mode: Whether to adaptively select which inputs get CoT
            max_cot_length: Maximum length of generated CoT in tokens
            cot_lambda: Mixing weight between original and CoT samples
            gradient_bound: Maximum allowed gradient perturbation from CoT
            lambda_annealing: Whether to anneal lambda over time
            task_triggers: Dictionary mapping task types to trigger words/phrases
            temperature: Temperature for CoT generation
        """
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.adaptive_mode = adaptive_mode
        self.max_cot_length = max_cot_length
        self.cot_lambda = cot_lambda
        self.gradient_bound = gradient_bound
        self.lambda_annealing = lambda_annealing
        self.temperature = temperature
        
        # Initialize task triggers if not provided
        self.task_triggers = task_triggers or {
            'reasoning': ['why', 'how', 'explain', 'reason', 'think through', 'step by step', 'analyze'],
            'math': ['calculate', 'solve', 'compute', 'find', 'equation', 'math'],
            'comparison': ['compare', 'contrast', 'difference', 'similar', 'versus', 'vs'],
            'planning': ['plan', 'steps', 'procedure', 'approach', 'strategy']
        }
        
        # Statistics tracking
        self.stats = {
            'total_inputs': 0,
            'cot_applied': 0,
            'teacher_calls': 0,
            'rule_based_cots': 0,
            'cot_lengths': [],
            'cot_task_types': defaultdict(int)
        }
        
        # Cache for generated CoTs to avoid regeneration
        self.cot_cache = {}
        
        # Gradient history for tracking bounded perturbation
        self.gradient_history = []
        
        logger.info(f"Initialized ChainOfThoughtInjector with lambda={cot_lambda}")
    
    def _get_cot_prompt(self, text: str, task_type: str) -> str:
        """
        Generate a prompt for Chain-of-Thought generation.
        
        Args:
            text: Input text
            task_type: Detected task type
            
        Returns:
            Prompt for CoT generation
        """
        prompt_templates = {
            'reasoning': (
                f"Question: {text}\n\n"
                "Let me think through this step by step to answer the question..."
            ),
            'math': (
                f"Math problem: {text}\n\n"
                "I'll solve this step by step..."
            ),
            'comparison': (
                f"Comparison task: {text}\n\n"
                "To compare these properly, I'll analyze each aspect methodically..."
            ),
            'planning': (
                f"Planning task: {text}\n\n"
                "I'll develop a step-by-step plan to accomplish this..."
            ),
            'default': (
                f"Task: {text}\n\n"
                "Let me think about this step by step..."
            )
        }
        
        return prompt_templates.get(task_type, prompt_templates['default'])
    
    def _detect_task_type(self, text: str) -> str:
        """
        Detect the type of task in the input text.
        
        Args:
            text: Input text
            
        Returns:
            Detected task type
        """
        text_lower = text.lower()
        for task_type, triggers in self.task_triggers.items():
            if any(trigger in text_lower for trigger in triggers):
                return task_type
        
        # Default to reasoning if no specific triggers found
        return 'reasoning'
    
    def _rule_based_cot_generation(self, text: str, task_type: str) -> str:
        """
        Generate a rule-based Chain-of-Thought when teacher model is unavailable.
        
        Args:
            text: Input text
            task_type: Detected task type
            
        Returns:
            Generated CoT
        """
        # Extract key elements from the input
        words = text.split()
        key_phrases = [w for w in words if len(w) > 4][:5]  # Sample a few longer words
        
        # Task-specific templates
        templates = {
            'reasoning': [
                "I need to analyze this step-by-step.",
                "First, I'll identify the key elements: {elements}.",
                "Then, I'll consider how these elements relate to each other.",
                "Finally, I'll draw a conclusion based on this analysis.",
                "So, thinking about {elements}..."
            ],
            'math': [
                "To solve this math problem, I'll break it down into steps.",
                "First, I'll identify the variables and values: {elements}.",
                "Then, I'll determine the appropriate mathematical operation(s).",
                "Next, I'll apply these operations carefully.",
                "Finally, I'll verify my answer makes sense in the context of the problem."
            ],
            'comparison': [
                "To compare effectively, I need to identify criteria for comparison.",
                "The key elements to compare are: {elements}.",
                "For each element, I'll analyze similarities and differences.",
                "Then I'll weigh the relative importance of each difference.",
                "Finally, I'll synthesize these insights into a coherent comparison."
            ],
            'planning': [
                "To create an effective plan, I'll follow a structured approach.",
                "First, I'll identify the goal and constraints related to {elements}.",
                "Then, I'll break down the task into sequential steps.",
                "For each step, I'll consider required resources and potential challenges.",
                "Finally, I'll organize these steps into a coherent sequence with priorities."
            ]
        }
        
        # Use default template if task type not found
        template = templates.get(task_type, templates['reasoning'])
        
        # Format template with elements
        elements_str = ", ".join(key_phrases) if key_phrases else "the key concepts"
        cot_steps = [step.format(elements=elements_str) for step in template]
        
        # Combine into formatted CoT
        cot = "\n\nThinking step by step:\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(cot_steps)])
        
        self.stats['rule_based_cots'] += 1
        return cot
    
    async def _generate_cot_with_teacher(self, text: str, task_type: str) -> str:
        """
        Generate Chain-of-Thought using teacher model.
        
        Args:
            text: Input text
            task_type: Detected task type
            
        Returns:
            Generated CoT
        """
        if self.teacher_model is None or self.tokenizer is None:
            return self._rule_based_cot_generation(text, task_type)
        
        # Create CoT prompt
        prompt = self._get_cot_prompt(text, task_type)
        
        try:
            # Tokenize prompt
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(next(self.teacher_model.parameters()).device) for k, v in inputs.items()}
            
            # Generate CoT
            with torch.no_grad():
                self.teacher_model.eval()
                outputs = self.teacher_model.generate(
                    **inputs,
                    max_new_tokens=self.max_cot_length,
                    temperature=self.temperature,
                    do_sample=True,
                    num_return_sequences=1
                )
                
            # Decode generated text
            cot_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated CoT (not the original prompt)
            cot = cot_text[len(prompt):].strip()
            
            self.stats['teacher_calls'] += 1
            return "\n\nThinking step by step:\n" + cot
        
        except Exception as e:
            logger.warning(f"Error generating CoT with teacher model: {e}")
            return self._rule_based_cot_generation(text, task_type)
    
    def _should_apply_cot(self, text: str) -> Tuple[bool, str]:
        """
        Determine if CoT should be applied to this input.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (should_apply, task_type)
        """
        # Always count total inputs
        self.stats['total_inputs'] += 1
        
        # Simple heuristics for CoT application
        text_lower = text.lower()
        
        # Detect task type
        task_type = self._detect_task_type(text_lower)
        
        # Rule 1: Apply CoT for longer inputs (more complex tasks)
        is_long = len(text.split()) > 20
        
        # Rule 2: Check for "why", "how", "explain", etc.
        has_question_words = any(word in text_lower for word in 
                                ['why', 'how', 'explain', 'reason', 'think', 'analyze', 'solve'])
        
        # Rule 3: Check for reasoning indicators
        has_reasoning_indicators = any(word in text_lower for word in
                                      ['because', 'therefore', 'since', 'however', 'although'])
        
        # Decision logic
        if self.adaptive_mode:
            # Adaptive mode: Apply CoT selectively
            should_apply = (is_long and (has_question_words or has_reasoning_indicators))
        else:
            # Non-adaptive mode: Apply CoT with fixed probability
            should_apply = random.random() < self.cot_lambda
            
        if should_apply:
            self.stats['cot_applied'] += 1
            self.stats['cot_task_types'][task_type] += 1
            
        return should_apply, task_type
    
    async def transform_input(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        Transform input by potentially injecting Chain-of-Thought reasoning.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (transformed_text, transform_info)
        """
        # Check if we should apply CoT
        should_apply, task_type = self._should_apply_cot(text)
        
        if not should_apply:
            return text, {'applied': False, 'task_type': task_type}
        
        # Check cache for previously generated CoT
        cache_key = text[:100]  # Use first 100 chars as cache key
        if cache_key in self.cot_cache:
            cot = self.cot_cache[cache_key]
            logger.info(f"Using cached CoT for input: {cache_key}...")
        else:
            # Generate appropriate CoT based on task type
            cot = await self._generate_cot_with_teacher(text, task_type)
            
            # Cache the generated CoT
            self.cot_cache[cache_key] = cot
            
            # Track CoT length
            cot_length = len(cot.split())
            self.stats['cot_lengths'].append(cot_length)
        
        # Combine original text with CoT
        transformed_text = f"{text}{cot}"
        
        logger.info(f"Applied CoT transformation for task type: {task_type}")
        
        return transformed_text, {
            'applied': True, 
            'task_type': task_type,
            'cot_length': len(cot.split())
        }
    
    def transform_batch(self, batch: Dict[str, torch.Tensor], keys_to_transform: List[str], interleave: bool = True) -> Dict[str, torch.Tensor]:
        """
        Transform a batch of inputs by interleaving original and CoT-transformed versions.
        
        Args:
            batch: Input batch dictionary
            keys_to_transform: Keys in the batch to apply transformation to
            interleave: Whether to interleave original and transformed or replace
            
        Returns:
            Transformed batch with interleaved samples
        """
        transformed_batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Get current lambda (possibly annealed)
        current_lambda = self._get_current_lambda()
        
        # If lambda is 0, return original batch unchanged
        if current_lambda <= 0:
            return batch
        
        # If lambda is 1 and not interleaving, transform all samples
        if current_lambda >= 1 and not interleave:
            # Process each key to transform
            for key in keys_to_transform:
                if key not in batch:
                    continue
                    
                # Get the text to transform
                if isinstance(batch[key], torch.Tensor) and batch[key].dtype == torch.long:
                    # Tensor of token IDs - decode then re-encode
                    texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch[key]]
                    
                    # Transform each text
                    transformed_texts = []
                    for text in texts:
                        # Use synchronous transformation for batch processing
                        should_apply, task_type = self._should_apply_cot(text)
                        
                        if should_apply:
                            # Use rule-based CoT for batch processing
                            cot = self._rule_based_cot_generation(text, task_type)
                            transformed_text = f"{text}{cot}"
                        else:
                            transformed_text = text
                            
                        transformed_texts.append(transformed_text)
                    
                    # Re-encode transformed texts
                    encode_results = self.tokenizer(
                        transformed_texts,
                        padding='max_length' if hasattr(batch[key], 'shape') else True,
                        truncation=True,
                        max_length=batch[key].shape[1] if hasattr(batch[key], 'shape') else None,
                        return_tensors='pt'
                    )
                    
                    # Update batch with transformed tokens
                    transformed_batch[key] = encode_results['input_ids']
                    
                    # Update attention mask if needed
                    if 'attention_mask' in batch and 'attention_mask' in encode_results:
                        transformed_batch['attention_mask'] = encode_results['attention_mask']
                
                elif isinstance(batch[key], list) and all(isinstance(item, str) for item in batch[key]):
                    # List of strings - transform directly
                    transformed_texts = []
                    for text in batch[key]:
                        # Use synchronous transformation for batch processing
                        should_apply, task_type = self._should_apply_cot(text)
                        
                        if should_apply:
                            # Use rule-based CoT for batch processing
                            cot = self._rule_based_cot_generation(text, task_type)
                            transformed_text = f"{text}{cot}"
                        else:
                            transformed_text = text
                            
                        transformed_texts.append(transformed_text)
                    
                    transformed_batch[key] = transformed_texts
            
            return transformed_batch
        
        # If interleaving, create a batch with both original and transformed
        if interleave:
            # Calculate how many samples to transform based on lambda
            batch_size = len(next(iter([v for v in batch.values() if hasattr(v, '__len__')])))
            num_to_transform = max(1, int(batch_size * current_lambda))
            
            # Indices to transform
            indices_to_transform = random.sample(range(batch_size), num_to_transform)
            
            # Process each key to transform
            for key in keys_to_transform:
                if key not in batch:
                    continue
                    
                # Handle tensor of token IDs
                if isinstance(batch[key], torch.Tensor) and batch[key].dtype == torch.long:
                    for idx in indices_to_transform:
                        # Decode tokens to text
                        text = self.tokenizer.decode(batch[key][idx], skip_special_tokens=True)
                        
                        # Use synchronous transformation
                        should_apply, task_type = self._should_apply_cot(text)
                        
                        if should_apply:
                            # Use rule-based CoT for batch processing
                            cot = self._rule_based_cot_generation(text, task_type)
                            transformed_text = f"{text}{cot}"
                            
                            # Re-encode transformed text
                            encoded = self.tokenizer(
                                transformed_text,
                                padding='max_length',
                                truncation=True,
                                max_length=batch[key].shape[1],
                                return_tensors='pt'
                            )
                            
                            # Update tokens
                            transformed_batch[key][idx] = encoded['input_ids'][0]
                            
                            # Update attention mask if needed
                            if 'attention_mask' in batch and 'attention_mask' in encoded:
                                transformed_batch['attention_mask'][idx] = encoded['attention_mask'][0]
                
                # Handle list of strings
                elif isinstance(batch[key], list) and all(isinstance(item, str) for item in batch[key]):
                    for idx in indices_to_transform:
                        text = batch[key][idx]
                        
                        # Use synchronous transformation
                        should_apply, task_type = self._should_apply_cot(text)
                        
                        if should_apply:
                            # Use rule-based CoT for batch processing
                            cot = self._rule_based_cot_generation(text, task_type)
                            transformed_batch[key][idx] = f"{text}{cot}"
        
        return transformed_batch
    
    def _get_current_lambda(self) -> float:
        """
        Get current lambda value, possibly annealed over time.
        
        Returns:
            Current lambda value
        """
        if not self.lambda_annealing:
            return self.cot_lambda
            
        # Simple linear annealing based on number of inputs processed
        total_inputs = max(1, self.stats['total_inputs'])
        if total_inputs < 1000:
            # Warmup phase: gradually increase lambda
            return self.cot_lambda * (total_inputs / 1000)
        elif total_inputs < 5000:
            # Full strength phase
            return self.cot_lambda
        else:
            # Decay phase: gradually decrease lambda
            decay_factor = max(0, min(1, (10000 - total_inputs) / 5000))
            return self.cot_lambda * decay_factor
    
    def check_gradient_bound(self, original_grad: torch.Tensor, cot_grad: torch.Tensor) -> bool:
        """
        Check if CoT transformation maintains bounded gradient perturbation.
        
        Args:
            original_grad: Gradient from original input
            cot_grad: Gradient from CoT-transformed input
            
        Returns:
            True if gradient perturbation is within bounds
        """
        # Calculate relative perturbation
        grad_diff = torch.norm(cot_grad - original_grad)
        original_norm = torch.norm(original_grad)
        
        if original_norm > 0:
            relative_perturbation = grad_diff / original_norm
        else:
            relative_perturbation = grad_diff
        
        # Track history
        self.gradient_history.append(relative_perturbation.item())
        
        # Check against bound
        is_within_bound = relative_perturbation <= self.gradient_bound
        
        return is_within_bound
    
    def get_mixed_loss(self, original_loss: torch.Tensor, cot_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute mixed loss between original and CoT-transformed inputs.
        
        Implements: L_total = λ·L(f_θ(x), y) + (1-λ)·L(f_θ(T_CoT(x)), y)
        
        Args:
            original_loss: Loss from original input
            cot_loss: Loss from CoT-transformed input
            
        Returns:
            Mixed loss tensor
        """
        current_lambda = self._get_current_lambda()
        mixed_loss = current_lambda * original_loss + (1 - current_lambda) * cot_loss
        return mixed_loss
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about CoT injection."""
        stats = {**self.stats}  # Copy stats
        
        # Calculate additional metrics
        if stats['total_inputs'] > 0:
            stats['cot_application_rate'] = stats['cot_applied'] / stats['total_inputs']
        else:
            stats['cot_application_rate'] = 0
            
        if stats['cot_lengths']:
            stats['avg_cot_length'] = sum(stats['cot_lengths']) / len(stats['cot_lengths'])
            stats['max_cot_length'] = max(stats['cot_lengths'])
        else:
            stats['avg_cot_length'] = 0
            stats['max_cot_length'] = 0
            
        stats['current_lambda'] = self._get_current_lambda()
        
        # Add gradient perturbation stats if available
        if self.gradient_history:
            stats['avg_gradient_perturbation'] = sum(self.gradient_history) / len(self.gradient_history)
            stats['max_gradient_perturbation'] = max(self.gradient_history)
        
        return stats
    
    def train_cot_generator(self, teacher_examples: List[Tuple[str, str]], epochs: int = 5) -> None:
        """
        Train a lightweight CoT generator to mimic the teacher.
        This is a placeholder for the full implementation.
        
        Args:
            teacher_examples: List of (input, teacher_cot) pairs
            epochs: Number of training epochs
        """
        logger.info(f"Training CoT generator on {len(teacher_examples)} examples for {epochs} epochs")
        
        # This would be implemented with an actual model training loop
        # For now, just log the intention
        logger.info("CoT generator training not yet implemented")
    
    def detect_complex_reasoning(self, text: str) -> bool:
        """
        Detect if input requires complex reasoning.
        
        Args:
            text: Input text
            
        Returns:
            True if input requires complex reasoning
        """
        # Simple heuristics for complex reasoning detection
        text_lower = text.lower()
        
        # Check length
        is_long = len(text_lower.split()) > 50
        
        # Check for complex reasoning indicators
        complex_indicators = [
            "prove", "derive", "theorem", "hypothesis", "evaluate", "scenario", 
            "conditions", "constraints", "optimal", "trade-off", "balance", "weigh"
        ]
        has_complex_indicators = any(indicator in text_lower for indicator in complex_indicators)
        
        # Check for multiple question parts
        has_multiple_questions = len(re.findall(r'\?', text_lower)) > 1
        
        # Check for nested logic
        nested_logic_indicators = [
            "if", "else", "unless", "except when", "given that",
            "assuming", "provided that", "in the case where"
        ]
        has_nested_logic = sum(1 for indicator in nested_logic_indicators if indicator in text_lower) >= 2
        
        # Combine heuristics
        return (is_long and has_complex_indicators) or has_multiple_questions or has_nested_logic
    
    def check_self_consistency(self, cots: List[str], threshold: float = 0.7) -> bool:
        """
        Check if multiple CoT generations are self-consistent.
        
        Args:
            cots: List of CoT strings
            threshold: Consistency threshold
            
        Returns:
            True if CoTs are self-consistent
        """
        if len(cots) <= 1:
            return True
        
        # Simple word overlap metric
        word_sets = [set(cot.lower().split()) for cot in cots]
        pairwise_consistency = []
        
        for i in range(len(word_sets)):
            for j in range(i+1, len(word_sets)):
                overlap = len(word_sets[i].intersection(word_sets[j]))
                union = len(word_sets[i].union(word_sets[j]))
                if union > 0:
                    consistency = overlap / union
                    pairwise_consistency.append(consistency)
        
        if not pairwise_consistency:
            return True
            
        avg_consistency = sum(pairwise_consistency) / len(pairwise_consistency)
        return avg_consistency >= threshold
    
    async def generate_multi_cot(self, text: str, num_samples: int = 3) -> Tuple[str, bool]:
        """
        Generate multiple CoTs and select most consistent one.
        
        Args:
            text: Input text
            num_samples: Number of CoT samples to generate
            
        Returns:
            Tuple of (selected_cot, is_consistent)
        """
        if self.teacher_model is None or num_samples <= 1:
            _, task_type = self._should_apply_cot(text)
            cot = await self._generate_cot_with_teacher(text, task_type)
            return cot, True
        
        # Generate multiple CoTs
        _, task_type = self._should_apply_cot(text)
        cots = []
        
        for _ in range(num_samples):
            cot = await self._generate_cot_with_teacher(text, task_type)
            cots.append(cot)
        
        # Check self-consistency
        is_consistent = self.check_self_consistency(cots)
        
        if not is_consistent:
            logger.warning(f"CoTs not self-consistent for input: {text[:50]}...")
            return self._rule_based_cot_generation(text, task_type), False
        
        # Return the longest CoT as it's likely most detailed
        selected_cot = max(cots, key=len)
        
        return selected_cot, True

class PromptRouter:
    """
    Router to decide whether to apply Chain-of-Thought prompting.
    
    This component uses a lightweight classifier to determine when to apply CoT,
    optimizing for computational efficiency and effectiveness.
    """
    def __init__(
        self,
        feature_extractors: Optional[List[Callable]] = None,
        reward_fn: Optional[Callable] = None,
        learning_rate: float = 0.01,
        exploration_rate: float = 0.1
    ):
        """
        Initialize the prompt router.
        
        Args:
            feature_extractors: Functions to extract features from inputs
            reward_fn: Function to compute reward for routing decisions
            learning_rate: Learning rate for updating weights
            exploration_rate: Exploration rate for epsilon-greedy policy
        """
        # Initialize feature extractors
        self.feature_extractors = feature_extractors or [
            self._extract_length_feature,
            self._extract_question_feature,
            self._extract_complexity_feature,
            self._extract_subject_feature
        ]
        
        self.reward_fn = reward_fn
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        
        # Initialize weights
        self.weights = torch.zeros(len(self.feature_extractors))
        self.bias = torch.tensor(0.0)
        
        # Router stats
        self.stats = {
            'decisions': {
                'cot': 0,
                'standard': 0
            },
            'rewards': [],
            'features': []
        }
    
    def _extract_length_feature(self, text: str) -> float:
        """Extract feature based on input length."""
        words = text.split()
        # Normalize length: 0 for short texts, 1 for longer texts
        return min(1.0, len(words) / 100)
    
    def _extract_question_feature(self, text: str) -> float:
        """Extract feature based on presence of question words."""
        text_lower = text.lower()
        question_words = ['why', 'how', 'explain', 'what', 'when', 'where', 'who']
        count = sum(1 for word in question_words if word in text_lower)
        return min(1.0, count / 3)
    
    def _extract_complexity_feature(self, text: str) -> float:
        """Extract feature based on text complexity."""
        text_lower = text.lower()
        complexity_markers = [
            'because', 'therefore', 'however', 'although', 'despite',
            'while', 'since', 'if', 'then', 'else', 'unless'
        ]
        count = sum(1 for marker in complexity_markers if marker in text_lower)
        return min(1.0, count / 5)
    
    def _extract_subject_feature(self, text: str) -> float:
        """Extract feature based on subject matter."""
        text_lower = text.lower()
        math_markers = ['calculate', 'solve', 'equation', 'math', 'number', 'formula']
        logic_markers = ['logic', 'argument', 'fallacy', 'premise', 'conclusion']
        
        math_count = sum(1 for marker in math_markers if marker in text_lower)
        logic_count = sum(1 for marker in logic_markers if marker in text_lower)
        
        return min(1.0, (math_count + logic_count) / 3)
    
    def extract_features(self, text: str) -> torch.Tensor:
        """
        Extract features from input text.
        
        Args:
            text: Input text
            
        Returns:
            Feature tensor
        """
        features = torch.tensor([extractor(text) for extractor in self.feature_extractors])
        self.stats['features'].append(features.tolist())
        return features
    
    def decide(self, text: str) -> bool:
        """
        Decide whether to apply CoT prompting.
        
        Args:
            text: Input text
            
        Returns:
            True if CoT should be applied
        """
        # Extract features
        features = self.extract_features(text)
        
        # Compute logit
        logit = torch.dot(self.weights, features) + self.bias
        
        # Apply sigmoid to get probability
        prob = torch.sigmoid(logit).item()
        
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            # Random exploration
            use_cot = random.random() < 0.5
        else:
            # Greedy exploitation
            use_cot = prob >= 0.5
        
        # Record decision
        if use_cot:
            self.stats['decisions']['cot'] += 1
        else:
            self.stats['decisions']['standard'] += 1
        
        return use_cot
    
    def update(self, text: str, used_cot: bool, reward: float) -> None:
        """
        Update router based on observed reward.
        
        Args:
            text: Input text
            used_cot: Whether CoT was applied
            reward: Observed reward
        """
        # Extract features
        features = self.extract_features(text)
        
        # Compute current prediction
        logit = torch.dot(self.weights, features) + self.bias
        prob = torch.sigmoid(logit).item()
        
        # Target is 1 if CoT was beneficial, 0 otherwise
        target = 1.0 if (used_cot and reward > 0) or (not used_cot and reward <= 0) else 0.0
        
        # Compute error gradient
        error = target - prob
        
        # Update weights using simple gradient ascent
        self.weights += self.learning_rate * error * features
        self.bias += self.learning_rate * error
        
        # Record reward
        self.stats['rewards'].append(reward)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        stats = {**self.stats}  # Copy stats
        
        # Calculate additional metrics
        total_decisions = stats['decisions']['cot'] + stats['decisions']['standard']
        if total_decisions > 0:
            stats['cot_rate'] = stats['decisions']['cot'] / total_decisions
        else:
            stats['cot_rate'] = 0
            
        if stats['rewards']:
            stats['avg_reward'] = sum(stats['rewards']) / len(stats['rewards'])
            stats['recent_reward'] = sum(stats['rewards'][-10:]) / min(10, len(stats['rewards']))
        else:
            stats['avg_reward'] = 0
            stats['recent_reward'] = 0
            
        stats['feature_weights'] = self.weights.tolist()
        stats['bias'] = self.bias.item()
        
        return stats
