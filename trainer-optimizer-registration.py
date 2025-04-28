def _initialize_components(self):
    """Initialize the convergent MetaMorph components."""
    # 1. Initialize Enhanced Metaplasticity Optimizer
    if self.config.use_metaplasticity:
        self._initialize_optimizer()
    
    # 2. Initialize Enhanced Convergent Architecture Controller (after optimizer)
    if self.config.use_dynamic_architecture:
        self._initialize_architecture_controller()
        
        # Register optimizer with architecture controller for param group repair
        if 'optimizer' in self.components and 'architecture_controller' in self.components:
            self.components['architecture_controller'].register_optimizer(self.components['optimizer'])
            logger.info("Registered optimizer with architecture controller for parameter group repair")
    
    # 3. Initialize Information-Theoretic Activation Engineering
    if self.config.use_activation_engineering:
        self.components['activation_engineer'] = EnhancedActivationEngineering(
            self.model, 
            self.device,
            update_freq=self.config.activation_update_freq
        )
        logger.info("Initialized Enhanced Information-Theoretic Activation Engineering")
    
    # 4. Initialize mixed precision training
    if self.config.use_mixed_precision and torch.cuda.is_available():
        self.components['scaler'] = GradScaler()
        logger.info("Initialized Mixed Precision Training")
