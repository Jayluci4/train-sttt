def _training_step(self, inputs):
    """Execute a single training step with all convergent components."""
    config = self.config
    
    # Check if we're accumulating gradients
    is_accumulating = (self.global_step + 1) % config.gradient_accumulation_steps != 0
    should_update = not is_accumulating
    
    # Zero gradients at the beginning of accumulation
    if not is_accumulating or self.global_step == 0:
        if 'optimizer' in self.components:
            self.components['optimizer'].zero_grad()
    
    # Mixed precision training
    use_mixed_precision = 'scaler' in self.components
    
    # Forward and backward pass with mixed precision
    if use_mixed_precision:
        with autocast():
            outputs = self.model(**inputs)
            loss = outputs.loss / config.gradient_accumulation_steps
        
        # Scale loss and backward
        self.components['scaler'].scale(loss).backward()
        
        # Update weights if needed
        if should_update:
            # Clip gradients
            if 'optimizer' in self.components:
                self.components['scaler'].unscale_(self.components['optimizer'])
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    config.max_grad_norm
                )
                
                # Update parameters
                self.components['scaler'].step(self.components['optimizer'])
                self.components['scaler'].update()
                
                # Update learning rate - MOVED OUTSIDE THE LOOP
                # Only update scheduler once per actual optimizer step, not per micro-batch
                # This fixes the learning rate scheduler drift issue
    else:
        # Standard training without mixed precision
        outputs = self.model(**inputs)
        loss = outputs.loss / config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights if needed
        if should_update and 'optimizer' in self.components:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                config.max_grad_norm
            )
            
            # Update parameters
            self.components['optimizer'].step()
    
    # Update learning rate scheduler OUTSIDE the gradient accumulation loop
    # This ensures the scheduler is stepped only once per REAL optimization step,
    # not once per micro-batch, fixing the LR scheduler drift issue
    if should_update and 'scheduler' in self.components:
        self.components['scheduler'].step()
    
    # Return full loss (not scaled by accumulation steps)
    return loss.item() * config.gradient_accumulation_steps
