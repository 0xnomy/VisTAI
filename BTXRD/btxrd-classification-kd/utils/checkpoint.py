import torch
import os


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, filepath, map_location='cpu'):
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Handle backbone. prefix from wrapped models
    if any(k.startswith('backbone.') for k in state_dict.keys()):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # Load with strict=False to ignore mismatches
    model.load_state_dict(state_dict, strict=False)
    
    return checkpoint.get('metrics', {})
