

import torch


class PassAll:
    def __init__(self, policy, **kwargs):
        print('Standard Sampling')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        seed = kwargs.get('seed', None)
        if isinstance(seed, str):
            seed = eval(seed)

        self.generator = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(seed)

    def __call__(self, actions):
        B = actions.shape[0]  # Batch dimension
        P = actions.shape[1]  # Population dimension
        
        if self.generator is None:
            sampled_actions = actions[:,0]
        else:  
            indices = torch.randint(0, P, (B,), device=self.device, generator=self.generator)
            sampled_actions = actions[torch.arange(B, device=self.device), indices]

        return sampled_actions
    
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.generator is not None:
            state['generator_state'] = self.generator.get_state()
            del state['generator']
        return state

    def __setstate__(self, state):
        # If generator_state exists, use it to restore the exact state
        generator_state = state.pop('generator_state', None)
        self.__dict__.update(state)
        
        if generator_state is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.set_state(generator_state)