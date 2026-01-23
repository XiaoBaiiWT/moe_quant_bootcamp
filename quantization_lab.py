import torch
import torch.nn.functional as F
import pickle

class MoeQuantizer:
    """
    The Universal Surgical Tool for MoE Research.
    Provides methods to simulate low-bit quantization and inject noisy states 
    into live models for stability analysis.
    """
    @staticmethod
    def quantize(tensor,bits=4,method = 'symmetric',granularity='group_wise',group_size=128):
        """
        Args:
            tensor: Float tensor. Can be (M, N) or (Experts, M, N).
            bits: Target bit-width (e.g., 4 or 8).
            method: 'symmetric' (center 0) or 'asymmetric' (min-max).
            granularity: 'per_tensor', 'per_channel', 'per_expert', or 'group_wise'.
            group_size: Block size for group_wise quantization.
        """
        original_shape = tensor.shape
        ndim=tensor.ndim

        working_tensor=tensor.clone()
        pad_amount=0
      
        # 1. Reshaping & Padding
        if granularity == 'group_wise':
            in_feature = original_shape[-1]
            if in_feature%group_size != 0:
                pad_amount = group_size-(in_feature%group_size)
                working_tensor = F.pad(working_tensor,(0,pad_amount))
            # (M, N) -> (M, K, G) or (E, M, N) -> (E, M, K, G)
            current_shape = list(working_tensor.shape) # [M,N]
            new_shape = current_shape[:-1]+[current_shape[-1]//group_size,group_size] #[M,]+[N/G,G]
            calc_dim=-1
        
        elif granularity == 'per_channel':
            # For (M, N), we scale per M. Dim to reduce is N (dim 1).
            # For (E, M, N), we scale per M per E. Dim to reduce is N (dim 2).
            calc_dim = -1      

        elif granularity == 'per_expert':
            # (E, M, N) -> Scale shape (E, M*N)
            if ndim!=3:
                raise ValueError('Granularity "per_expert" requires a 3D tensor')
            working_tensor = working_tensor.view(original_shape[0],-1)
            calc_dim = -1
        
        elif granularity == 'per_tensor':
            working_tensor = working_tensor.view(-1)
            calc_dim=-1
        
        # 2. Setup Bit-Limits
        # Signed Integers [-8, 7] for BOTH Symmetric and Asymmetric
        q_min = -(2**(bits - 1))      
        q_max = (2**(bits - 1)) - 1
        
        # 3. Calculate Scale & Zero Point
        if method == 'symmetric':
            abs_max = working_tensor.abs().amax(dim=calc_dim,keepdim=True)
            scale = abs_max/q_max
            zp = 0.0
        else:
            min_val = working_tensor.amin(dim=calc_dim,keepdim=True)
            max_val = working_tensor.amax(dim=calc_dim,keepdim=True)

            scale = (max_val-min_val)/(q_max-q_min)
            zp = q_min - torch.round(min_val / scale)
            
        scale = scale.clamp(min=1e-5) # Safety buffer

        # 4. (Quantize -> Dequantize)
        q_tensor = torch.round((working_tensor/scale)+zp).clamp(q_min,q_max)
        dq_tensor = (q_tensor-zp)*scale

        # 5. UNIVERSAL SNAP-BACK (Robust Fix)
        # Flatten everything to clear the quantization groups/dimensions
        dq_tensor = dq_tensor.reshape(-1)

        # Handle Padding Strip
        if pad_amount > 0:
            # First, reshape to the PADDED version of the original rows
            # This ensures we slice off the end of each feature row correctly
            padded_total_width = original_shape[-1] + pad_amount #[M,N] [N]+[pad_amount]
            temp_shape = list(original_shape[:-1]) + [padded_total_width] #[M]+[N+pad_amount]
            dq_tensor = dq_tensor.view(temp_shape) #[M,N+pad_amount]
            # Slice off the extra padding
            dq_tensor = dq_tensor[..., :original_shape[-1]] #[..., ]
        
        # Final Step: Force it back to exactly what it was when it entered
        return dq_tensor.reshape(original_shape)    
    
    @classmethod
    def inject_quantisation(cls,model,state_path,expert_bits=4,router_bits=8,
                            method='asymmetric',granularity='group_wise',group_size=128):
        """
        Loads a 'healthy' brain state, applies specific quantization noise, 
        and performs a 'surgical' weight transplant into the live model.

        Args:
            model: The live pytorch MoeLayer object.
            state_path: Path to the .pkl file containing the healthy weights.
            expert_bits: Bit-width for experts (Target of the experiment).
            router_bits: Bit-width for router (Usually higher to prevent control collapse).
        """
        print('---Surgical Injection Started---')   
        print(f'Target: Experts ({expert_bits}-bit {granularity}), Router ({router_bits}-bit)')     
        
        # 1. Load moe_brain.pkl
        with open(state_path,'rb') as f:
            state = pickle.load(f)

        # 2. Inject Router
        # (8-bits) or (16/32fp)
        if router_bits<16:
            noisy_router = cls.quantize(
                state['router']['w'],
                bits=router_bits,
                method='symmetric',
                granularity='per_tensor'
            )
            model.router.gate.weight.data = noisy_router
        else:
            model.router.gate.weight.data = state['router']['w']
        model.router.gate.bias.data = state['router']['b']

        # 3. Inject Experts
        for i,expert_state in enumerate(state['experts']):
            # Quantize Layer 1 
            w1_noisy = cls.quantize(
                expert_state['w1'],
                bits=expert_bits,
                method=method,
                granularity=granularity,
                group_size=group_size
                )
            model.experts[i].layer1.weight.data = w1_noisy
            model.experts[i].layer1.bias.data = expert_state['b1']

            # Quantize Layer 2
            w2_noisy=cls.quantize(
                expert_state['w2'],
                bits=expert_bits,
                method=method,
                granularity=granularity,
                group_size=group_size
            )
            model.experts[i].layer2.weight.data = w2_noisy
            model.experts[i].layer2.bias.data = expert_state['b2']
        print('---Injection Complete. Model is ready for audit---')
        return model


