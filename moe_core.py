import torch
import torch.nn as nn
import torch.nn.functional as F

def relu(x):
    return torch.maximum(torch.tensor(0.0),x) 

def relu_backward(relu_input,grad_output):
    #dL/dz = dL/dRelu x dRelu/dz
    return (relu_input>0).float()*grad_output

class LinearLayer(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        limit = (6/input_dim)**0.5 #Kaiming Initialization
        self.weight = nn.Parameter((torch.rand(output_dim, input_dim) * 2 * limit - limit).requires_grad_(True))
        self.bias = nn.Parameter(torch.zeros(output_dim,requires_grad=True))
        self.weight_grad = None
        self.bias_grad = None
    
    def __repr__(self):
        return f'LinearLayer(in={self.w.shape[1]},out={self.w.shape[0]})'
    
    def forward(self,x):
        return x@self.weight.T + self.bias
    
    def backward(self, grad_output, x):
        self.weight_grad = grad_output.T@x #[Batch_size, output_dim]^T x [Batch_size, input_dim]
        self.bias_grad = grad_output.sum(dim=0)
        return grad_output@self.w
    
    def update(self, learning_rate=0.01, weight_decay = 1e-4, max_grad_norm=1.0):
        if self.weight_grad is not None:
            self.weight_grad += weight_decay*self.weight #L2 Regularisation
            norm = torch.norm(self.weight_grad)
            if norm>max_grad_norm: # Gradient Clipping
                self.weight_grad = self.weight_grad*(max_grad_norm/(norm+1e-6))
            self.weight -= learning_rate*self.weight_grad

        if self.bias_grad is not None:
            self.bias -= learning_rate*self.bias_grad
        self.weight_grad, self.bias_grad = None,None

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = LinearLayer(input_dim, hidden_dim)
        self.layer2 = LinearLayer(hidden_dim, output_dim)
        self.cache = {}

    def forward(self, x):
        self.cache['x'] = x
        z1 = self.layer1.forward(x)
        self.cache['z1'] = z1
        h1 = relu(z1)
        self.cache['h1'] = h1
        return self.layer2.forward(h1)
    
    def backward(self, grad_output):
        grad_h1 = self.layer2.backward(grad_output, self.cache['h1'])
        grad_z1 = relu_backward(self.cache['z1'], grad_h1)
        grad_x = self.layer1.backward(grad_z1, self.cache['x'])
        return grad_x

    def update(self, learning_rate=0.01):
        self.layer1.update(learning_rate)
        self.layer2.update(learning_rate)
        self.cache = {}

class Router(nn.Module):
    def __init__(self,input_dim, num_expert,temperature=1.0):
        super().__init__()
        self.gate = LinearLayer(input_dim,num_expert)
        self.temperature = temperature
        self.cache = {}

    def forward(self,x,top_k=2):
        self.cache['x'] = x
        logits = self.gate.forward(x) #[Batch_size,num_expert]
        logits = logits/self.temperature

        #Top-K Logic
        values,indices = torch.topk(logits,top_k,dim=-1) #[batch_size,top_k]
        # sparse_logits = torch.full_like(logits,float('-inf')) #[batch_size,num_expert]
        # sparse_logits.scatter_(-1,indices,values)
        weights = F.softmax(values,dim=-1)
        self.cache['weight'] = weights
        self.cache['indices'] = indices
        return weights, indices

    def backward(self,grad_output):
        sum_grad_weight = torch.sum(self.cache['weight']*grad_output,dim=-1,keepdim=True)
        grad_experts = self.cache['weight']*(grad_output-sum_grad_weight)
        grad_logits = grad_experts/self.temperature
        return self.gate.backward(grad_logits,self.cache['x'])
    
    def update(self,lr):
        self.gate.update(lr)
        self.cache = {}
    
    def gumbel_routing(self,x):
        logits = self.gate.forward(x) #[Batch_size,num_experts]
        unif = torch.rand_like(logits)

        gumbel_noise = -torch.log(-torch.log(unif+1e-6)+1e-6)
        return F.softmax((logits+gumbel_noise)/self.temperature,dim=-1)
    
class MoeLayer(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_experts=8):
        super().__init__()
        self.router = Router(input_dim,num_experts)
        self.experts = nn.ModuleList([Expert(input_dim,hidden_dim,input_dim) for _ in range(num_experts)])
        self.num_experts = num_experts
        self.cache = {}

    def forward(self,x,top_k=2):
        weights, indices = self.router.forward(x,top_k) #[Batch_size,top_k]
        final_output = torch.zeros_like(x) #[Batch_size,output_dim(input_dim)]
        self.cache['weights']= weights
        self.cache['indices'] = indices
        self.cache['expert_outs'] = {}
        for i, expert in enumerate(self.experts):
            mask = (indices==i).any(-1) #[Batch_size,]
            if mask.any():
                expert_outs = expert.forward(x[mask]) #[num_selected,output_dim(input_dim)]  
                self.cache['expert_outs'][i] = expert_outs
                expert_pos = (indices[mask]==i)                
                token_weight = weights[mask][expert_pos].unsqueeze(1) # weights[batch_size,top_k], w1=weights[mask] --> [num_selected,top_k] --> w1[expert_loc_expert]=[num_selected]
                final_output[mask] += (expert_outs*token_weight) #softmax_robability/weights*logits_topk
        return final_output, weights, indices
    
    def backward(self,grad_output):
        grad_router = torch.zeros_like(self.cache['weights'])
        for i, expert in enumerate(self.experts):
            mask = (self.cache['indices'] == i).any(dim=-1)
            if mask.any():
                pos_mask = (self.cache['indices'][mask]==i) 
                w = self.cache['weights'][mask][pos_mask].unsqueeze(dim=-1)
                expert.backward(grad_output[mask]*w)
                grad_router[mask][pos_mask] = (grad_output[mask]*self.cache['expert_outs'][i]).sum(dim=-1)
        self.router.backward(grad_router)
    
    def update(self,lr):
        self.router.update(lr)
        for e in self.experts:
            e.update(lr)
        self.cache = {}

    def load_balancing_loss(self,weights,indices):
        # weights: [batch, top_k]
        # indices: [batch, top_k]
        batch_size = weights.shape[0]
        # 1. How many tokens actually went to each expert? (Fraction f)
        # We flatten the indices and count occurrences
        counts = torch.bincount(indices.view(-1),minlength=self.num_experts).float()
        f = counts/counts.sum() #[num_experts,]
    
        # 2. What was the average probability assigned to each expert? (Probability P)
        # We take the mean of the weights the router assigned
        # Note: We need to map the weights back to their expert IDs
         # Normalize to a distribution
        full_probs = torch.zeros(batch_size,self.num_experts,device=weights.device)
        full_probs.scatter_(-1, indices,weights)
        p = full_probs.mean(dim=0) #[num_experts,]
        aux_loss = self.num_experts*torch.sum(f*p)  
        return aux_loss

    def sync_gradients(self, max_grad_norm=1.0, clip=True):
        """
        Moves manual gradients into PyTorch .grad slots for Adam.
        Optionally applies Manual Gradient Clipping per-tensor during the transfer.
        """
        
        # Helper function to keep code clean
        def _transfer(manual_grad, param):
            if manual_grad is not None:
                # 1. Apply Manual Clipping (Safety Brakes)
                if clip:
                    norm = torch.norm(manual_grad)
                    if norm > max_grad_norm:
                        # Rescale the gradient vector
                        scale_factor = max_grad_norm / (norm + 1e-6)
                        manual_grad = manual_grad * scale_factor
                
                # 2. The Hand-off to PyTorch
                param.grad = manual_grad.clone()

        # --- Sync Experts ---
        for expert in self.experts:
            # Layer 1
            _transfer(expert.layer1.weight_grad, expert.layer1.weight)
            _transfer(expert.layer1.bias_grad,   expert.layer1.bias)
            # Layer 2
            _transfer(expert.layer2.weight_grad, expert.layer2.weight)
            _transfer(expert.layer2.bias_grad,   expert.layer2.bias)
        
        # --- Sync Router ---
        _transfer(self.router.gate.weight_grad, self.router.gate.weight)
        _transfer(self.router.gate.bias_grad,   self.router.gate.bias)
        

