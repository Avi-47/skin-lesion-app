import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.1):
    # Concatenate embeddings
    z = torch.cat([z1, z2], dim=0)
    
    z = F.normalize(z, dim=1)
    
    sim = torch.mm(z, z.T) / temperature
    
    N = z1.size(0)
    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
    pos_mask[torch.arange(N), torch.arange(N, 2*N)] = True
    pos_mask[torch.arange(N, 2*N), torch.arange(N)] = True
    
    pos_sim = sim[pos_mask].unsqueeze(1)
    neg_mask = ~torch.eye(2*N, dtype=torch.bool, device=z.device)
    neg_mask[pos_mask] = False
    neg_sim = torch.logsumexp(sim * neg_mask, dim=1, keepdim=True)
    loss = -(pos_sim - neg_sim).mean()
    return loss

# def nt_xent_loss(z1, z2, temperature=0.1):
#     z = torch.cat([z1, z2], dim=0)
#     sim = torch.mm(z, z.T) / temperature
    
#     mask = (~torch.eye(z.size(0), dtype=bool, device=z.device)).float()
    
#     pos = torch.cat([torch.diag(sim, z1.size(0)), 
#                     torch.diag(sim, -z1.size(0))])
    
#     neg = torch.logsumexp(sim * mask, dim=1)
    
#     return -(pos - neg).mean()