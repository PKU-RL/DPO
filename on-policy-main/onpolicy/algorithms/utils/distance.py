import numpy as np
import torch

def Hellinger_distance(dist_p,dist_q,continuous=False):
    if not continuous:
        sqrt_p = torch.sqrt(dist_p)
        sqrt_q = torch.sqrt(dist_q)
        diff = sqrt_p - sqrt_q
        try:
            ret = torch.linalg.vector_norm(diff,ord = 2,dim = -1)
        except:
            ret = torch.norm(diff, p=2, dim=-1)
        ret = ret / np.sqrt(2)
        return ret
    else:
        eps_1 = 1e-9
        # eps_2 = 1e-12
        #dist_p.loc = mean or mu , dist_p.scale = std or sigma
        sum_sigma_square = dist_p.scale.pow(2) + dist_q.scale.pow(2)
        # print('sum_sigma_square = {}'.format(sum_sigma_square))
        mu_diff_square = (dist_p.loc - dist_q.loc).pow(2)
        # print('mu_diff_square = {}'.format(mu_diff_square))
        sigma_multi = dist_p.scale * dist_q.scale
        # print('sigma_multi = {}'.format(sigma_multi))
        coeff_part = torch.sqrt(2*sigma_multi/sum_sigma_square)
        # print('coeff_part = {}'.format(coeff_part))
        exp_part = torch.exp(-0.25 * mu_diff_square / sum_sigma_square)
        # print('exp_part = {}'.format(exp_part))
        ret_square = 1 - coeff_part * exp_part
        # print('ret_square = {}'.format(ret_square))
        ret_square_clip = torch.maximum(ret_square, (torch.ones_like(ret_square)*eps_1).to(ret_square.device) )
        # print('ret_square_clip = {}'.format(ret_square_clip))
        ret = torch.sqrt(ret_square_clip)
        # print('ret = {}'.format(ret))
        return ret
def Total_variance(dist_p, dist_q):
    diff = dist_p - dist_q
    try:
        ret = torch.linalg.vector_norm(diff,ord=1,dim = -1)
    except:
        ret = torch.norm(diff, p=1, dim=-1)
    ret = ret * 0.5
    return ret

def sqrt_TV(dist_p,dist_q):
    ret = Total_variance(dist_p,dist_q)
    eps_tensor = torch.ones_like(ret) * 1e-8
    return torch.sqrt(torch.max( ret + eps_tensor , eps_tensor) )

def sqrt_E(dist_p,dist_q):

    TV = Total_variance(dist_p,dist_q)
    ret = torch.exp(TV) - 1
    eps_tensor = torch.ones_like(ret) * 1e-8
    return torch.sqrt(torch.max( ret + eps_tensor), eps_tensor )






D_dict ={}
D_dict['H'] = Hellinger_distance
D_dict['TV'] = Total_variance
D_dict['sqrt_TV'] = sqrt_TV
D_dict['sqrt_E'] = sqrt_E