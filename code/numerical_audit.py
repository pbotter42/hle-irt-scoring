"""Memory-bounded accumulation for the author's EM and score-audit equations.

Item-wise derivatives are summed in blocks. The original arrowhead Newton
step, clipping, backtracking, iteration limit, and stopping tolerance are kept.
The likelihood, penalty, quadrature and EM model are unchanged.
"""
import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from transposed_extended import ExtendedTransposed as OriginalTransposed, score_value_grad, score_loss

@partial(jax.jit, static_argnames=('domains',))
def _bifactor_statistics(scores,nodes,weights,y,mask,membership,domains):
    # Same sufficient derivatives as the author's newton_scores kernel.
    a=jnp.exp(nodes[:,:,1]);c=nodes[:,:,2]
    groups=jax.nn.one_hot(membership,domains)
    s=scores[:,1:][:,membership].T
    z=a[:,:,None]*(scores[None,None,:,0]-nodes[:,:,0,None])+c[:,:,None]*s[:,None,:]
    prob=jax.nn.sigmoid(z)
    r=weights[:,:,None]*mask[:,None,:]*(prob-y[:,None,:])
    v=weights[:,:,None]*mask[:,None,:]*prob*(1-prob)
    loss=jnp.sum(weights[:,:,None]*mask[:,None,:]*(jnp.logaddexp(0.,z)-y[:,None,:]*z))
    gg=jnp.einsum('jkn,jk->n',r,a)
    hgg=jnp.einsum('jkn,jk->n',v,a*a)
    gs=jnp.einsum('jd,jkn,jk->nd',groups,r,c)
    hss=jnp.einsum('jd,jkn,jk->nd',groups,v,c*c)
    hgs=jnp.einsum('jd,jkn,jk->nd',groups,v,a*c)
    return loss,gg,hgg,gs,hss,hgs

def batched_newton_scores(scores,nodes,weights,y,mask,membership,penalty,family,domains):
    """The original arrowhead Newton algorithm, summing item blocks exactly."""
    if family!='bifactor':raise ValueError('This memory wrapper is only for the bifactor kernel')
    n=len(scores);p=len(nodes);slices=[slice(i,min(i+25,p)) for i in range(0,p,25)]
    fixed_tau=np.zeros(domains)
    def loss(x):
        value=.5*penalty*np.sum(x*x)
        for sl in slices:
            value+=float(score_loss(x.ravel(),nodes[sl],weights[sl],y[sl],mask[sl],membership[sl],
                         fixed_tau,0.,family=family,n=n,domains=domains))
        return value
    x=np.array(scores,copy=True)
    for iteration in range(7):
        total=None
        for sl in slices:
            batch=_bifactor_statistics(x,nodes[sl],weights[sl],y[sl],mask[sl],membership[sl],domains=domains)
            batch=[np.asarray(a) for a in batch]
            total=batch if total is None else [a+b for a,b in zip(total,batch)]
        old,gg,hgg,gs,hss,hgs=total
        old=float(old)+.5*penalty*np.sum(x*x)
        gg=gg+penalty*x[:,0];hgg=hgg+penalty
        gs=gs+penalty*x[:,1:];hss=hss+penalty
        dg=(gg-np.sum(hgs*gs/hss,axis=1))/(hgg-np.sum(hgs*hgs/hss,axis=1))
        ds=(gs-hgs*dg[:,None])/hss
        delta=np.column_stack((dg,ds))
        delta=delta/max(1.,np.max(abs(delta))/1.5)
        trial=np.clip(x-delta,-8.,8.);value=loss(trial);k=0
        while value>old and k<12:
            k+=1;trial=np.clip(x-(.5**k)*delta,-8.,8.);value=loss(trial)
        updated=trial if value<=old else x
        error=np.max(abs(updated-x));x=updated
        if error<=1e-6:break
    return x

def batched_score_residual(est, state, e, batch_size=25):
    v=np.r_[state['scores'].ravel(),state['tau']] if est.family=='testlet' else state['scores'].ravel()
    grad=np.zeros_like(v);batches=0
    for first in range(0,est.p,batch_size):
        stop=min(first+batch_size,est.p);batches+=1
        _,part=score_value_grad(v,e['nodes'][first:stop],e['weights'][first:stop],
          est.Y[first:stop],est.mask[first:stop],est.membership[first:stop],state['tau'],0.,
          family=est.family,n=est.n,domains=est.domains)
        grad+=np.asarray(part)
    cut=est.n*est.nf
    grad[:cut]+=est.settings.score_penalty*v[:cut]
    # The supplied score_loss includes the tau penalty even at score penalty 0.
    if est.family=='testlet':grad[cut:]-=(batches-1)*state['tau']
    grad[:cut][((v[:cut]<=-8+1e-7)&(grad[:cut]>0))|((v[:cut]>=8-1e-7)&(grad[:cut]<0))]=0.
    counts=np.bincount(est.membership,minlength=est.domains)
    divisors=np.ones((est.n,est.nf))*est.p
    if est.nf>1:divisors[:,1:]=counts
    residual=list(abs(grad[:cut])/divisors.ravel())
    postmean=np.einsum('jk,jkd->jd',e['weights'],e['nodes'])
    for d in range(est.domains):
        group=est.membership==d;count=counts[d]
        bg=(count*state['means'][d]-postmean[group,0].sum())/est.variance[0]+state['means'][d]/est.settings.mean_intercept_sd**2
        residual.append(abs(bg)/count)
        if est.family=='bifactor':tg=(count*state['tau'][d]-postmean[group,2].sum())/est.variance[2]+state['tau'][d]
        elif est.family=='testlet':tg=grad[cut+d]
        else:continue
        upper=20. if est.family=='testlet' else est.settings.bifactor_mean_upper
        if (state['tau'][d]<=1e-7 and tg>0) or (state['tau'][d]>=upper-1e-7 and tg<0):tg=0.
        residual.append(abs(tg)/count)
    return float(max(residual))

class MemoryBoundedTransposed(OriginalTransposed):
    def mstep(self,state,e):
        if self.family=='bifactor' and e['nodes'].shape[1]>500:
            # Execute the author's original M-step body; replace only its
            # Newton reduction kernel while this synchronous call is active.
            namespace=OriginalTransposed.mstep.__globals__
            original=namespace['newton_scores']
            namespace['newton_scores']=batched_newton_scores
            try:return super().mstep(state,e)
            finally:namespace['newton_scores']=original
        return super().mstep(state,e)

    def normalized_score(self,state,e):
        if e['nodes'].shape[1] > 500:
            return batched_score_residual(self,state,e)
        return super().normalized_score(state,e)
