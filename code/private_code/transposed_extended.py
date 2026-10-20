"""Regularized item-marginal estimation for U2PL, bifactor, and testlet/HO.

The item vector, never the system score vector, is integrated out. System
coordinates have an explicit quadratic penalty. This is a research extension,
not the unpenalized estimator or asymptotic theorem in the source manuscript.
"""
from dataclasses import dataclass, asdict
from functools import partial
from itertools import product
import time
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import logit
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@dataclass
class Settings:
    points: int = 5
    max_iterations: int = 180
    tolerance: float = 1e-4
    score_tolerance: float = 1e-3
    parameter_tolerance: float = 1e-2
    prediction_tolerance: float = 2e-3
    score_penalty: float = 1.0
    intercept_sd: float = 2.0
    log_slope_sd: float = 0.5
    specific_sd: float = 0.75
    bifactor_mean_upper: float = 4.0
    mean_intercept_sd: float = 2.0
    mode_iterations: int = 20


def _kernel(eta, y, mask, g, s, tau, mean, variance, family):
    a = jnp.exp(eta[1])
    c = eta[2] if family == "bifactor" else (a*tau if family == "testlet" else 0.)
    z = a*(g-eta[0]) + c*s
    return jnp.sum(mask*(jnp.logaddexp(0., z)-y*z)) + .5*jnp.sum((eta-mean)**2/variance)


@partial(jax.jit, static_argnames=("family", "steps"))
def item_modes(initial, y, mask, g, s, tau, mean, variance, family, steps=20):
    f = partial(_kernel, family=family)
    vf = jax.vmap(f, in_axes=(0,0,0,None,0,0,0,None))
    vg = jax.vmap(jax.grad(f), in_axes=(0,0,0,None,0,0,0,None))
    vh = jax.vmap(jax.hessian(f), in_axes=(0,0,0,None,0,0,0,None))
    def advance(_, eta):
        grad = vg(eta,y,mask,g,s,tau,mean,variance)
        h = vh(eta,y,mask,g,s,tau,mean,variance)
        mineig = jnp.linalg.eigvalsh(h)[:,0]
        h = h + jnp.maximum(1e-4-mineig,0.)[:,None,None]*jnp.eye(h.shape[-1])
        step = jnp.linalg.solve(h,grad[...,None])[...,0]
        step = step/jnp.maximum(1.,jnp.max(jnp.abs(step),axis=1)/1.5)[:,None]
        old = vf(eta,y,mask,g,s,tau,mean,variance)
        def backtrack(k, state):
            chosen, done = state
            trial = eta-(.5**k)*step
            value = vf(trial,y,mask,g,s,tau,mean,variance)
            # At an almost stationary mode, the true Newton improvement can
            # be below floating-point resolution of the summed log likelihood.
            use = (~done)&(value<=old+1e-12)
            return jnp.where(use[:,None],trial,chosen), done|use
        return jax.lax.fori_loop(0,12,backtrack,(eta,jnp.zeros(len(eta),dtype=bool)))[0]
    def more(state):return (state[0]<steps)&(state[2]>1e-8)
    def iterate(state):
        updated=advance(0,state[1])
        error=jnp.max(jnp.abs(vg(updated,y,mask,g,s,tau,mean,variance)))
        return state[0]+1,updated,error
    eta=jax.lax.while_loop(more,iterate,(0,initial,jnp.array(jnp.inf)))[1]
    return eta, vh(eta,y,mask,g,s,tau,mean,variance), vg(eta,y,mask,g,s,tau,mean,variance)


def gh_grid(points, dim):
    z,w = hermgauss(points)
    indices = np.array(list(product(range(points),repeat=dim)))
    grid = z[indices]
    logweights = np.log(w[indices]).sum(1)+np.sum(grid*grid,axis=1)
    return grid,logweights


@partial(jax.jit, static_argnames=("family",))
def integrate(modes, hessians, y, mask, g, s, tau, mean, variance, grid, logw, family):
    # hessian = L L'; the covariance square root is L^{-T}.
    root = jnp.linalg.cholesky(hessians)
    eye = jnp.broadcast_to(jnp.eye(modes.shape[1]),root.shape)
    invroot = jnp.swapaxes(jnp.linalg.solve(root,eye),1,2)
    nodes = modes[:,None,:]+jnp.sqrt(2.)*jnp.einsum("kd,jed->jke",grid,invroot)
    a = jnp.exp(nodes[:,:,1])
    c = nodes[:,:,2] if family=="bifactor" else (a*tau[:,None] if family=="testlet" else jnp.zeros_like(a))
    z = a[:,:,None]*(g[None,None,:]-nodes[:,:,0,None])+c[:,:,None]*s[:,None,:]
    loglik = jnp.sum(mask[:,None,:]*(y[:,None,:]*z-jnp.logaddexp(0.,z)),axis=2)
    logprior = -.5*jnp.sum((nodes-mean[:,None,:])**2/variance,axis=2)-.5*jnp.sum(jnp.log(2*jnp.pi*variance))
    terms = loglik+logprior+logw[None,:]
    norm = logsumexp(terms,axis=1)
    logint = norm+.5*modes.shape[1]*jnp.log(2.)-jnp.log(jnp.diagonal(root,axis1=1,axis2=2)).sum(1)
    posterior = jnp.exp(terms-norm[:,None])
    pred = jnp.sum(posterior[:,:,None]*jax.nn.sigmoid(z),axis=1)
    return logint,nodes,posterior,pred


@partial(jax.jit, static_argnames=("family","n","domains"))
def score_loss(v, nodes, weights, y, mask, membership, fixed_tau, penalty, family, n, domains):
    nf = 1 if family=="unidimensional" else domains+1
    scores = v[:n*nf].reshape(n,nf)
    a = jnp.exp(nodes[:,:,1])
    if family=="unidimensional":
        c = jnp.zeros_like(a); s = jnp.zeros((len(nodes),n)); tau_penalty=0.
    else:
        s = scores[:,1:][:,membership].T
        if family=="testlet":
            tau = v[n*nf:]; c=a*tau[membership,None]; tau_penalty=.5*jnp.sum(tau*tau)
        else:
            c=nodes[:,:,2]; tau_penalty=0.
    z = a[:,:,None]*(scores[None,None,:,0]-nodes[:,:,0,None])+c[:,:,None]*s[:,None,:]
    loss = jnp.sum(weights[:,:,None]*mask[:,None,:]*(jnp.logaddexp(0.,z)-y[:,None,:]*z))
    return loss+.5*penalty*jnp.sum(scores*scores)+tau_penalty


score_value_grad = jax.jit(jax.value_and_grad(score_loss),static_argnames=("family","n","domains"))


@partial(jax.jit, static_argnames=("family","domains"))
def newton_scores(scores,nodes,weights,y,mask,membership,penalty,family,domains):
    """Exact arrowhead Hessian: clusters do not share a specific coordinate."""
    a=jnp.exp(nodes[:,:,1]);c=nodes[:,:,2] if family=='bifactor' else jnp.zeros_like(a)
    groups=jax.nn.one_hot(membership,domains)
    def loss(x):
        s=x[:,1:][:,membership].T if family=='bifactor' else jnp.zeros_like(y)
        z=a[:,:,None]*(x[None,None,:,0]-nodes[:,:,0,None])+c[:,:,None]*s[:,None,:]
        return jnp.sum(weights[:,:,None]*mask[:,None,:]*(jnp.logaddexp(0.,z)-y[:,None,:]*z))+.5*penalty*jnp.sum(x*x)
    def step(_,x):
        s=x[:,1:][:,membership].T if family=='bifactor' else jnp.zeros_like(y)
        z=a[:,:,None]*(x[None,None,:,0]-nodes[:,:,0,None])+c[:,:,None]*s[:,None,:]
        prob=jax.nn.sigmoid(z);r=weights[:,:,None]*mask[:,None,:]*(prob-y[:,None,:])
        v=weights[:,:,None]*mask[:,None,:]*prob*(1-prob)
        gg=jnp.einsum('jkn,jk->n',r,a)+penalty*x[:,0]
        hgg=jnp.einsum('jkn,jk->n',v,a*a)+penalty
        if family=='bifactor':
            gs=jnp.einsum('jd,jkn,jk->nd',groups,r,c)+penalty*x[:,1:]
            hss=jnp.einsum('jd,jkn,jk->nd',groups,v,c*c)+penalty
            hgs=jnp.einsum('jd,jkn,jk->nd',groups,v,a*c)
            dg=(gg-jnp.sum(hgs*gs/hss,axis=1))/(hgg-jnp.sum(hgs*hgs/hss,axis=1))
            ds=(gs-hgs*dg[:,None])/hss
            delta=jnp.column_stack((dg,ds))
        else:delta=(gg/hgg)[:,None]
        delta=delta/jnp.maximum(1.,jnp.max(jnp.abs(delta))/1.5)
        old=loss(x)
        def cond(state):return (state[2]>old)&(state[0]<12)
        def back(state):
            k=state[0]+1;trial=jnp.clip(x-(.5**k)*delta,-8.,8.)
            return k,trial,loss(trial)
        trial=jnp.clip(x-delta,-8.,8.)
        _,trial,value=jax.lax.while_loop(cond,back,(0,trial,loss(trial)))
        return jnp.where(value<=old,trial,x)
    def more(state):return (state[0]<7)&(state[2]>1e-6)
    def advance(state):
        updated=step(0,state[1])
        return state[0]+1,updated,jnp.max(jnp.abs(updated-state[1]))
    return jax.lax.while_loop(more,advance,(0,scores,jnp.array(jnp.inf)))[1]


@partial(jax.jit, static_argnames=("domains",))
def newton_testlet(scores,tau,nodes,weights,y,mask,membership,penalty,domains):
    """Damped joint Newton update via a K-by-K Schur complement.

    The system-specific Hessian blocks are (K+1)-by-(K+1). Only the K
    shared testlet coefficients couple them. Includes the residual term in
    the mixed derivative of a_j*tau_d*s_id; omitting it is not exact Newton.
    """
    a=jnp.exp(nodes[:,:,1]);groups=jax.nn.one_hot(membership,domains)
    def loss(x,t):
        v=x[:,0,None]+x[:,1:]*t
        z=a[:,:,None]*(v[:,membership].T[:,None,:]-nodes[:,:,0,None])
        return jnp.sum(weights[:,:,None]*mask[:,None,:]*(jnp.logaddexp(0.,z)-y[:,None,:]*z))+.5*penalty*jnp.sum(x*x)+.5*jnp.sum(t*t)
    def step(x,t):
        v=x[:,0,None]+x[:,1:]*t
        z=a[:,:,None]*(v[:,membership].T[:,None,:]-nodes[:,:,0,None]);prob=jax.nn.sigmoid(z)
        residual=weights[:,:,None]*mask[:,None,:]*(prob-y[:,None,:])
        variance=weights[:,:,None]*mask[:,None,:]*prob*(1-prob)
        rr=jnp.einsum('jd,jkn,jk->nd',groups,residual,a)
        vv=jnp.einsum('jd,jkn,jk->nd',groups,variance,a*a)
        gx=jnp.column_stack((rr.sum(1)+penalty*x[:,0],rr*t+penalty*x[:,1:]))
        gt=jnp.sum(rr*x[:,1:],axis=0)+t
        nn=x.shape[0];nf=domains+1;dd=jnp.arange(domains)
        aa=jnp.zeros((nn,nf,nf));aa=aa.at[:,0,0].set(vv.sum(1)+penalty)
        aa=aa.at[:,dd+1,dd+1].set(vv*t*t+penalty)
        aa=aa.at[:,0,1:].set(vv*t);aa=aa.at[:,1:,0].set(vv*t)
        bb=jnp.zeros((nn,nf,domains));bb=bb.at[:,0,:].set(vv*x[:,1:])
        bb=bb.at[:,dd+1,dd].set(vv*t*x[:,1:]+rr)
        invag=jnp.linalg.solve(aa,gx[...,None])[...,0]
        invab=jnp.linalg.solve(aa,bb)
        schur=jnp.diag(jnp.sum(vv*x[:,1:]**2,axis=0)+1.)-jnp.einsum('nfd,nfe->de',bb,invab)
        rhs=gt-jnp.einsum('nfd,nf->d',bb,invag)
        damping=jnp.maximum(.05-jnp.linalg.eigvalsh(schur)[0],0.)
        dt=jnp.linalg.solve(schur+damping*jnp.eye(domains),rhs)
        dx=invag-jnp.einsum('nfd,d->nf',invab,dt)
        scale=jnp.maximum(1.,jnp.maximum(jnp.max(abs(dx)),jnp.max(abs(dt)))/1.5)
        dx,dt=dx/scale,dt/scale;old=loss(x,t)
        def cond(s):return (s[3]>old)&(s[0]<14)
        def back(s):
            k=s[0]+1;xx=jnp.clip(x-(.5**k)*dx,-8.,8.);tt=jnp.clip(t-(.5**k)*dt,0.,20.)
            return k,xx,tt,loss(xx,tt)
        xx=jnp.clip(x-dx,-8.,8.);tt=jnp.clip(t-dt,0.,20.)
        _,xx,tt,value=jax.lax.while_loop(cond,back,(0,xx,tt,loss(xx,tt)))
        return jnp.where(value<=old,xx,x),jnp.where(value<=old,tt,t)
    def more(s):return (s[0]<10)&(s[3]>1e-6)
    def advance(s):
        xx,tt=step(s[1],s[2]);change=jnp.maximum(jnp.max(abs(xx-s[1])),jnp.max(abs(tt-s[2])))
        return s[0]+1,xx,tt,change
    result=jax.lax.while_loop(more,advance,(0,scores,tau,jnp.array(jnp.inf)))
    return result[1],result[2]


class ExtendedTransposed:
    def __init__(self,y,membership,family,settings=None):
        if family not in ("unidimensional","bifactor","testlet"):
            raise ValueError("unknown family")
        self.family=family; self.settings=settings or Settings()
        self.Y=np.asarray(y,float).T; self.mask=np.isfinite(self.Y).astype(float)
        if not np.all(np.isin(self.Y[np.isfinite(self.Y)],[0.,1.])):
            raise ValueError("responses must be binary or missing")
        self.Y=np.nan_to_num(self.Y)
        self.membership=np.asarray(membership,int)
        self.p,self.n=self.Y.shape; self.domains=int(self.membership.max())+1
        if len(self.membership)!=self.p or set(self.membership)!=set(range(self.domains)):
            raise ValueError("domains must be contiguous nonnegative integers")
        self.nf=1 if family=="unidimensional" else self.domains+1
        self.dim=3 if family=="bifactor" else 2
        st=self.settings
        self.variance=np.array([st.intercept_sd**2,st.log_slope_sd**2]+([st.specific_sd**2] if self.dim==3 else []))
        self.grid,self.logw=gh_grid(st.points,self.dim)

    def start(self,seed):
        rng=np.random.default_rng(seed)
        raw=(self.Y*self.mask).sum(0)/np.maximum(self.mask.sum(0),1)
        g=logit(np.clip(raw,.025,.975)); g-=g.mean()
        scores=np.zeros((self.n,self.nf)); scores[:,0]=g
        if self.nf>1:
            for d in range(self.domains):
                group=self.membership==d
                rawd=(self.Y[group]*self.mask[group]).sum(0)/np.maximum(self.mask[group].sum(0),1)
                scores[:,d+1]=.5*(logit(np.clip(rawd,.025,.975))-g)
        if seed:
            scores+=rng.normal(0,.45,scores.shape)
        tau=np.ones(self.domains)*.7
        if seed: tau=np.exp(rng.normal(-.35,.25,self.domains))
        means=np.array([-logit(np.clip(self.Y[self.membership==d].mean(),.025,.975)) for d in range(self.domains)])
        if seed: means+=rng.normal(0,.3,self.domains)
        modes=np.zeros((self.p,self.dim))
        modes[:,0]=-logit(np.clip(self.Y.sum(1)/np.maximum(self.mask.sum(1),1),.02,.98))
        if self.dim==3: modes[:,2]=tau[self.membership]
        return dict(scores=scores,tau=tau,means=means,modes=modes)

    def estep(self,state,points=None):
        scores=state['scores']; tau=state['tau']; means=state['means']
        s=scores[:,1:][:,self.membership].T if self.nf>1 else np.zeros((self.p,self.n))
        popmean=np.zeros((self.p,self.dim)); popmean[:,0]=means[self.membership]
        if self.dim==3: popmean[:,2]=tau[self.membership]
        arguments=(self.Y,self.mask,scores[:,0],s,tau[self.membership],popmean,self.variance)
        modes,hess,grad=item_modes(state['modes'],*arguments,family=self.family,steps=self.settings.mode_iterations)
        hess=np.asarray(hess)
        if not np.isfinite(hess).all() or np.min(np.linalg.eigvalsh(hess))<=0:
            raise FloatingPointError("non-positive item posterior Hessian")
        grid,logw=(self.grid,self.logw) if points is None else gh_grid(points,self.dim)
        if len(grid)>500:
            # Keep high-order audit rules from constructing a full p*Q^k*n
            # tensor in memory. Item independence makes this batching exact.
            chunks=[]
            for first in range(0,self.p,25):
                last=min(first+25,self.p)
                batchargs=(self.Y[first:last],self.mask[first:last],scores[:,0],s[first:last],
                           tau[self.membership[first:last]],popmean[first:last],self.variance)
                result=integrate(modes[first:last],hess[first:last],*batchargs,grid,logw,family=self.family)
                chunks.append(tuple(np.asarray(a) for a in result))
            logint,nodes,weights,pred=(np.concatenate([chunk[k] for chunk in chunks],axis=0) for k in range(4))
        else:
            logint,nodes,weights,pred=integrate(modes,hess,*arguments,grid,logw,family=self.family)
        penalty=.5*self.settings.score_penalty*np.sum(scores**2)+.5*np.sum(means**2)/self.settings.mean_intercept_sd**2
        if self.nf>1: penalty+=.5*np.sum(tau**2)
        return dict(objective=float(np.sum(logint))-penalty,loglik=float(np.sum(logint)),
                    modes=np.asarray(modes),hessians=hess,nodes=np.asarray(nodes),weights=np.asarray(weights),
                    prediction=np.asarray(pred).T,mode_gradient=float(np.max(np.abs(grad))))

    def mstep(self,state,e):
        v=np.r_[state['scores'].ravel(),state['tau']] if self.family=='testlet' else state['scores'].ravel()
        def fun(v):
            val,grad=score_value_grad(v,e['nodes'],e['weights'],self.Y,self.mask,self.membership,state['tau'],
               self.settings.score_penalty,family=self.family,n=self.n,domains=self.domains)
            return float(val),np.asarray(grad)
        bounds=[(-8.,8.)]*(self.n*self.nf)
        if self.family=='testlet': bounds += [(0.,20.)]*self.domains
        if self.family=='testlet':
            scores,tau=newton_testlet(state['scores'],state['tau'],e['nodes'],e['weights'],self.Y,self.mask,
                        self.membership,self.settings.score_penalty,domains=self.domains)
            scores,tau=np.asarray(scores),np.asarray(tau);m_success=True
            updated=np.r_[scores.ravel(),tau]
            oldloss,_=fun(v);newloss,newgrad=fun(updated)
            if (not np.isfinite(newloss)) or (oldloss-newloss<1e-8 and np.max(abs(newgrad))/self.p>1e-3):
                opt=minimize(fun,updated,jac=True,method='L-BFGS-B',bounds=bounds,
                    options=dict(maxiter=65,ftol=1e-11,gtol=1e-5,maxls=25))
                scores=opt.x[:self.n*self.nf].reshape(self.n,self.nf);tau=opt.x[self.n*self.nf:];m_success=bool(opt.success)
        else:
            scores=np.asarray(newton_scores(state['scores'],e['nodes'],e['weights'],self.Y,self.mask,
                     self.membership,self.settings.score_penalty,family=self.family,domains=self.domains))
            tau=state['tau'].copy();m_success=True
        means=np.zeros(self.domains)
        postmean=np.einsum('jk,jkd->jd',e['weights'],e['nodes'])
        for d in range(self.domains):
            group=self.membership==d; count=group.sum()
            means[d]=postmean[group,0].sum()/(count+self.variance[0]/self.settings.mean_intercept_sd**2)
            if self.family=='bifactor':
                tau[d]=np.clip(postmean[group,2].sum()/(count+self.variance[2]),0.,self.settings.bifactor_mean_upper)
        # Profile the exact joint location invariance. It improves the penalty
        # without changing a single conditional response probability or prior
        # residual; this is the location transform used by the source method.
        scores=np.array(scores,copy=True);modes=e['modes'].copy()
        gamma=self.settings.score_penalty;vm=self.settings.mean_intercept_sd**2
        shift=-(gamma*scores[:,0].sum()+means.sum()/vm)/(gamma*self.n+self.domains/vm)
        if np.max(np.abs(scores[:,0]+shift))<8.:
            scores[:,0]+=shift;means+=shift;modes[:,0]+=shift
        if self.family=='testlet' and np.min(tau)>1e-6:
            # Profile likelihood-invariant directions. Each step minimizes
            # the stated quadratic penalty while preserving v_id-b_j.
            tau=np.array(tau,copy=True)
            for _ in range(4):
                inv=1/(tau*tau)
                hh=np.zeros((self.domains+1,self.domains+1))
                hh[0,0]=gamma*self.n*(1+inv.sum())
                hh[0,1:]=hh[1:,0]=-gamma*self.n*inv
                hh[1:,1:]=np.diag(gamma*self.n*inv+1/vm)
                gg=np.r_[gamma*scores[:,0].sum()-gamma*np.sum(scores[:,1:].sum(0)/tau),
                         gamma*scores[:,1:].sum(0)/tau+means/vm]
                change=-np.linalg.solve(hh,gg)
                candidate=scores.copy();candidate[:,0]+=change[0]
                candidate[:,1:]+=(change[1:]-change[0])/tau
                if np.max(abs(candidate))<8.:
                    scores=candidate;means+=change[1:];modes[:,0]+=change[1:][self.membership]
                composite=scores[:,0,None]+scores[:,1:]*tau
                optimum=np.sum(composite*inv,axis=1)/(1+inv.sum())
                lower=np.maximum(-8.,np.max(composite-8*tau,axis=1))
                upper=np.minimum(8.,np.min(composite+8*tau,axis=1))
                scores[:,0]=np.clip(optimum,lower,upper)
                residual=composite-scores[:,0,None]
                optimum_tau=(gamma*np.sum(residual**2,axis=0))**.25
                lower_tau=np.maximum(np.max(abs(residual),axis=0)/8.,1e-6)
                tau=np.clip(optimum_tau,lower_tau,20.)
                scores[:,1:]=residual/tau
        return dict(scores=scores,tau=tau,means=means,modes=modes),m_success

    def fit(self,seed=0,initial=None,verbose=False):
        started=time.monotonic(); state=self.start(seed) if initial is None else {k:np.array(v,copy=True) for k,v in initial.items()}
        e=self.estep(state); history=[]; converged=False; patience=0;stationary_patience=0;stopping_reason='iteration_cap'
        for iteration in range(self.settings.max_iterations):
            candidate,mok=self.mstep(state,e)
            enew=self.estep(candidate)
            acceleration_accepted=False
            # SQUAREM-style safeguarded fixed-point extrapolation. Both base
            # steps remain transposed EM; candidates must improve the same
            # approximate marginal objective and respect parameter bounds.
            if self.nf>1 and iteration>=4:
                second,mok2=self.mstep(candidate,enew);esecond=self.estep(second)
                def pack(z):return np.r_[z['scores'].ravel(),z['means'],z['tau']]
                x0,x1,x2=pack(state),pack(candidate),pack(second)
                r=x1-x0;v=x2-2*x1+x0
                alpha=-np.clip(np.sqrt(np.dot(r,r)/max(np.dot(v,v),1e-30)),1.,20.)
                proposal=x0-2*alpha*r+alpha*alpha*v
                cut=self.n*self.nf
                accelerated=dict(scores=np.clip(proposal[:cut].reshape(self.n,self.nf),-8.,8.),
                    means=proposal[cut:cut+self.domains],tau=np.clip(proposal[cut+self.domains:],0.,20. if self.family=='testlet' else self.settings.bifactor_mean_upper),modes=second['modes'])
                candidate,enew,mok=second,esecond,mok and mok2
                try:
                    ea=self.estep(accelerated)
                    if np.isfinite(ea['objective']) and ea['objective']>=esecond['objective']:
                        candidate,enew=accelerated,ea;acceleration_accepted=True
                except FloatingPointError:
                    pass
            delta=enew['objective']-e['objective']
            maxchange=float(np.max(np.abs(candidate['scores']-state['scores'])))
            prediction_change=float(np.max(np.abs(enew['prediction']-e['prediction'])))
            score_residual=self.normalized_score(candidate,enew)
            # AGHQ is approximate: record decreases instead of claiming exact EM monotonicity.
            history.append(dict(iteration=iteration+1,objective=enew['objective'],delta=delta,score_change=maxchange,mstep_success=mok,mode_gradient=enew['mode_gradient'],acceleration_accepted=acceleration_accepted,prediction_change=prediction_change,normalized_score_residual=score_residual))
            state,e=candidate,enew
            if verbose and ((iteration+1)%10==0):
                print(self.family,seed,iteration+1,round(e['objective'],5),round(maxchange,6),round(score_residual,7),flush=True)
            numerical_ok=(abs(delta)/self.p<self.settings.tolerance and maxchange<self.settings.parameter_tolerance
                    and prediction_change<self.settings.prediction_tolerance and score_residual<self.settings.score_tolerance)
            if self.nf==1:
                numerical_ok=(abs(delta)<2e-5 and maxchange<2e-4)
            if iteration>=9 and numerical_ok:
                patience+=1
                if patience>=3: converged=True;stopping_reason='converged';break
            else: patience=0
            if (iteration>=19 and abs(delta)<2e-4 and maxchange<2e-4 and prediction_change<2e-4
                    and score_residual>=self.settings.score_tolerance):
                stationary_patience+=1
                if stationary_patience>=5:
                    stopping_reason='stationary_but_score_check_unmet';break
            else:stationary_patience=0
        return dict(family=self.family,seed=int(seed),converged=converged,iterations=len(history),
                    elapsed_seconds=time.monotonic()-started,settings=asdict(self.settings),history=history,
                    engine='arrowhead/Schur-Newton location-profiled generalized EM',stopping_reason=stopping_reason,
                    state=state,**e)

    def normalized_score(self,state,e):
        """Projected, posterior-averaged marginal score per relevant item.

        This audits an AGHQ approximation to the Fisher score identity, not
        the derivative of a moving finite quadrature grid.
        """
        v=np.r_[state['scores'].ravel(),state['tau']] if self.family=='testlet' else state['scores'].ravel()
        _,grad=score_value_grad(v,e['nodes'],e['weights'],self.Y,self.mask,self.membership,state['tau'],
                    self.settings.score_penalty,family=self.family,n=self.n,domains=self.domains)
        grad=np.asarray(grad).copy();cut=self.n*self.nf
        grad[:cut][((v[:cut]<=-8+1e-7)&(grad[:cut]>0))|((v[:cut]>=8-1e-7)&(grad[:cut]<0))]=0
        counts=np.bincount(self.membership,minlength=self.domains)
        divisors=np.ones((self.n,self.nf))*self.p
        if self.nf>1:divisors[:,1:]=counts
        residual=list(abs(grad[:cut])/divisors.ravel())
        postmean=np.einsum('jk,jkd->jd',e['weights'],e['nodes'])
        for d in range(self.domains):
            group=self.membership==d;count=counts[d]
            bgrad=(count*state['means'][d]-postmean[group,0].sum())/self.variance[0]+state['means'][d]/self.settings.mean_intercept_sd**2
            residual.append(abs(bgrad)/count)
            if self.family=='bifactor':
                tg=(count*state['tau'][d]-postmean[group,2].sum())/self.variance[2]+state['tau'][d]
            elif self.family=='testlet':tg=grad[cut+d]
            else:continue
            upper=20. if self.family=='testlet' else self.settings.bifactor_mean_upper
            if (state['tau'][d]<=1e-7 and tg>0) or (state['tau'][d]>=upper-1e-7 and tg<0):tg=0.
            residual.append(abs(tg)/count)
        return float(max(residual))


def higher_order(state):
    tau=state['tau']; lam=1/np.sqrt(1+tau*tau)
    scores=state['scores']; h=(scores[:,0,None]+tau*scores[:,1:])*lam
    return lam,h
