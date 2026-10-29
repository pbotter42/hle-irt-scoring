"""CFA/IRT-style explanatory figures. All quantities use the stated reference."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

BG, INK, MUTED = '#fbfaf7', '#24343b', '#667880'
GENERAL, SPECIFIC = '#267e88', '#a85d36'
NAMES=['Biology / medicine','Math','Humanities /\nsocial science','Computer science / AI',
       'Other','Physics','Chemistry','Engineering']

def latent(ax,x,y,label,color,r=.40):
    p=Circle((x,y),r,facecolor=BG,edgecolor=color,lw=1.8,zorder=3)
    ax.add_patch(p);ax.text(x,y,label,ha='center',va='center',fontsize=18,color=color,zorder=4)
    return p

def observed(ax,x,y,label,w=.9,h=.60):
    p=Rectangle((x-w/2,y-h/2),w,h,facecolor='white',edgecolor=MUTED,lw=1.25,zorder=3)
    ax.add_patch(p);ax.text(x,y,label,ha='center',va='center',fontsize=13,color=INK,zorder=4)
    return p

def center(p):
    return p.center if isinstance(p,Circle) else (p.get_x()+p.get_width()/2,p.get_y()+p.get_height()/2)

def arrow(ax,a,b,color,style='-|>',connectionstyle='arc3'):
    p=FancyArrowPatch(center(a),center(b),patchA=a,patchB=b,shrinkA=2,shrinkB=2,
                     arrowstyle=style,mutation_scale=12,lw=1.4,color=color,zorder=2,
                     connectionstyle=connectionstyle)
    ax.add_patch(p)

def bifactor_paths(counts):
    counts=np.asarray(counts,int)
    if counts.shape!=(8,):raise ValueError('Show all eight subject groups.')
    fig=plt.figure(figsize=(16.5,6.9),facecolor=BG)
    ax=fig.add_axes([.025,.16,.95,.69])
    ax.set(xlim=(0,24),ylim=(0,6.65),aspect='equal');ax.axis('off')
    general=latent(ax,12,5.70,r'$g$',GENERAL,r=.45)
    ax.text(12,6.4,'General factor',ha='center',fontsize=13,color=GENERAL)
    for d,(x,n,name) in enumerate(zip(np.arange(8)*3+1.5,counts,NAMES),1):
        ax.add_patch(Rectangle((x-1.40,.10),2.80,3.80,facecolor='none',edgecolor='#aebfc2',
                               lw=1,linestyle=(0,(4,4))))
        left=observed(ax,x-.80,3.30,r'$Y_1$')
        right=observed(ax,x+.80,3.30,rf'$Y_{{{n}}}$')
        ax.text(x,3.30,r'$\cdots$',ha='center',va='center',fontsize=20,color=MUTED)
        specific=latent(ax,x,1.72,rf'$s_{d}$',SPECIFIC)
        for item in [left,right]:
            arrow(ax,general,item,GENERAL)
            arrow(ax,specific,item,SPECIFIC)
        ax.text(x,.81,name,ha='center',va='center',fontsize=10,color=INK,linespacing=1.15)
        ax.text(x,.28,f'{n} questions',ha='center',fontsize=10,color=MUTED)
    ax.text(.15,4.43,r'General paths: $\lambda_{gj}$',fontsize=12,color=GENERAL,
            bbox=dict(facecolor=BG,edgecolor='none',pad=2))
    ax.text(.15,2.35,r'$\lambda_{sj}$',fontsize=13,color=SPECIFIC,
            bbox=dict(facecolor=BG,edgecolor='none',pad=1))
    fig.suptitle('One general factor. Eight residual factors. Every question has two paths.',
                 fontsize=20,fontweight='bold',x=.04,ha='left',y=.98,color=INK)
    fig.text(.04,.885,'Circles represent factors; rectangles represent item responses. Colors distinguish general and specific paths.',
             fontsize=12,color=MUTED)
    fig.text(.04,.115,f'Every subject is explicit. Each first item, ellipsis, and last item together represent all {counts.sum()} questions.',
             fontsize=11,color=INK)
    fig.text(.04,.073,'Conventional bifactor: all factors are mutually orthogonal. Intercepts and item errors are omitted here.',
             fontsize=11,color=MUTED)
    fig.text(.04,.031,'This fit uses the bifactor loading pattern; it does not force the estimated fixed score columns to be empirically orthogonal.',
             fontsize=11,color=MUTED)
    return fig

def ecv_path(a,c,domain,item_id):
    scale=np.sqrt(a*a+c*c+np.pi**2/3)
    lg,ls=a/scale,c/scale
    iecv=lg*lg/(lg*lg+ls*ls)
    fig=plt.figure(figsize=(13,6.8),facecolor=BG)
    ax=fig.add_axes([.035,.25,.46,.58]);ax.set(xlim=(0,7),ylim=(0,5.4),aspect='equal');ax.axis('off')
    g=latent(ax,1.0,4.0,r'$g$',GENERAL,r=.45)
    s=latent(ax,1.0,1.3,r'$s_{\rm math}$',SPECIFIC,r=.51)
    y=observed(ax,4.7,2.65,r'$Y_j$',w=1.3,h=.8)
    arrow(ax,g,y,GENERAL);arrow(ax,s,y,SPECIFIC)
    ax.text(2.0,3.70,rf'$\lambda_{{gj}} = {lg:.3f}$',color=GENERAL,fontsize=14)
    ax.text(2.0,1.35,rf'$\lambda_{{sj}} = {ls:.3f}$',color=SPECIFIC,fontsize=14)
    error=latent(ax,6.2,2.65,r'$\epsilon_j$',MUTED,r=.32)
    arrow(ax,error,y,MUTED)
    ax.text(.3,.28,r'$g\perp s$; both factor variances set to 1',fontsize=12,color=MUTED)
    fig.suptitle('ECV compares the two common paths, leaving item error outside the ratio',
                 fontsize=19,fontweight='bold',x=.04,ha='left',y=.98,color=INK)
    fig.text(.04,.895,f'An actual {domain} item: {item_id}',fontsize=12,color=MUTED)
    fig.text(.55,.76,'Square the general and specific loadings:',fontsize=13,color=INK)
    fig.text(.55,.67,rf'General: $\lambda_{{gj}}^2 = {lg*lg:.3f}$',fontsize=16,color=GENERAL)
    fig.text(.55,.58,rf'Specific: $\lambda_{{sj}}^2 = {ls*ls:.3f}$',fontsize=16,color=SPECIFIC)
    fig.text(.55,.44,rf'$\mathrm{{IECV}}_j = \frac{{{lg*lg:.3f}}}{{{lg*lg:.3f}+{ls*ls:.3f}}} \approx {iecv*100:.2f}\%$',
             fontsize=22,color=INK)
    fig.text(.55,.31,'Subject / whole-test ECV: sum squared loadings\nacross the relevant items before taking the ratio.',
             fontsize=12,color=INK,linespacing=1.55)
    fig.text(.04,.17,r'Conversion from IRT: $\lambda_{gj}=a_j/\sqrt{a_j^2+c_j^2+\pi^2/3}$, '
             r'$\lambda_{sj}=c_j/\sqrt{a_j^2+c_j^2+\pi^2/3}$.',fontsize=13,color=INK)
    fig.text(.04,.103,'The logistic residual variance is included when standardizing loadings, but not in the common-variance ratio.',
             fontsize=11,color=MUTED)
    fig.text(.04,.044,'Orthogonal, unit-variance reference only. The fixed-score fit does not impose these empirical score constraints.',
             fontsize=11,color=MUTED)
    return fig
