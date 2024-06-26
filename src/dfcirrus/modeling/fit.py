import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from astropy import stats 

class Fitter:
    
    """ 
    
    A fitter helper storing fitting inputs and outputs.
    
    """
    
    def __init__(self, x, y,
                 xconstr=None,
                 xmin=None, ymin=None, clip=None, name='1'):
        
        self.name = name
        
        assert len(x) == len(y), print("Length of x do not match y!")
        if xconstr is not None:
            assert len(x) == len(xconstr), print("Length of x do not match xconstr!")
        
        # set domain of x and y
        cond = ~np.isnan(x) & ~np.isnan(y)
        if xmin is not None: cond &= (x>xmin)
        if ymin is not None: cond &= (y>ymin)
        
        if clip is not None:
            x_q01, x_q99 = np.nanquantile(x, [clip, 1-clip])
            cond &= (x>=x_q01) & (x<=x_q99)
        
        self.cond = cond
        self.x, self.y = x, y
        self.xdata, self.ydata = x[cond], y[cond]
        self.xmin, self.ymin = xmin, ymin
    
        if xconstr is not None:
            xconstr = xconstr[cond]
        self.xconstr = xconstr 
        
        self.runned = False
        
        # Initialize labels for populations
        labels = -1 * np.ones_like(x)
        self.labels = labels
        
                
    def setup(self, 
              n_model=1, 
              poly_deg=2,
              include_noise=True,
              piecewise=False,
              penalty=1e-2, 
              sigmoid=True, sigmoid_a=0.2,
              fit_bkg2d=False, 
              verbose=True):
        
        """ Setup fitting """
        
        from .models import (eval_likelihood_poly,
                             eval_likelihood_poly_mix_constr,
                             eval_likelihood_piecewise_lin_mix_constr,
                             eval_likelihood_multi_lin_mix_constr)
        
        xdata, ydata = self.xdata, self.ydata
        
        self.mixture_model = False
        self.piecewise = piecewise
        if piecewise:
            n_model = 2
        
        # printout message
        if verbose: print("Fitting Parameters: ")
        if n_model==1:
            msg = ''.join(["a%d={:.4g}, "%(poly_deg-i) for i in range(poly_deg+1)]) + "eps={:.4g}"
        elif piecewise:
            msg = ''.join(["A1={:.4g}, x0={:.4g}, y0={:.4g}, A2={:.4g}, "]) + "eps={:.4g}"
        else:
            msg = ''.join(["A%d={:.4g}, B%d={:.4g}, frac%d={:.4g}, "%tuple([i+1]*3) for i in range(n_model)]) + "eps={:.4g}"
            self.mixture_model = True
        
        if sigmoid:
            self.sigmoid_a = sigmoid_a
            msg += ", x_min={:.4g}"
        self.sigmoid = sigmoid
            
        if include_noise:
            msg += ", mu={:.4g}, sigma={:.4g}"
            if not self.mixture_model:
                msg += ", frac={:.4g}"
        self.include_noise = include_noise
                
        xconstr = self.xconstr
        if xconstr is not None:
            msg += ", b1={:.4g}, b0={:.4g}"
        if fit_bkg2d:
            msg += ", P10={:.4g}, P01={:.4g}, P11={:.4g}, P20={:.4g}, P02={:.4g}"
        
        self.msg = msg
        if verbose: print(f"  [{msg.replace('={:.4g}', '')}]")
        
        if n_model==1:
            n_param = poly_deg + 2
            if verbose: print("    - polynomial degree = {}".format(poly_deg))
        elif piecewise:
            n_param = 6
            poly_deg = 1
        else:
            n_param = 3 * n_model + 1
            if verbose: 
                print("    - # of models : {}".format(n_model))
                print("    - Only poly_deg = 1 available for mixture models.")
            poly_deg = 1
         
        self.n_model = n_model
        self.poly_deg = poly_deg
    
        if fit_bkg2d:
            from .models import make_legendre2d_grids
            # make legendre grids
            self.legendre_grids = make_legendre2d_grids(self.image_shape)
            legendre_terms = [grid[self.fit_range].ravel()[cond] for grid in self.legendre_grids]
            n_param += 5  # add A10, A01, A11, A20, A02
            if verbose:  print("    - Fit Legendre 2D background.")
        else:
            legendre_terms = None
            
        self.legendre_terms = legendre_terms
        
        fit_kwargs = dict(xconstr=xconstr, 
                          sigmoid=sigmoid, 
                          a=sigmoid_a,
                          penalty=penalty, 
                          legendre_terms=legendre_terms,
                          give_prob=False)
        
        self.penalty = penalty
        self.fit_kwargs = fit_kwargs
        
        if sigmoid: n_param += 1  # add x_min
        
        if include_noise:
            # mixtue model including outliers
            n_param +=2   # add mu, sigma
            
            if xconstr is not None:
                if verbose: print("    - Penelized fitting with linear constraint.")
                n_param += 2 # add c, d
            
            if n_model==1:
                n_param +=1   # add frac
                
                p_eval_lld = partial(eval_likelihood_poly_mix_constr, 
                                     xdata=xdata, ydata=ydata, 
                                     poly_deg=poly_deg, 
                                     **fit_kwargs)
                
            elif piecewise:
                p_eval_lld = partial(eval_likelihood_piecewise_lin_mix_constr, 
                                     xdata=xdata, ydata=ydata, 
                                     **fit_kwargs)
            else:
                p_eval_lld = partial(eval_likelihood_multi_lin_mix_constr, 
                                     xdata=xdata, ydata=ydata, 
                                     n_model=n_model, 
                                     **fit_kwargs)
            
        else:
            if n_model==1:
                p_eval_lld = partial(eval_likelihood_poly, 
                                     xdata=xdata, ydata=ydata,
                                     poly_deg=poly_deg, 
                                     sigmoid=sigmoid, a=a, 
                                     legendre_terms=legendre_terms)
            else:
                p_eval_lld = partial(eval_likelihood_multi_lin_mix_constr, 
                                     xdata=xdata, ydata=ydata, 
                                     n_model=n_model, 
                                     include_noise=False,
                                     **fit_kwargs)
                
        self.n_param = n_param  # number of parameters
        self.lld = p_eval_lld
        
        
    def run(self, p0, method='Nelder-Mead', verbose=True):
        """ Run fitting """
        from scipy.optimize import minimize
        
        # Check if # of parameters match
        n_param = self.n_param
        assert n_param == len(p0), print("# of p0 does no match!")
        
        
        self.p0 = p0
        self.method = method
        
        lld = self.lld
        
        # run negative likelihood minimization
        res = minimize(lld, p0, method=method)
        params_fit = res.x
        
        self.res = res
        self.params_fit = params_fit
        self.runned = True
        
        # evaluate population probabilities
        if not self.mixture_model:
            if self.include_noise:
                prob = lld(params_fit, give_prob=True)
                self.prob = prob
                self.labels[self.cond] = (prob>0.5).astype(int)
        else:
            probs = lld(params_fit, give_prob=True)
            self.probs = probs
            
            prob = np.sum(probs, axis=0)
            self.labels[self.cond] = np.argmax(probs / prob, axis=0)
        
        # print out
        if verbose:
            print(self.msg.format(*self.params_fit))
    
    @property
    def BIC(self):
        lld = - self.lld(self.params_fit)
        n_params = len(self.params_fit)
        n_samples = len(self.ydata)
        
        bic = stats.bayesian_info_criterion(lld, n_params, n_samples)
        return bic
    
    def plot(self, 
             xlabel='', ylabel='',
             xrange=(-5,10), yrange=(-5,10),
             nbins=None, color='k', 
             return_fig=False):
        
        """ Plot fitting result. """
        
        from numpy.polynomial import Polynomial
        from scipy.special import expit
        
        x, y = self.x, self.y
        cond = self.cond
        
        n_model = self.n_model
        poly_deg = self.poly_deg
        params_fit = self.params_fit
        piecewise = self.piecewise
        mixture_model = self.mixture_model
        
        if self.normed:
            xx = np.linspace(xrange[0],xrange[1])
        else:
            # xx = np.linspace(np.quantile(x,0.00001), np.quantile(x,0.99999))
            xx = np.linspace(xrange[0],xrange[1])

        if n_model==1:
            # 1d polynomial
            coefs = params_fit[:poly_deg+1]
            poly = Polynomial(coefs[::-1])
            yfunc = lambda x: poly(x)
        elif piecewise:
            A1 = params_fit[0]
            x0, y0 = params_fit[1:3]
            A2 = params_fit[3]
            
            B1 = y0 - A1 * x0 
            B2 = y0 - A2 * x0
            
            f1 = lambda x: A1 * x + B1
            f2 = lambda x: A2 * x + B2

            yfunc = lambda x: np.piecewise(x, [x<x0, x>=x0], [f1, f2])
            
        else:
            As = params_fit[:3*n_model:3]
            Bs = params_fit[1:3*n_model:3]
            fracs = params_fit[2:3*n_model:3]
            yfunc = lambda x: np.sum(np.r_['0,2,1', [A * x * frac + B * frac for (A, B, frac) in zip(As, Bs, fracs)]], axis=0)
            
        self.yfunc = yfunc
        
        # smoothing with sigmoid function
        if self.sigmoid:
            a = self.sigmoid_a
            if n_model == 1:
                x_min = params_fit[poly_deg+2]
                y_min = poly(x_min-a)
            elif piecewise:
                x_min = params_fit[5]
                y_min = A1 * (x_min-a) + B1
            else:
                x_min = params_fit[3*n_model+1]
                y_min = yfunc(xx)[np.argmin(abs(xx-(x_min-a)))]

            h = lambda x: expit((x-x_min)/a)
            yy = yfunc(xx) * h(xx) + y_min * (1-h(xx))
        else:
            yy = yfunc(xx)

        # remove 2D background
        if self.legendre_terms is not None:
            coefs_bkg = params_fit[-5:]  # last five parameters
            bkg = np.sum([A*P for (A, P) in zip(coefs_bkg, self.legendre_terms)], axis=0)
            y -= bkg

        labels = self.labels

        # set canvas
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12, 5))

        # Hess plot
        xyrange = [[xrange[0],xrange[1]],[yrange[0],yrange[1]]]
        if nbins is None:
            nbins = int(100*self.rescale_factor)
        H,xbins,ybins,image = ax1.hist2d(x, y, bins=nbins, range=xyrange)
        levels = np.linspace(0, np.log10(0.8*H.max()), 7)[1:]
        ax1.contour(np.log10(H.T+1e-2), levels, extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()], linewidths=1, cmap='gray')

        # Draw points outside contours
        indices = np.column_stack((np.digitize(x, xbins[1:-1]), np.digitize(y, ybins[1:-1])))
        inside = np.array([np.log10(H[ind[0],ind[1]]) for ind in indices]) >= levels[0]
        ax1.scatter(x[(~inside)&cond], y[(~inside)&cond], s=3, alpha=0.2, color='w')

        # scatter plot
        if mixture_model:
            # plot labeled data
            for k in range(n_model):
                if fracs[k] > 1e-4:
                    ax2.scatter(x[labels==k+1], y[labels==k+1], s=0.1, alpha=0.02)
        else:
            ax2.scatter(x[cond], y[cond], s=1, alpha=0.05, color=color)
                    
        # plot data not in use
        ax2.scatter(x[~cond], y[~cond], s=0.1, alpha=0.02, color='gray')
        if self.include_noise:
            # plot outliers
            ax2.scatter(x[labels==0], y[labels==0], s=0.1, alpha=0.05, color='gray')
        if self.normed:
            xlim, ylim = xrange, yrange
        else:
            # xlim, ylim = np.nanquantile(x,[0.0005, 0.999995]), np.nanquantile(y,[0.0005, 0.999995])
            xlim, ylim = xrange, yrange

        ax2.set_xlim(xlim[0],xlim[1])
        ax2.set_ylim(ylim[0],ylim[1])

        for ax in [ax1,ax2]:
            if mixture_model:
                for (A, B, frac) in zip(As, Bs, fracs):
                    if frac > 1e-4:
                        ax.plot(xx, A * xx + B, ls='--')
            ax.plot(xx, yy, color='lime')
            if self.xmin is not None:
                ax.axvline(self.xmin, color='gold', ls='--')
            if self.ymin is not None:
                ax.axhline(self.ymin, color='gold', ls='--')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        
        plt.tight_layout()
        
        if return_fig:
            return fig, (ax1,ax2)
        else:
            plt.show()
            return None
        
