import numpy as np




def pkalman(t, y, f, yerr):
  ''' PKALMAN Power Spectral Density (PSD) estimate via Kalman filtering and smoothing.
 INPUT
   x: the input signal, which is a column vector

 The following optional input arguments can be specified in the form of name/value pairs:
 [default value in brackets]
   Ns: the frequency resolution. If it is in the input argument list, Ns must 
       show before Fs. [Ns = 64]
   Fs: Fs can be either a scalar or a row vector.
      (scalar) the sampling frequency Fs in Hz, or twice of the freqeunce range of a 
       given signal. [Fs=6 (Hz)]
      (vector) the given frequencies bases for spectrum estimation
   t: the sampling time.   [(1:n)/(Fs*2)]
   sigma: the process noise variance    [1e1*eye(2*length(Fs)+1)]
   lam: the observation noise variance   [1e-2]
   m0: the mean of the prior distribution    [ones(2*length(Fs)+1,1)]
   v0: the variance of the prior distribution  [1e2*eye(2*length(Fs)+1,1)]
 

 OUTPUT
 varargout = [amp, phs, vtar]  
   amp: estimated frequency amplitudes from time 1 to n
   phs: estimated frequency phases from time 1 to n
   vtar: the variances from last block's smoothing it's an empty 
        matrix when we only do Kalman filtering

 EXAMPLE 1: Estimate the spectrum of a signal with fast decaying amplitude and 10 percent missing data
   fs = 500
   tt = 0:(1/fs):2
   perc = 0.9
   tr = randperm(length(tt))
   tc = sort(tr(1:floor(length(tt)*perc)))  truncated index of tt
   t = tt(tc)
   w = 50
   aplmd = exp(-t*1)
   x = (sin(2*pi*t*w).*aplmd)'
   m = pkalman(x,'t',t,'Ns',50,'Fs', 200, 'smoothing',1, 'LogScale',0,'disp',1)
 EXAMPLE 2: Estimate the spectrum of a signal consisting of three frequency components which are close to each other
   fs = 50. 
   t = np.arange(0,3, 1./fs)
   y = np.sin(2*np.pi*21*t+ 20.) + 1*np.sin(2*np.pi*20*t + 20)+.5*np.sin(2*np.pi*19*t + 20)
   nslevel = np.sqrt(1e-1) 
   x = y + nslevel*np.random.randn(len(y))
   pkalman(x,'Ns',64,'Fs',fs,'ts',length(x),'axis',...
   [0 25 -55 10],'XTICK',[0 5 10 15 19 20 21 25], 'YTICK',[-40 -20 0],'FontSize',6)

 References
     [1] Bayesian Spectrum Estimation of Unevenly Sampled Nonstationary Data", 
      Yuan Qi, Thomas P. Minka, and Rosalind W. Picard, MIT Media Lab 
      Technical Report Vismod-TR-556, 2002
      http://web.media.mit.edu/~yuanqi/papers/qi-spec-02.pdf
     [2] A shorter version in ICASSP 02, May 2002, Orlando, Florida
      http://web.media.mit.edu/~yuanqi/papers/qi-icassp02.pdf

 Author: Yuan Qi
        MIT Media lab
        Date: April 12, 2003
  ported by: Thuso Simon 2015
  '''
  assert len(t) == len(y), 'length of imput must be the same.'
  # make m0, v0, sigma and lam
  # param error (2*len(f) + 1)
  v0 = np.eye(2 * len(f) + 1) 
  # mean amplitude prior (uniform for now) 
  m0 = np.ones(2 * len(f) + 1)
  # how v0 and m0 change?
  sigma = 1000*np.eye(len(v0)) 
  # estimate the posterior mean by Kalman filtering/smoothing
  m, vt =  kalman_est(t, y, f, sigma, yerr, m0, v0)
  # estimate amplitude
  amp = np.sqrt([np.sum(m[[j,j+len(f)],:]**2) for j in range(1,1+ len(f))])
  # uncertanty
  err = np.sqrt([np.sum(vt[[j,j+len(f)]]) for j in range(1,1+ len(f))])
  # phs: phase estimate
  '''phs = []
    for j = 1:ds
       rowwd = [j j+ds]
      
      phs(j,:) = atan(m(j, :)./m(j+ds,:))
   
    phs = ([zeros(1,n) phs])
    varargout{2} = phs

  if nargout > 2
    varargout{3} = vtar

  if nargout == 0 | disp_flg == 1
    disp_spec(amp, t, Fs, varargin{:})'''
  return amp, err #, phs, vtar

def kalman_est(t, y, Fs, sigma, lam, m0, v0):
  ''' KALMAN_EST is called by PKALMAN 
  # Yuan Qi
  # April 2003'''
  
  twopi = 2 * np.pi
  m = np.zeros((len(m0), len(y)))
  C = np.array([np.concatenate([np.sin(twopi*Fs*t[0]), np.cos(twopi*Fs*t[0]), [1]])])
  # initalise matricis
  v0C = v0.dot(C.T)
  k1 = v0C.dot(np.linalg.inv(C.dot(v0C) + lam[0]))
  vt = v0 - k1.dot(v0C.T)
  m[:, 0] = m0 + k1.dot(y[0] - C.dot(m0))
  # kalman filter
  for j in range(100):
   y_model = [C.dot(m[:,0])]
   for i in range(1, len(y)):
    inttp = t[i] - t[i-1]
    # update the state noise variance
    sigma_t = sigma * (inttp)  
    # if mod(i,150)==0,  vt = v0  end
    pt = vt + sigma_t
    C = np.array([np.concatenate([np.sin(twopi*Fs*t[i]), np.cos(twopi*Fs*t[i]), [1]])])
    ptC = pt.dot(C.T)
    kt = ptC.dot(np.linalg.inv(C.dot(ptC) + lam[i]))
    vtp = pt - kt.dot(C.dot(pt))       
    y_model.append(C.dot(m[:,i-1]))
    m[:,i] = m[:,i-1] + kt.dot(y[i]- C.dot(m[:,i-1]))
    vt = vtp + 0.
   m[:,0] = np.copy(m[:,-1])
  #return m, v and y_model
  return m, vt.diagonal(), np.array(y_model)