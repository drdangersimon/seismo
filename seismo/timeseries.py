'''

This contains some useful routines I need for finding and analysing
frequencies in pulsating star lightcurves

'''
import multiprocessing

import numpy as np
import f90periodogram
from scipy.interpolate import interpolate

try:
    import pyopencl as cl
    OPENCL = True
except ImportError:
    print("opencl not available")
    OPENCL = False


def find_nan(array):
    " strips NaN from array and return stripped array"

    # strip nan
    valid = np.logical_not(np.isnan(array))
    return valid


def fast_deeming(times, values, pad_n=None):
    ''' Interpolate time values to an even grid then run an FFT

    returns (frequencies, amplitudes)

    Input
    -----
    times : numpy array containing time values
    values: numpy array containing measurements
    pad_n : (optional) Calculate fft of this size. If this is larger than the
    input data, it will be zero padded. See numpy.fft.fft's help for details.


    Output
    ------
    frequencies: numpy array containing frequencies
    amplitudes : numpy array containing amplitudes
    even_times : numpy array containing interpolated times
    even_values: numpy array containing interpolated values

    Details
    -------
    Time values are interpolated to an even grid from min(times) to max(times)
    containing times.size values. Interpolation is done using linear spline
    method.

    NOTE: This may not give you results as precise as deeming(), the
    interpolation may cause spurious effects in the fourier spectrum. This
    method is however, very fast for large N, compared to deeming()

    NOTE: This method strips nan from arrays first.
    '''
    valid = find_nan(values)
    values = values[valid]
    times = times[valid]

    interpolator = interpolate.interp1d(times, values)

    even_times = np.linspace(times.min(), times.max(), times.size)
    even_vals = interpolator(even_times)
    if pad_n:
        amplitudes = np.abs(np.fft.fft(even_vals, pad_n))
    else:
        amplitudes = np.abs(np.fft.fft(even_vals, 2*even_vals.size))

    amplitudes *= 2.0 / times.size
    frequencies = np.fft.fftfreq(amplitudes.size,
                                 d=even_times[1]-even_times[0])
    pos = frequencies >= 0

    return frequencies[pos], amplitudes[pos], even_times, even_vals


def periodogram_opencl(t, m, f):
    ''' Calculate the Deeming periodogram using numpy using a parallel O(N*N)
    algorithm. Parallelisation is obtained via opencl and could be run on a
    GPU.

    Inputs:
        t: numpy array containing timestamps
        m: numpy array containing measurements
        f: numpy array containing frequencies at which DFT must be
        calculated

    Returns:
        amplitudes: numpy array of len(freqs) containing amplitudes of
        periodogram

    Note: This routine strips datapoints if it is nan
    '''
    valid = find_nan(m)
    t = t[valid]
    m = m[valid]


    # create a context and a job queue
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # create buffers to send to device
    mf = cl.mem_flags
    # input buffers
    times_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=t)
    mags_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=m)
    freqs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=f)

    # output buffers
    amps_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, f.nbytes)
    amps_g = np.empty_like(f)


    kernel = '''
    // Kernel to compute the deeming periodogram for a given frequency over a
    // set of data

    #define PYOPENCL_DEFINE_CDOUBLE
    #pragma OPENCL EXTENSION cl_khr_fp64: enable

    __kernel void periodogram(
            __global const double *times_g,
            __global const double *mags_g,
            __global const double *freqs_g,
            __global double *amps_g,
            const int datalength) {

        int gid = get_global_id(0);
        double realpart = 0.0;
        double imagpart = 0.0;
        double pi = 3.141592653589793;
        double twopif = freqs_g[gid]*2.0*pi;

        for (int i=0; i < datalength; i++){
            realpart = realpart + mags_g[i]*cos(twopif*times_g[i]);
            imagpart = imagpart + mags_g[i]*sin(twopif*times_g[i]);
        }
        amps_g[gid] = 2.0*sqrt(pow(realpart, 2) + pow(imagpart, 2))/datalength;
    }
    '''

    # read and compile the opencl kernel
    prg = cl.Program(ctx, kernel)
    try:
        prg.build()
    except:
        print("Error:")
        print(prg.get_build_info(ctx.devices,
                                 cl.program_build_info.LOG))
        raise

    # call the function and copy the values from the buffer to a numpy array
    prg.periodogram(queue, amps_g.shape, None,
                    times_g,
                    mags_g,
                    freqs_g,
                    amps_buffer,
                    np.int32(t.size))
    cl.enqueue_copy(queue, amps_g, amps_buffer)

    return amps_g


def periodogram_parallel(t, m, f, threads=None):
    ''' Calculate the Deeming periodogram using Fortran with OpenMP
    '''
    if not threads:
        threads = 4

    # strip nan
    valid = find_nan(m)
    t = t[valid]
    m = m[valid]

    ampsf90omp_2 = f90periodogram.periodogram2(t, m, f, t.size, f.size,
                                               threads)

    return ampsf90omp_2


def periodogram_numpy(t, m, freqs):
    ''' Calculate the Deeming periodogram using numpy using a serial O(N*N)
    algorithm.

    Inputs:
        t: numpy array containing timestamps
        m: numpy array containing measurements
        freqs: numpy array containing frequencies at which DFT must be
        calculated

    Returns:
        amplitudes: numpy array of len(freqs) containing amplitudes of
        periodogram

    Note: This routine strips datapoints if it is nan
    '''

    # strip nan
    valid = find_nan(m)
    t = t[valid]
    m = m[valid]

    # calculate the dft
    amps = np.zeros(freqs.size, dtype='float')
    twopit = 2.0*np.pi*t
    for i, f in enumerate(freqs):
        twopift = f*twopit
        real = (m*np.cos(twopift)).sum()
        imag = (m*np.sin(twopift)).sum()
        amps[i] = real**2 + imag**2

    amps = 2.0*np.sqrt(amps)/t.size

    return amps


def deeming(times, values, frequencies=None, method='opencl',
            opencl_max_chunk=10000):
    ''' Calculate the Deeming periodogram of values at times.
    Inputs:
        times: numpy array containing time_stamps
        values: numpy array containing the measured values at times.
        frequencies: optional numpy array at which the periodogram will be
        calculated. If not given, (times.size) frequencies between 0 and
        the nyquist frequency will be used.
        method: One of 'opencl', 'openmp', 'numpy'.
            'opencl' requires `pyopencl` to be present as well as a working
            opencl driver. This method runs in parallel on the opencl device
            and is potentially the fastest of the 3. This option is default.

            'openmp' runs in parallel via openmp in fortran code. This can only
            run on your CPU. It defaults to the number of cores in your machine.

            'numpy' uses a serial implementation that only requires numpy to be
            installed. This one is probably the slowest of the 3 options for
            larger input data sizes
        opencl_max_chunk: defaults to 10000. If you get "Cl out of resources"
        error, make this smaller

    Returns (frequency, amplitude) arrays.
    '''

    if frequencies is None:
        # frequencies array not given. Create one

        # find the smallest differnce between two successive timestamps and use
        # that for the nyquist calculation
        t = np.arange(times.size-1)
        smallest = np.min(times[t+1] - times[t])
        nyquist = 0.5 / smallest

        frequencies = np.linspace(0, nyquist, times.size)

    if method == 'opencl':
        if OPENCL:
            # split the calculation by frequency in chunks at most
            # 10000 (for now)
            chunks = (frequencies.size / opencl_max_chunk) + 1
            f_split = np.array_split(frequencies, chunks)
            amps_split = []
            for f in f_split:
                amps = periodogram_opencl(times, values, f)
                amps_split.append(amps)

            amps = np.concatenate(amps_split)



        else:
            print("WARNING! pyopencl not found. Falling back to openmp version")
            cores = multiprocessing.cpu_count()
            amps = periodogram_parallel(times, values, frequencies, cores)
    elif method == 'openmp':
        cores = multiprocessing.cpu_count()
        amps = periodogram_parallel(times, values, frequencies, cores)
    elif method == 'numpy':
        amps = periodogram_numpy(times, values, frequencies)
    else:
        raise ValueError("{} is not a valid method!".format(method))

    return frequencies, amps


def find_peak(frequencies, amplitudes, fmin=None, fmax=None):
    ''' Return the return (freq, amp) where amp is maximum'''
    if fmin is None:
        fmin = frequencies.min()

    if fmax is None:
        fmax = frequencies.max()

    _f = np.logical_and(frequencies < fmax, frequencies > fmin)

    ampmax = np.where(amplitudes[_f] == amplitudes[_f].max())

    return float(frequencies[_f][ampmax]), float(amplitudes[_f][ampmax])
