from astropy import units as u

def DER_SNR(flux):
   """
   DESCRIPTION This function computes the signal and the noise DER_SNR following the
               definition set forth by the Spectral Container Working Group of ST-ECF,
           MAST and CADC.

               signal = median(flux)
               noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
           snr    = signal / noise
               values with padded zeros are skipped

   USAGE       noise = DER_SNR(flux)
   PARAMETERS  none
   INPUT       flux (the computation is unit independent)
   OUTPUT      the estimated signal-to-noise ratio [dimensionless]
   USES        numpy
   NOTES       The DER_SNR algorithm is an unbiased estimator describing the spectrum
           as a whole as long as
               * the noise is uncorrelated in wavelength bins spaced two pixels apart
               * the noise is Normal distributed
               * for large wavelength regions, the signal over the scale of 5 or
             more pixels can be approximated by a straight line

               For most spectra, these conditions are met.

   REFERENCES  * ST-ECF Newsletter, Issue #42:
               www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
               * Software:
           www.stecf.org/software/ASTROsoft/DER_SNR/
   AUTHOR      Felix Stoehr, ST-ECF
               24.05.2007, fst, initial import
               01.01.2007, fst, added more help text
               28.04.2010, fst, return value is a float now instead of a numpy.float64
               06.12.2018, mro, ported to python 3: variables start and end made explicitely integer
                                (division between integers is a float in py3 and an integer in py2)
                                Also, edited to return both the signal and the noise separately
               17.11.2019, mro, raise error if the input is not monodimensional
   """
   from numpy import array, where, nanmedian, abs, zeros, int

   flux = array(flux)
   if flux.ndim > 1:
      raise ValueError('The input array to der_snr has to be one-dimensional (%iD provided instead)' % (flux.ndim))

   # Values that are exactly zero (padded) are skipped
   flux = array(flux[where(flux != 0.0)])

   # For spectra shorter than this, no value can be returned
   nbins = len(flux)
   nranges = 10

   noise = zeros(nranges)
   for i in range(nranges):
      start = int(i*nbins/nranges)
      end = int((i+1)*nbins/nranges)
      if end > nbins-1:
          end = nbins-1
      flux_range = flux[start:end]

      n = len(flux_range)
      noise[i]  = 0.6052697 * nanmedian(abs(2.0 * flux_range[2:n-2] - flux_range[0:n-4] - flux_range[4:n]))

   return nanmedian(flux_range), nanmedian(noise)


def doppler_shift(wave, flux, radial_velocity):
    assert(radial_velocity.unit == u.kilometer/u.second) # radial velocity must be [km/s]
    doppler_factor = (1 + radial_velocity.value/299792.458)
    new_wave = wave * doppler_factor
    flux_preserved = flux / doppler_factor # increase of bins shouldn't increase flux

    return new_wave, flux_preserved
