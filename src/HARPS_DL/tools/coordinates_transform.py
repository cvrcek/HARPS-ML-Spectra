def topo_to_bary(lambda_topo,berv):
    lambda_bary = lambda_topo * 1/(1-(berv/299792.458))
    return lambda_bary

#----------------------------------------------------------------------
def bary_to_topo(lambda_bary,berv):
    lambda_topo = lambda_bary * (1-(berv/299792.458))
    return lambda_topo

def get_berv(ID):
    return bervs[bervs['dp_id']==ID]['kw_value'].values[0]

def baryname2topo(dp_id, bervs):
        # constants

        # load data
        hdu = fits.open(dp_id)
        WAVE = hdu[1].data.field('WAVE').astype(np.float32).T # barycentric
        spectrum = hdu[1].data.field('FLUX').astype(np.float32).T
        berv = bervs[bervs['dp_id']==dp_id]['kw_value'].values[0]

        # - topocentric transform
        WAVE_airtopo = topo_to_bary(WAVE,berv)
        WAVE_airtopo_aux = np.insert(WAVE_airtopo,0,WAVE[0])
        WAVE_airtopo_aux = np.append(WAVE_airtopo_aux,WAVE[-1])
        flux_aux = data.T
        flux_aux = np.insert(flux_aux,0,0)
        flux_aux = np.append(flux_aux,0)
        func = interp1d(WAVE_airtopo_aux,flux_aux)
        flux_new = func(WAVE)
        flux_new = flux_new.astype(np.float32)


        wave = WAVE_airtopo

        #- Trim
        wave = wave[ np.logical_and(wave>=(Spectra_dataset.desiredMinW-Spectra_dataset.eps),
                                    wave<=(Spectra_dataset.desiredMaxW+Spectra_dataset.eps)) ]
        spectrum  = spectrum[np.logical_and(wave>=(Harps_spec_info.desiredMinW-Harps_spec_info.eps),
                                    wave<=(Harps_spec_info.desiredMaxW+Harps_spec_info.eps)) ]

        if(self.fix_for_artifacts):
            spectrum = self.fix_mid(spectrum)
            spectrum = self.fix_paddings(spectrum)

        if(self.median_norm):
            spectrum = self.norm_by_median(spectrum)

        spectrum = np.expand_dims(spectrum,axis=0)
        return torch.Tensor(spectrum)
