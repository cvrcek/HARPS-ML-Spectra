### Support functions for ETC processing
## imports
import json
from pathlib import Path
import sys
import os
import pickle
import tempfile

import numpy as np

from matplotlib import pyplot as plt

from astropy.io import fits
from astropy import units as u

from scipy.interpolate import interp1d


from pdb import set_trace

from HARPS_DL.datasets.Spectra_info import Harps_spec_info
from HARPS_DL.datasets.Dataset_sim_memmap import Harps_sim_memmap_layer



## Support functions for ETC data manipulation
# functions to combine orders
def split_orders(N, target_fun, wave_fun):
    # given N orders, where no more than two subsequent orders overlap:
    # split target_fun into two arrays (odd/even order), where all orders are concatenated
    st = Harps_spec_info.wave_resolution
    emul_wave = np.arange(Harps_spec_info.desiredMinW-Harps_spec_info.padL*st,Harps_spec_info.desiredMaxW+.001+Harps_spec_info.padR*st,step=st)
    emul_wave *= u.AA
    emul_wave = emul_wave.to(u.m).value

    st *= u.AA
    st = st.to(u.m).value

    even_orders = np.full(emul_wave.shape, np.nan)
    odd_orders = np.full(emul_wave.shape, np.nan)

    for i in range(N):
        # target region
        ind_wave = np.logical_and(emul_wave > (np.min(wave_fun(i)) - st),
                                  emul_wave < (np.max(wave_fun(i)) + st))
        sub_wave = emul_wave[ind_wave]
        interp_fun = interp1d(x=wave_fun(i), y=target_fun(i), bounds_error=False)
        if i % 2 == 0:
            even_orders[ind_wave] = interp_fun(sub_wave)
        else:
            odd_orders[ind_wave] = interp_fun(sub_wave)
    if 0:
        plt.figure()
        idxs = slice(224000,234000)
        wave = emul_wave[idxs]
        wave = wave * u.m
        wave = wave.to(u.AA).value
        plt.plot(wave, even_orders[idxs])
        plt.plot(wave, odd_orders[idxs], linestyle='dotted')
    return even_orders, odd_orders, emul_wave

def combine_even_odd(S_even, S_odd, comb_fun=np.mean):
    # given two partially overlapping spectra:
    # S_even = [a_1, a_2, a_3, ...]
    # S_odd = [b_1, b_2, b_3, ...]
    # they are combined into:
    # S_c = [c_1, c_2, c_3, ...]
    # where c_i = a_i if b_i == nan and a_i != nan
    #       c_i = b_i if a_i == nan and b_i != nan
    #       c_i = comb_fun(a_i, b_i) if a_i != nan and b_i != nan
    #       c_i = nan if a_i == nan and b_i == nan

    # combine even and odd flux
    combined_orders = np.copy(S_even)
    ind_both = np.logical_and(~np.isnan(S_even), ~np.isnan(S_odd))
    combined_orders[ind_both] = comb_fun(np.vstack((S_even[ind_both],S_odd[ind_both])), axis=0)
    ind_odd_only = np.logical_and(np.isnan(S_even), ~np.isnan(S_odd))
    combined_orders[ind_odd_only] = S_odd[ind_odd_only]

    err = np.nanmean(np.abs(S_even[ind_both] - S_odd[ind_both]))

    return combined_orders, err

def combine_orders(N, S_fun, wave_fun, comb_fun=np.mean):
    S_even, S_odd, wave = split_orders(N, S_fun, wave_fun)
    return combine_even_odd(S_even, S_odd, comb_fun=comb_fun)[0], wave

def spread_instrument_term(N, n, wave_fun):
    # n is an array with single value for each order
    # this function will 1) split even/odd orders
    #                    2) fill every wavelength of order i with n(i)
    st = Harps_spec_info.wave_resolution
    emul_wave = np.arange(Harps_spec_info.desiredMinW-Harps_spec_info.padL*st,Harps_spec_info.desiredMaxW+.001+Harps_spec_info.padR*st,step=st)
    emul_wave *= u.AA
    emul_wave = emul_wave.to(u.m).value

    st *= u.AA
    st = st.to(u.m).value

    even_orders = np.full(emul_wave.shape, np.nan)
    odd_orders = np.full(emul_wave.shape, np.nan)

    for i in range(N):
        # target region
        ind_wave = np.logical_and(emul_wave > (np.min(wave_fun(i)) - st),
                                  emul_wave < (np.max(wave_fun(i)) + st))
        sub_wave = emul_wave[ind_wave]
        if i % 2 == 0:
            even_orders[ind_wave] = n(i)
        else:
            odd_orders[ind_wave] = n(i)

    return even_orders, odd_orders

## Fits processing + Doppler shift
def spectrum2flam(spectrum, wave):
    c_speed = 2.99792458 * 1e18 # speed of light as Angstrom per second
    return np.array(spectrum)*c_speed/np.power(wave, 2)*4*np.pi

def create_shifted_fits(pickle_name, rv=0, flux_units='flam'):
    st = Harps_spec_info.wave_resolution
    emul_wave = np.arange(Harps_spec_info.desiredMinW-Harps_spec_info.padL*st,Harps_spec_info.desiredMaxW+.001+Harps_spec_info.padR*st,step=st)
    tmp_fits_file =  tempfile.NamedTemporaryFile(suffix='.fits', delete=False)

    with pickle_name.open(mode='rb') as f:
        data = pickle.load(f)
    shifted_wave, shifted_spectrum = \
        Harps_sim_memmap_layer.doppler_shift(np.array(data['wave']), np.array(data['spectrum']), rv)

    spectrum = np.interp(emul_wave, shifted_wave, shifted_spectrum)

    if flux_units == 'flam':
        spectrum = spectrum2flam(spectrum, emul_wave)
    elif flux_units == 'fnu':
        pass
    else:
        raise('Unknown flux units.')

    wave = emul_wave * u.AA
    c1 = fits.Column(name='WAVE', array=wave.to(u.m).value, format='D')
    c2 = fits.Column(name='FLUX', array=spectrum, format='D')
    t = fits.BinTableHDU.from_columns([c1, c2])
    t.writeto(tmp_fits_file, overwrite=True)
    tmp_fits_file.close()

    return emul_wave, spectrum, tmp_fits_file, data['parameters']

def dopp_shift(wave, signal, rv):
    shifted_wave, shifted_signal = \
        Harps_sim_memmap_layer.doppler_shift(wave, signal, rv)
    return np.interp(wave, shifted_wave, shifted_signal)


## Main computation routines
def precompute(pickle_name, etc_settings, etc_form_file, rv=0, flux_units='flam', debug=False):
    _, _, fits_file, autokur_params = create_shifted_fits(pickle_name=pickle_name, rv=0, flux_units=flux_units)

    with etc_form_file.open('r') as f:
        etc_json = json.load(f)
    etc_json['target']['sed']['redshift']['redshift']= rv/299792.458
    etc_json['sky']['airmass']= etc_settings['airmass']
    etc_json['sky']['pwv']= etc_settings['pwv']
    etc_json['target']['brightness']['params']['mag'] = etc_settings['mag']
    etc_json['timesnr']['DET1.WIN1.UIT1']= etc_settings['Texp']

    json_data = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    json_data.close()

    json_in = tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w')

    json.dump(etc_json, json_in, indent=4)
    json_in.close()

    cmd = f'./etc_cli.py harps {json_in.name} -u {fits_file.name} -s https://etc.eso.org -o {json_data.name}'
    os.system(cmd)

    with Path(json_data.name).open( 'r') as f:
        data_dic = json.load(f)

    if 0:
        os.unlink(json_data.name)
        os.unlink(json_in.name)
        os.unlink(fits_file.name)
    else:
        os.remove(json_data.name)
        os.remove(json_in.name)
        os.remove(fits_file.name)

    output = parse_etc_output(data_dic, debug=debug)
    output['autokur_params'] = autokur_params
    output['etc_settings'] = etc_settings
    return output

def precompute_marcs(etc_settings, etc_form_file, rv=0, debug=False):
    with etc_form_file.open('r') as f:
        etc_json = json.load(f)
    etc_json['target']['sed']['redshift']['redshift']= rv/299792.458
    etc_json['sky']['airmass']= etc_settings['airmass']
    etc_json['sky']['pwv']= etc_settings['pwv']
    etc_json['target']['brightness']['params']['mag'] = etc_settings['mag']
    etc_json['timesnr']['DET1.WIN1.UIT1']= etc_settings['Texp']
    etc_json['target']['sed']['spectrum']['params']['spectype']= etc_settings['marcs']

    json_data = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    json_data.close()

    json_in = tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w')

    json.dump(etc_json, json_in, indent=4)
    json_in.close()

    cmd = f'./etc_cli.py harps {json_in.name} -s https://etctest2.hq.eso.org -o {json_data.name}'
    os.system(cmd)

    with Path(json_data.name).open( 'r') as f:
        data_dic = json.load(f)

    os.unlink(json_data.name)
    os.unlink(json_in.name)

    output = parse_etc_output(data_dic, debug=debug)
    output['etc_settings'] = etc_settings

    return output



def parse_etc_output(data_dic, debug):
    data_collection = {}
    data_collection['N'] = len(data_dic['data']['orders'])
    N = data_collection['N']

    data = lambda i: data_dic['data']['orders'][i]['detectors'][0]['data']

    data_collection['sed_target_fun'] = lambda i: np.array(data(i)['sed']['target']['data'])
    data_collection['obstarget_fun'] =  lambda i: np.array(data(i)['signals']['obstarget']['data'])

    data_collection['obssky_fun'] =  lambda i: np.array(data(i)['signals']['obssky']['data'])

    blaze_fun = lambda i: np.array(data(i)['throughput']['blaze']['data'])
    # fiber_fun = lambda i: np.array(data(i)['throughput']['fibinj']['data'])

    dispersion_fun = lambda i: np.array(data(i)['dispersion']['dispersion']['data'])
    disp_max = np.max([np.nanmax(dispersion_fun(i)) for i in range(N)])
    dispersion_fun = lambda i: np.array(data(i)['dispersion']['dispersion']['data']/disp_max)

    data_collection['h'] = lambda i: blaze_fun(i)*dispersion_fun(i)
    data_collection['b'] = lambda i: blaze_fun(i)
    # data_collection['f'] = lambda i: fiber_fun(i)
    data_collection['d'] = lambda i: dispersion_fun(i)

    #data_collection['throughput_fun'] = lambda i: np.array(data(i)['throughput']['totalinclsky']['data'])
    data_collection['throughput_fun'] = lambda i: np.array(data_collection['obstarget_fun'](i)/data_collection['sed_target_fun'](i))
    #T_max = np.max([np.nanmax(data_collection['throughput_fun'](i)) for i in range(N)])
    #data_collection['throughput_fun'] = lambda i: np.array(data_collection['obstarget_fun'](i)/data_collection['sed_target_fun'](i)/T_max)

    data_collection['atm_fun'] = lambda i: np.array(data(i)['throughput']['atmosphere']['data'])
    data_collection['snr_fun'] = lambda i: np.array(data(i)['snr']['snr']['data'])

    data_collection['wave_fun'] = lambda i: data(i)['wavelength']['wavelength']['data']
    #sed_sky_fun = lambda i: data(i)['sed']['sky']['data']
    instr_vars = lambda i: data(i)['snr']['noise_components']['noise_info']
    data_collection['n'] = lambda i: \
        instr_vars(i)['nspat']*instr_vars(i)['nspec']*instr_vars(i)['ndit']*\
        (instr_vars(i)['dit']*instr_vars(i)['dark'] + instr_vars(i)['ron2'])

#    print(data['data']['orders'][i]['detectors'][0]['data']['snr']['noise_components']['noise_info'].keys())
    N = data_collection['N']
    output = {}
    output['target_SED'], output['wave'] = combine_orders(N,
                                          data_collection['sed_target_fun'],
                                          data_collection['wave_fun'],
                                          )
    output['B_A'], output['B_B'], _ = split_orders(N,
                                 data_collection['obssky_fun'],
                                 data_collection['wave_fun'],
                                 )
    output['T_A'], output['T_B'], _ = split_orders(N,
                                 data_collection['throughput_fun'],
                                 data_collection['wave_fun'],
                                 )
    output['h_A'], output['h_B'], _ = split_orders(N,
                                 data_collection['h'],
                                 data_collection['wave_fun'],
                                 )

    output['n_A'], output['n_B'] = spread_instrument_term(N,
                                 data_collection['n'],
                                 data_collection['wave_fun'],
                                 )

    # debug/comparison
    if debug:
        output['snr_A'], output['snr_B'], _ = split_orders(N,
                                    data_collection['snr_fun'],
                                    data_collection['wave_fun'],
                                    )

        output['S_A'], output['S_B'], _ = split_orders(N,
                                    data_collection['obstarget_fun'],
                                    data_collection['wave_fun'],
                                    )

        output['b_A'], output['b_B'], _ = split_orders(N,
                                    data_collection['b'],
                                    data_collection['wave_fun'],
                                    )

        # output['f_A'], output['f_B'], _ = split_orders(N,
        #                             data_collection['f'],
        #                             data_collection['wave_fun'],
        #                             )
        output['d_A'], output['d_B'], _ = split_orders(N,
                                    data_collection['d'],
                                    data_collection['wave_fun'],
                                    )
    return output

def runtime(input, rv = 0, debug=False, noise_free=False):
    # my function runtime
    st = Harps_spec_info.wave_resolution
    wave = np.arange(Harps_spec_info.desiredMinW-Harps_spec_info.padL*st,Harps_spec_info.desiredMaxW+.001+Harps_spec_info.padR*st,step=st)

    S_v = dopp_shift(wave, input['target_SED'], rv)
    S_A = input['T_A']*S_v
    S_B = input['T_B']*S_v

    inter_region = np.logical_and(~np.isnan(input['T_A']), ~np.isnan(input['T_B']))
    odd_region = np.logical_and(np.isnan(input['T_A']), ~np.isnan(input['T_B']))
    even_region = np.logical_and(np.isnan(input['T_B']), ~np.isnan(input['T_A']))

    S = np.full(S_A.shape, np.nan)
    sigma2 = np.full(S_A.shape, np.nan)

    S[even_region] = S_A[even_region]/input['h_A'][even_region]
    S[odd_region] = S_B[odd_region]/input['h_B'][odd_region]
    S[inter_region] = (S_A[inter_region]/input['h_A'][inter_region] +\
        S_B[inter_region]/input['h_B'][inter_region])/2

    sigma2[even_region] = (S_A[even_region] + input['B_A'][even_region])/\
                        np.power(input['h_A'][even_region], 2) +\
                        input['n_A'][even_region]/np.power(input['h_A'][even_region],2)
    sigma2[odd_region] = (S_B[odd_region] + input['B_B'][odd_region])/\
                        input['h_B'][odd_region]/input['h_B'][odd_region] +\
                            input['n_B'][odd_region]/np.power(input['h_B'][odd_region],2)
    sigma2[inter_region] = ((S_A[inter_region] + input['B_A'][inter_region])/\
                            np.power(input['h_A'], 2)[inter_region] +\
                            (S_B[inter_region] + input['B_B'][inter_region])/\
                            np.power(input['h_B'], 2)[inter_region] +\
                            input['n_A'][inter_region]/np.power(input['h_A'][inter_region],2)
 +\
                            input['n_B'][inter_region]/np.power(input['h_B'][inter_region],2))/4

    sigma = np.power(sigma2, 0.5)

    output = {}
    # main output
    ## debugging (useful for debugging)
    if debug:
        # debugging computations
        output['S'] = S
        output['sigma'] = sigma
        output['S_samp'] = np.random.normal(S, sigma)
        S_comb_err = np.nanmean(np.abs(
            S_A[inter_region]/input['h_A'][inter_region] -
            S_B[inter_region]/input['h_B'][inter_region]))

        sigma_A = np.full(S_A.shape, np.nan)
        sigma_A = np.power(S_A + input['B_A'] + input['n_A'], 0.5)

        sigma_B = np.full(S_B.shape, np.nan)
        sigma_B = np.power(S_B + input['B_B'] + input['n_B'], 0.5)

        output['S_A'] = S_A
        output['S_B'] = S_B
        output['S_comb_err'] = S_comb_err
        output['sigma_A'] = sigma_A
        output['sigma_B'] = sigma_B
        output['snr_A'] = S_A/sigma_A
        output['snr_B'] = S_B/sigma_B
        output['even_region'] = even_region
        output['odd_region'] = odd_region
        output['h_A'] = input['h_A']
        output['h_B'] = input['h_B']
    else:
        # fast output (only what is really necessary)
        if noise_free:
            output['S'] = S
        else:
            output['S_samp'] = np.random.normal(S, sigma)
    return output

def runtime_fast(input, rv = 0, debug=False, noise_free=False):
    # my function runtime
    st = Harps_spec_info.wave_resolution
    wave = np.arange(Harps_spec_info.desiredMinW-Harps_spec_info.padL*st,Harps_spec_info.desiredMaxW+.001+Harps_spec_info.padR*st,step=st)

    # create physical copies
    T_A = input['T_A'].copy()
    T_B = input['T_B'].copy()
    h_A = input['h_A'].copy()
    h_B = input['h_B'].copy()

    S_v = dopp_shift(wave, input['target_SED'], rv)
    S_A = T_A*S_v
    S_B = T_B*S_v

    inter_region = np.logical_and(~np.isnan(T_A), ~np.isnan(T_B))
    odd_region = np.logical_and(np.isnan(T_A), ~np.isnan(T_B))
    even_region = np.logical_and(np.isnan(T_B), ~np.isnan(T_A))

    S = np.full(S_A.shape, np.nan)
    if debug or (not noise_free):
        sigma2 = np.full(S_A.shape, np.nan)

    S[even_region] = S_A[even_region]/h_A[even_region]
    S[odd_region] = S_B[odd_region]/h_B[odd_region]
    S[inter_region] = (S_A[inter_region]/h_A[inter_region] +\
        S_B[inter_region]/h_B[inter_region])/2

    if debug or (not noise_free):
        # create physical copies
        B_A = input['B_A'].copy()
        B_B = input['B_B'].copy()

        n_A = input['n_A'].copy()
        n_B = input['n_B'].copy()

        sigma2[even_region] = (S_A[even_region] + B_A[even_region])/\
                            np.power(h_A[even_region], 2) +\
                            n_A[even_region]/np.power(h_A[even_region],2)
        sigma2[odd_region] = (S_B[odd_region] + B_B[odd_region])/\
                            h_B[odd_region]/h_B[odd_region] +\
                                n_B[odd_region]/np.power(h_B[odd_region],2)
        sigma2[inter_region] = ((S_A[inter_region] + B_A[inter_region])/\
                                np.power(h_A, 2)[inter_region] +\
                                (S_B[inter_region] + B_B[inter_region])/\
                                np.power(h_B, 2)[inter_region] +\
                                n_A[inter_region]/np.power(h_A[inter_region],2)
    +\
                                n_B[inter_region]/np.power(h_B[inter_region],2))/4

        sigma = np.power(sigma2, 0.5)

    output = {}
    # main output
    ## debugging (useful for debugging)
    if debug:
        # debugging computations
        output['S'] = S
        output['sigma'] = sigma
        output['S_samp'] = np.random.normal(S, sigma)
        S_comb_err = np.nanmean(np.abs(
            S_A[inter_region]/input['h_A'][inter_region] -
            S_B[inter_region]/input['h_B'][inter_region]))

        sigma_A = np.full(S_A.shape, np.nan)
        sigma_A = np.power(S_A + input['B_A'] + input['n_A'], 0.5)

        sigma_B = np.full(S_B.shape, np.nan)
        sigma_B = np.power(S_B + input['B_B'] + input['n_B'], 0.5)

        output['S_A'] = S_A
        output['S_B'] = S_B
        output['S_comb_err'] = S_comb_err
        output['sigma_A'] = sigma_A
        output['sigma_B'] = sigma_B
        output['snr_A'] = S_A/sigma_A
        output['snr_B'] = S_B/sigma_B
        output['even_region'] = even_region
        output['odd_region'] = odd_region
        output['h_A'] = input['h_A']
        output['h_B'] = input['h_B']
    else:
        # fast output (only what is really necessary)
        if noise_free:
            output['S'] = S
        else:
            output['S_samp'] = np.random.normal(S, sigma)
    return output

def runtime_fast_2(input, rv = 0, debug=False, noise_free=False):
    # my function runtime
    st = Harps_spec_info.wave_resolution
    wave = np.arange(Harps_spec_info.desiredMinW-Harps_spec_info.padL*st,Harps_spec_info.desiredMaxW+.001+Harps_spec_info.padR*st,step=st)

    S_v = dopp_shift(wave, input['target_SED'], rv)
    S_A = input['T_A']*S_v
    S_B = input['T_B']*S_v

    inter_region = np.logical_and(~np.isnan(input['T_A']), ~np.isnan(input['T_B']))
    odd_region = np.logical_and(np.isnan(input['T_A']), ~np.isnan(input['T_B']))
    even_region = np.logical_and(np.isnan(input['T_B']), ~np.isnan(input['T_A']))

    S = np.full(S_A.shape, np.nan)

    S[even_region] = S_A[even_region]/input['h_A'][even_region]
    S[odd_region] = S_B[odd_region]/input['h_B'][odd_region]
    S[inter_region] = (S_A[inter_region]/input['h_A'][inter_region] +\
        S_B[inter_region]/input['h_B'][inter_region])/2

    if not noise_free:
        sigma2 = np.full(S_A.shape, np.nan)

        sigma2[even_region] = (S_A[even_region] + input['B_A'][even_region])/\
                            np.power(input['h_A'][even_region], 2) +\
                            input['n_A'][even_region]/np.power(input['h_A'][even_region],2)
        sigma2[odd_region] = (S_B[odd_region] + input['B_B'][odd_region])/\
                            input['h_B'][odd_region]/input['h_B'][odd_region] +\
                                input['n_B'][odd_region]/np.power(input['h_B'][odd_region],2)
        sigma2[inter_region] = ((S_A[inter_region] + input['B_A'][inter_region])/\
                                np.power(input['h_A'], 2)[inter_region] +\
                                (S_B[inter_region] + input['B_B'][inter_region])/\
                                np.power(input['h_B'], 2)[inter_region] +\
                                input['n_A'][inter_region]/np.power(input['h_A'][inter_region],2)
    +\
                                input['n_B'][inter_region]/np.power(input['h_B'][inter_region],2))/4

        sigma = np.power(sigma2, 0.5)

    output = {}
    # main output
    ## debugging (useful for debugging)
    if debug:
        # debugging computations
        output['S'] = S
        output['sigma'] = sigma
        output['S_samp'] = np.random.normal(S, sigma)
        S_comb_err = np.nanmean(np.abs(
            S_A[inter_region]/input['h_A'][inter_region] -
            S_B[inter_region]/input['h_B'][inter_region]))

        sigma_A = np.full(S_A.shape, np.nan)
        sigma_A = np.power(S_A + input['B_A'] + input['n_A'], 0.5)

        sigma_B = np.full(S_B.shape, np.nan)
        sigma_B = np.power(S_B + input['B_B'] + input['n_B'], 0.5)

        output['S_A'] = S_A
        output['S_B'] = S_B
        output['S_comb_err'] = S_comb_err
        output['sigma_A'] = sigma_A
        output['sigma_B'] = sigma_B
        output['snr_A'] = S_A/sigma_A
        output['snr_B'] = S_B/sigma_B
        output['even_region'] = even_region
        output['odd_region'] = odd_region
        output['h_A'] = input['h_A']
        output['h_B'] = input['h_B']
    else:
        # fast output (only what is really necessary)
        if noise_free:
            output['S'] = S
        else:
            output['S_samp'] = np.random.normal(S, sigma)
    return output

