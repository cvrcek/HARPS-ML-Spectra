from sklearn.feature_selection import mutual_info_regression
import numpy as np
from astropy.stats import sigma_clip
from pdb import set_trace

def get_labels(names, csv_file):
    for i in range(0, len(names)):
        names[i] = os.path.splitext(os.path.split(names[i])[1])[0]
    df = pd.read_csv(csv_file)
    labels = df[df['dp_id'].isin(names)]
    labels = labels.reset_index()
    return labels

def get_significant_codes_ind(codes):
    return mad(codes) > 0.025


# entry point
def get_MIs(codes, names):
    codes_ind = get_significant_codes_ind(codes)
    codes_active = codes[:,codes_ind]
    labels = get_labels(names)
    # visualisation of codes + significant codes
    codes_idx = np.arange(len(codes_ind))[codes_ind]
    plt.figure()
    plt.plot(mad(codes))
    plt.plot(codes_idx, mad(codes)[codes_ind], 'o')
    plt.title(codes_idx)
    plt.show()
    # add significant codes to labels
    for i in range(0,codes_active.shape[1]):
        labels['code_' + str(i)] = codes_active[:,i]
    # compute MI for selection of labels vs every active code and print resulting matrix
    stellar_params = ['radvel','Teff','Mass','[M/H]','airmass','snr','vsini', 'logg']
    #stellar_params = ['radvel']
    nbinss = [5,10,20,40,80,160]

    MI = np.zeros(len(codes_active))
    MIs = np.zeros([len(stellar_params), \
                    codes.shape[1],\
                    len(nbinss)])
    for li,stellar_param in enumerate(stellar_params):

        label = labels.loc[:,stellar_param].values
        ind_mask = ~np.isnan(label)
        label_ = sigma_clip(label,sigma=5,masked=True)
        ind_mask = ind_mask & (~label_.mask)
        label_ = label_.data[ind_mask]

        for ci,codes_ in enumerate(codes.T):
          #  set_trace()
            codes_ = codes_[ind_mask]
            for b,nbins in enumerate(nbinss):
                MIs[li,ci, b] = mutual_info_regression(codes_.reshape(-1, 1),label_)

    # get maximum over codes
    MIs_max  = np.amax(MIs,1)
    #MIs_max = np.repeat(MIs_max[:, :, np.newaxis], MIs.shape[2], axis=2)
    for li,stellar_param in enumerate(stellar_params):
        for b,nbins in enumerate(nbinss):
            MIs[li,:, b] /=MIs_max[li,b]

    MIs = np.mean(MIs, axis = 2)
    MIs = MIs[:,codes_ind]
    return MIs, stellar_params, codes_idx

# entry point
def get_MIs_by_nearest_neighbor(codes, labels):
    # compute mutual information table
    # codes  ... array, shape(n_samples, n_codes)
    # labels ... array, shape(n_samples, n_labels)
    #
    # WARNING!: case n_codes == n_labels will accept transpose matrices, resulting in MI
    #
    assert(codes.shape[0] == labels.shape[0]) # rows are numbers of samples and must be equal
    assert(codes.shape[0] > codes.shape[1]) # I expecting more samples than features (this can discover bugs)
    MI_table = np.zeros((codes.shape[1],  labels.shape[1]))
    
    for li, label in enumerate(labels.T):
        for ci, code in enumerate(codes.T):                
            ind = ~np.logical_or(np.isnan(label), np.isnan(code))
            # I need new variables, code, label are views!
            MI_table[ci,li] = mutual_info_regression(code[ind].reshape(-1, 1),label[ind])

    return MI_table



def get_MIs_for_model(dataloader, model):
    assert(dataloader.dataset.dataset.watched_labels == model.watched_labels)

    codes = np.zeros((len(dataloader.dataset), 1, model.bottleneck))
    labels = np.zeros((len(dataloader.dataset), 1, len(model.watched_labels)))

    for idx, (data, labels_) in tqdm(enumerate(dataloader)):
        idxs = slice(idx, (idx+dataloader.batch_size),1)
        labels[idxs, :,:] = labels_
        codes[idxs,0,:] = model(data)[2].detach().numpy()

    codes = codes[:,:len(model.watched_labels)]

    return get_MIs_by_nearest_neighbor(codes, labels)
