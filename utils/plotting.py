import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np


def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_spectrogram_to_numpy(spectrogram, feature_type='auto'):
    """
    Plot spectrogram/filterbank features for visualization
    
    Args:
        spectrogram: 2D array to plot
        feature_type: 'filterbank', 'spectrogram', or 'auto'
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower',
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Time Frames')
    
    # Smart ylabel based on feature type
    if feature_type == 'auto':
        # Auto-detect based on common dimensions
        freq_bins = spectrogram.shape[0] if len(spectrogram.shape) > 1 else spectrogram.shape[-1]
        if freq_bins == 128:
            ylabel = 'Filterbank Bins (VoiceFilter-Lite 2020)'
        elif freq_bins == 384:
            ylabel = 'Stacked Filterbank Features (128×3)'
        elif freq_bins == 601:
            ylabel = 'Frequency Bins (VoiceFilter 2019)'
        else:
            ylabel = 'Feature Channels'
    elif feature_type == 'filterbank':
        ylabel = 'Filterbank Bins'
    elif feature_type == 'spectrogram':
        ylabel = 'Frequency Bins'
    else:
        ylabel = 'Channels'
    
    plt.ylabel(ylabel)
    plt.tight_layout()

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data
