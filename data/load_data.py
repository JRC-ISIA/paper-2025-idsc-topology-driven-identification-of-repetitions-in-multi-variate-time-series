import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_dataset(fp: str):
    df = pd.read_parquet(fp, engine="fastparquet") 
    df['time'] = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.set_index(np.arange(df.shape[0]))

    # Throw away know constant and trend features
    constant_features = [
	    'Control.mcf.extTempCtrl.mpTempController1.in.setTemperature',
	    'Control.mcf.extTempCtrl.mpTempController2.in.setTemperature',
	    'Control.mcf.extTempCtrl.mpTempController3.in.setTemperature', 
	    'Control.mcf.extTempCtrl.mpTempController4.in.setTemperature',
	    'Control.mcf.extTempCtrl.mpTempController5.in.setTemperature',
	    ]
    df = df.drop(columns=constant_features)

    cycle_counter = df['Control.imm.status.productionCycleCounter'].to_numpy()
    df = df.drop(columns=['Control.imm.status.productionCycleCounter', 'time'])

    # cycle transition points
    T = np.where(np.diff(cycle_counter) != 0)[0] + 1
    T = T[5:] 

    # cycle durations
    tau = T[1:] - T[:-1]

    sections = {
	    'periodic'  : {'noiseless': T[0:21],   'noise': T[20:41]},
	    'repetitive': {'noiseless': T[40:66],  'noise': T[65:91]},
	    'recurring':  {'noiseless': T[90:106], 'noise': T[105:120]},
    }

    periods = {
	    'periodic'  : {'noiseless': tau[0:20],   'noise': tau[20:40]},
	    'repetitive': {'noiseless': tau[40:65],  'noise': tau[65:90]},
	    'recurring' : {'noiseless': tau[90:105], 'noise': tau[105:119]},
    }

    X = MinMaxScaler().fit_transform(df.to_numpy()).astype(np.float64)

    return X, sections, periods, 1000

def load_data(fp):
    """
    returns: the full dataset, section cycle transition points, cycle durations within each section, sampling freq [Hz]
    """
    return prepare_dataset(fp)

def load_section(fp, section, noise):
    """
    returns: section of dataset, cycle durations, sampling freq [Hz]
    """
    X, sections, periods, fs = prepare_dataset(fp)

    x = X[sections[section][noise][0]:sections[section][noise][-1]]
    y = np.array(periods[section][noise])

    return x, y, fs

def test_section_plot(fp, section, dim=10):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    x, y, fs = load_section(fp, section, 'noiseless')
    t = np.arange(0, len(x)) / fs

    x_noise, y_noise, _ = load_section(fp, section, 'noise')
    t_noise = np.arange(0, len(x_noise)) / fs


    _, (ax0, ax1) = plt.subplots(2, 1, figsize=[15, 7])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax0.plot(t, x[:, 10], color=colors[1])
    ax0.vlines(np.cumsum(y) / fs, 0, 1, color=colors[0])
    ax1.plot(t_noise, x_noise[:, 10], color=colors[1])
    ax1.vlines(np.cumsum(y_noise) / fs, 0, 1, color=colors[0])
    ax1.set_xlabel('t [sec]')

    ax0.set_title(f'({section}, noisless), dim={dim}')
    ax1.set_title(f'({section}, noise), dim={dim}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    sections = ['periodic', 'repetitive', 'recurring'] 
    test_section_plot('data/IMM.parquet', sections[0])
    test_section_plot('data/IMM.parquet', sections[1])
    test_section_plot('data/IMM.parquet', sections[2])