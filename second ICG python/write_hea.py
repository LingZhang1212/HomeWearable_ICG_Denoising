def write_hea(record_name, fs, datapoints, annotator, gain, offset, unit='mV'):
    """
    Write a WFDB-compatible .hea header file.

    Parameters:
        record_name (str): Name of the record (without extension)
        fs (int): Sampling frequency (Hz)
        datapoints (int): Total number of data points
        annotator (str): Annotation type (e.g., 'jqrs')
        gain (int): ADC gain (adu/unit, e.g., 2000 for mV)
        offset (int): Baseline offset (not used in .hea here)
        unit (str): Unit of the signal (default 'mV')
    """
    numsig = 1  # Number of signals
    filename = f"{record_name}.{annotator}"
    hea_filename = f"{record_name}.hea"

    with open(hea_filename, 'w') as f:
        f.write(f"{record_name} {numsig} {fs} {datapoints}\n")
        f.write(f"{filename} 16+24 {gain}/{unit} 12\n")
        f.write("#Creator: HRV_toolbox write_hea.py\n")
