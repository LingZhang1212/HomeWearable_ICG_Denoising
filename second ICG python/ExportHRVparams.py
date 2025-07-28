import pandas as pd
import os
from typing import Tuple, Dict, Any

def export_hrv_params(HRVparams: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export HRVparams dictionary into a LaTeX-compatible table and CSV file.

    Parameters:
        HRVparams (dict): Dictionary of various settings for HRV analysis

    Returns:
        T (DataFrame): Flattened DataFrame of HRVparams
        Tab (DataFrame): Table formatted for LaTeX export
    """
    general_params = pd.json_normalize(HRVparams, sep='_')
    T = pd.DataFrame()

    for col in general_params.columns:
        val = general_params.at[0, col]

        if isinstance(val, dict):
            temp_df = pd.json_normalize(val, sep='_')
            # Flatten multi-row frequency limit arrays
            for k, v in temp_df.items():
                if isinstance(v[0], list) and all(isinstance(i, list) for i in v[0]):
                    limits_str = [f"{low}-{high}" for low, high in v[0]]
                    freq_df = pd.DataFrame([limits_str], columns=['ulf', 'vlf', 'lf', 'hf'])
                    T = pd.concat([T, freq_df], axis=1)
                    temp_df[k] = ["[]"]
                elif isinstance(v[0], (list, str)):
                    temp_df[k] = [str(v[0])]
            temp_df.columns = [f"{col}_{subcol}" for subcol in temp_df.columns]
            T = pd.concat([T, temp_df], axis=1)
        else:
            T[col] = [str(val) if not isinstance(val, str) else val.replace('_', ' ')]

    table_names = [col.replace('_', ' ') for col in T.columns]
    table_values = T.values.tolist()[0]

    for i in range(len(table_values)):
        if isinstance(table_values[i], str):
            table_values[i] = table_values[i].replace('\\', '$\\$').replace('_', ' ')
        else:
            table_values[i] = str(table_values[i])

    Tab = pd.DataFrame({"Parameter": table_names, "Value": table_values})

    # Save CSV
    output_dir = HRVparams.get('writedata', 'HRV_Output')
    filename = HRVparams.get('filename', 'HRV_Params')
    csv_path = os.path.join(output_dir, f"ParametersTable_{filename}.csv")
    Tab.to_csv(csv_path, index=False)

    # Save LaTeX if possible
    try:
        latex_path = os.path.join(output_dir, f"ParametersTable_{filename}.tex")
        with open(latex_path, 'w') as f:
            f.write(Tab.to_latex(index=False, escape=True))
    except Exception:
        pass

    return T, Tab
