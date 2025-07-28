def matrix2latex(matrix, filename, **kwargs):
    """
    Convert a matrix to a LaTeX tabular and write to a file.

    Parameters:
    - matrix: 2D list or numpy array (numeric or string)
    - filename: output .tex file path
    - Optional kwargs:
        - rowLabels: list of row labels
        - columnLabels: list of column labels
        - alignment: 'l', 'c', or 'r'
        - format: format string like '%.2f'
        - size: LaTeX font size (e.g., 'tiny', 'Large')
    """
    import numpy as np

    rowLabels = kwargs.get('rowLabels', [])
    colLabels = kwargs.get('columnLabels', [])
    alignment = kwargs.get('alignment', 'l')
    fmt = kwargs.get('format', None)
    textsize = kwargs.get('size', None)

    if isinstance(matrix, np.ndarray):
        matrix = matrix.tolist()

    height = len(matrix)
    width = len(matrix[0]) if height > 0 else 0

    # Convert all elements to string
    matrix_str = []
    for row in matrix:
        row_str = []
        for val in row:
            if fmt:
                row_str.append(fmt % val if isinstance(val, (int, float)) else str(val))
            else:
                row_str.append(str(val))
        matrix_str.append(row_str)

    with open(filename, 'w') as f:
        if textsize:
            f.write(f'\\begin{{{textsize}}}\n')

        # Begin tabular
        f.write('\\begin{tabular}{|')
        if rowLabels:
            f.write('l|')
        f.write(''.join([f'{alignment}|' for _ in range(width)]))
        f.write('}\n\\hline\n')

        # Column labels
        if colLabels:
            if rowLabels:
                f.write('&')
            f.write('&'.join([f'\\textbf{{{col}}}' for col in colLabels]))
            f.write('\\\\\\hline\n')

        # Data rows
        for i, row in enumerate(matrix_str):
            if rowLabels:
                f.write(f'\\textbf{{{rowLabels[i]}}}&')
            f.write('&'.join(row))
            f.write('\\\\\\hline\n')

        f.write('\\end{tabular}\n')

        if textsize:
            f.write(f'\\end{{{textsize}}}\n')
