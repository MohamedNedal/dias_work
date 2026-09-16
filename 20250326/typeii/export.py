"""Formatting and file output for tables and the results text.
"""
import numpy as np
import pandas as pd

from .config import MU, MU0, M_P
from .models import B_dulk_mclean, B_mann2023


def df_to_latex(df, caption, label, float_fmt='{:.4g}'):
    """Render a table as a LaTeX tabular."""
    def esc(x):
        if isinstance(x, float):
            return '' if not np.isfinite(x) else float_fmt.format(x)
        s = str(x)
        for a, b in [('\\', r'\textbackslash '), ('_', r'\_'), ('%', r'\%'), ('&', r'\&'),
                     ('#', r'\#'), ('^', r'\^{}'), ('~', r'\~{}')]:
            s = s.replace(a, b)
        return s
    cols = list(df.columns)
    out = ['\\begin{table}[htbp]', '\\centering',
           f'\\caption{{{caption}}}', f'\\label{{{label}}}',
           '\\begin{tabular}{' + 'l' * len(cols) + '}', '\\hline',
           ' & '.join(esc(c) for c in cols) + ' \\\\', '\\hline']
    for _, r in df.iterrows():
        out.append(' & '.join(esc(v) for v in r.tolist()) + ' \\\\')
    out += ['\\hline', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)

def decimals(e, min_dp=1, max_dp=5):
    """Decimal places to use for a value given its uncertainty."""
    if e is None or not np.isfinite(e) or e <= 0:
        return min_dp
    return int(np.clip(-np.floor(np.log10(abs(e))) + 1, min_dp, max_dp))

def fmt_value(v, e, min_dp=1, sep=' +/- '):
    """Value +/- error, to a sensible number of decimals."""
    if not np.isfinite(v):
        return 'n/a'
    nd = decimals(e, min_dp)
    if e is None or not np.isfinite(e) or e <= 0:
        return f'{v:.{nd}f}'
    return f'{v:.{nd}f}{sep}{e:.{nd}f}'

def fmt_sci(v, nd=1):
    """Scientific notation as LaTeX inline maths."""
    if not np.isfinite(v) or v == 0:
        return 'n/a'
    ex = int(np.floor(np.log10(abs(v))))
    return f'${v / 10 ** ex:.{nd}f} \\times 10^{{{ex}}}$'

def fmt_tex(v, e, min_dp=1):
    """Value +/- error wrapped as LaTeX inline maths."""
    if not np.isfinite(v):
        return 'n/a'
    nd = decimals(e, min_dp)
    if e is None or not np.isfinite(e) or e <= 0:
        return f'${v:.{nd}f}$'
    return f'${v:.{nd}f} \\pm {e:.{nd}f}$'

def field_context(r_, B_):
    """Compare a measured field with the published radial profiles by evaluating
    them.

    At the heights of this burst the profiles are themselves a factor of two
    apart, so one measurement cannot agree with all of them and which it matches
    is a result rather than a formality.
    """
    if not (np.isfinite(r_) and np.isfinite(B_)) or B_ <= 0 or r_ <= 1:
        return ''
    refs = [('Dulk \\& McLean (1978)', 'dulk1978', float(np.atleast_1d(B_dulk_mclean(r_))[0])),
            ('Mann et al. (2023)', 'mann2023', float(np.atleast_1d(B_mann2023(r_))[0]))]
    refs = [x for x in refs if np.isfinite(x[2]) and x[2] > 0]
    if not refs:
        return ''
    listed = ' and '.join(f'{v:.2f}~G \\citep{{{k}}}' for _, k, v in refs)
    ranked = sorted(((abs(np.log(B_ / v)), n, v) for n, _, v in refs))
    clauses = []
    for lg, n, v in ranked:
        if lg < 0.18:                                    # within ~20%, i.e. the model spread itself
            clauses.append(f'agrees with {n} to {100 * abs(B_ / v - 1):.0f}\\%')
        else:
            clauses.append(f'sits a factor of {max(B_ / v, v / B_):.1f} '
                           f'{"above" if B_ > v else "below"} {n}')
    out = [f'At that height the published radial profiles give {listed}, so the measurement '
           + ' but '.join(clauses) + '.']
    if len(refs) > 1:
        sp = max(x[2] for x in refs) / min(x[2] for x in refs)
        out.append(f'Those two profiles are themselves a factor of {sp:.1f} apart here, so '
                   f'consistency with either one on its own is a weak constraint.')
    return ' '.join(out)
