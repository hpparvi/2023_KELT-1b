import warnings
from pathlib import Path
from typing import Union, Optional, List

from astropy.table import Table
from numpy import median, array, full, concatenate, diff, sqrt, ones

import astropy.io.fits as pf


def read_tess_spoc(tic: int,
                   datadir: Union[Path, str],
                   sectors: Optional[Union[List[int], str]] = 'all',
                   use_pdc: bool = False,
                   remove_contamination: bool = True,
                   use_quality: bool = True):

    def file_filter(f, partial_tic, sectors):
        _, sector, tic, _, _ = f.name.split('-')
        if sectors != 'all':
            return int(sector[1:]) in sectors and str(partial_tic) in tic
        else:
            return str(partial_tic) in tic

    files = [f for f in sorted(Path(datadir).glob('tess*_lc.fits')) if file_filter(f, tic, sectors)]
    fcol = 'PDCSAP_FLUX' if use_pdc else 'SAP_FLUX'
    times, fluxes, sectors, quality = [], [], [], []
    for f in files:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tb = Table.read(f)
        bjdrefi = tb.meta['BJDREFI']
        df = tb.to_pandas().dropna(subset=['TIME', fcol])
        q = df['QUALITY'].values.copy()
        m = (q == 0) if use_quality else ones(df.shape[0], bool)
        quality.append(q[m])
        times.append(df['TIME'].values[m].copy() + bjdrefi)
        fluxes.append(array(df[fcol].values[m], 'd'))
        fluxes[-1] /= median(fluxes[-1])
        if use_pdc and not remove_contamination:
            contamination = 1 - tb.meta['CROWDSAP']
            fluxes[-1] = contamination + (1 - contamination) * fluxes[-1]
        sectors.append(full(fluxes[-1].size, pf.getval(f, 'sector')))

    return (concatenate(times), concatenate(fluxes), concatenate(sectors), concatenate(quality),
            [diff(f).std() / sqrt(2) for f in fluxes])
