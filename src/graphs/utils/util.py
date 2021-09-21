from numpy.random import binomial, geometric, poisson


def generate_pdf_parameter_dicts(size: int, distribution: str, diag_prob: float, out_prob: float, n: int = 4):
    pdf: list = []
    _in: dict = dict()
    _out: dict = dict()
    if distribution == 'discrete-binomial':
        pdf = [[binomial] * size] * size
        _in = dict(n=n, p=diag_prob)
        _out = dict(n=n, p=out_prob)
    elif distribution == 'discrete-geometric':
        pdf = [[geometric] * size] * size
        _in = dict(p=diag_prob)
        _out = dict(p=out_prob)
    elif distribution == 'discrete-poisson':
        pdf = [[poisson] * size] * size
        _in = dict(lam=diag_prob)
        _out = dict(lam=out_prob)

    parameters = []
    for ii in range(size):
        tmp_param = []
        for jj in range(size):
            if jj == ii:
                tmp_param.append(_in)
            else:
                tmp_param.append(_out)
        parameters.append(tmp_param)

    return pdf, parameters
