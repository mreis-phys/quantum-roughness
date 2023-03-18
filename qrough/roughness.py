from mpmath import mpf, fac, power, sqrt, hyp2f1, matrix


def binom(a, b):
    return fac(a) / (fac(b) * fac(a - b))


def r2fock(k):
    """Squared roughness of a Fock number state `k`. """
    rq = binom(2 * k, k) / power(2, 2 * k + 1)
    rwq = ((mpf(4) / 3) * power(-3, -k) *
           hyp2f1(-k, k + 1, 1, 4 / 3))
    return float(1 + rq - rwq)


def _s2(n, m, el):
    s2a = sqrt(fac(n + el) * fac(m + el) / (fac(n) * fac(m))) * (-1) ** n
    s2b = power(2, el + 3) * hyp2f1(-n, m + el + 1, el + 1, 4 / 3) * power(3, -m - el - 1) / fac(el)
    return s2a * s2b


def r2a(coef, n, m, min_coef=1e-15):
    """Evaluates the coefficient (n, m) for the 2 index sum."""
    if abs(coef) <= min_coef:
        return 0
    du = coef * (binom(n + m, n) * power(2, -n - m - 1)
                 - 4 * hyp2f1(-n, m + 1, 1, 4 / 3) * power(3, -m - 1) * (-1) ** n)
    return du


def r2b(coef, n, m, el, min_coef=1e-15):
    """Evaluates the coefficient (n, m, el) for the 3 index sum."""
    if abs(coef) <= min_coef:
        return 0
    s1 = power(2, -n - m - el) * fac(n + m + el) / sqrt(fac(n) * fac(m) * fac(n + el) * fac(m + el))
    s2 = _s2(n, m, el)
    return coef * (s1 - s2)


def rho_indexed(rho):
    """Organizes matrix elements according to their usage in calculations."""
    size = len(rho)
    rho_adj = rho.conjugate()
    two_index = [((rho[n, n] * rho[m, m]).real, n, m) for n in range(size)
                 for m in range(size)]
    three_index = [((rho_adj[n, n + el] * rho[m, m + el]).real, n, m, el)
                   for n in range(size) for m in range(size) for el in range(1, min(size - n, size - m))]
    return two_index, three_index


def squared_rough(rho, min_coef=1e-15):
    """Calculates the squared roughness of a quantum state.

    This is simply the squared value of `rough(rho, min_coef)`,
    see its docstring for more info.

    Parameters
    ----------
    rho : 2D array like
        The matrix representing the quantum state.
    min_coef : float, optional
        The minimum absolute value for the product
        of matrix elements to be considered in the
        calculations. If lower than this, the result
        will be treated as zero. The default value is
        1e-15.

    Returns
    -------
    roughness : mpf
        A real number in the interval [0, 1] represented
        as a mpmath's mpf obect. Works like a float.

    """
    rho = matrix(rho)
    size = len(rho)
    two_index, three_index = rho_indexed(rho)
    rho2 = rho ** 2
    t1 = sum([rho2[k, k] for k in range(size)])
    t2_l = [r2a(*c, min_coef) for c in two_index]
    t3_l = [r2b(*c, min_coef) for c in three_index]
    return t1.real + sum(t2_l) + sum(t3_l)


def rough(rho, min_coef=1e-15):
    """Calculates the roughness of a quantum state.

    Given a matrix `rho` representing a quantum state
    in Fock states basis, returns a real number in the
    interval [0, 1]. The returned value is an `mpf`
    object which can be converted to float as needed.

    Parameters
    ----------
    rho : 2D array like
        The matrix representing the quantum state.
    min_coef : float, optional
        The minimum absolute value for the product
        of matrix elements to be considered in the
        calculations. If lower than this, the result
        will be treated as zero. The default value is
        1e-15.

    Returns
    -------
    roughness : mpf
        A real number in the interval [0, 1] represented
        as a mpmath's mpf obect. Works like a float.

    """
    return sqrt(squared_rough(rho, min_coef))
