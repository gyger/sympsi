"""Pauli operators and states"""

from sympy import I, Mul, Integer
from sympsi import (Operator, Commutator, AntiCommutator,
                                   Dagger, IdentityOperator, Ket, Bra)
from sympsi import ComplexSpace
from sympy.matrices import Matrix
from sympy.functions.special.tensor_functions import KroneckerDelta

__all__ = [
    'SigmaX', 'SigmaY', 'SigmaZ', 'SigmaMinus', 'SigmaPlus', 'SigmaZKet',
    'SigmaZBra'
]


class SigmaOpBase(Operator):
    """Pauli sigma operator, base class"""

    @property
    def name(self):
        return self.args[0]

    @property
    def use_name(self):
        return bool(self.args[0]) is not False

    @classmethod
    def default_args(self):
        return (False,)

    def __new__(cls, *args, **hints):
        return Operator.__new__(cls, *args, **hints)

    def _eval_commutator_BosonOp(self, other, **hints):
        return Integer(0)


class SigmaX(SigmaOpBase):
    """Pauli sigma x operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympsi import represent
    >>> from sympsi.pauli import SigmaX
    >>> sx = SigmaX()
    >>> sx
    SigmaX()
    >>> represent(sx)
    Matrix([
    [0, 1],
    [1, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args, **hints)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return 2 * I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return - 2 * I * SigmaY(self.name)

    def _eval_commutator_BosonOp(self, other, **hints):
        return Integer(0)

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return Integer(0)

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return Integer(0)

    def _eval_adjoint(self):
        return self

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_x^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_x}'

    def _print_contents(self, printer, *args):
        return 'SigmaX()'

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaX(self.name).__pow__(int(e) % 2)

    def __mul__(self, other):

        if isinstance(other, SigmaOpBase) and self.name != other.name:
            # Pauli matrices with different labels commute; sort by name
            if self.name < other.name:
                return Mul(self, other)
            else:
                return Mul(other, self)

        if isinstance(other, SigmaX) and self.name == other.name:
            return Integer(1)

        if isinstance(other, SigmaY) and self.name == other.name:
            return I * SigmaZ(self.name)

        if isinstance(other, SigmaZ) and self.name == other.name:
            return - I * SigmaY(self.name)

        if isinstance(other, SigmaMinus) and self.name == other.name:
            return (Integer(1)/2 + SigmaZ(self.name)/2)

        if isinstance(other, SigmaPlus) and self.name == other.name:
            return (Integer(1)/2 - SigmaZ(self.name)/2)

        if isinstance(other, Mul):
            args1 = tuple(arg for arg in other.args if arg.is_commutative)
            args2 = tuple(arg for arg in other.args if not arg.is_commutative)
            x = self
            for y in args2:
                x = x * y
            return Mul(*args1) * x

        return Mul(self, other)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 1], [1, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaY(SigmaOpBase):
    """Pauli sigma y operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympsi import represent
    >>> from sympsi.pauli import SigmaY
    >>> sy = SigmaY()
    >>> sy
    SigmaY()
    >>> represent(sy)
    Matrix([
    [0, -I],
    [I,  0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return 2 * I * SigmaX(self.name)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return - 2 * I * SigmaZ(self.name)

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return Integer(0)

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return Integer(0)

    def _eval_adjoint(self):
        return self

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_y^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_y}'

    def _print_contents(self, printer, *args):
        return 'SigmaY()'

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaY(self.name).__pow__(int(e) % 2)

    def __mul__(self, other):

        if isinstance(other, SigmaOpBase) and self.name != other.name:
            # Pauli matrices with different labels commute; sort by name
            if self.name < other.name:
                return Mul(self, other)
            else:
                return Mul(other, self)

        if isinstance(other, SigmaX) and self.name == other.name:
            return - I * SigmaZ(self.name)

        if isinstance(other, SigmaY) and self.name == other.name:
            return Integer(1)

        if isinstance(other, SigmaZ) and self.name == other.name:
            return I * SigmaX(self.name)

        if isinstance(other, SigmaMinus) and self.name == other.name:
            return -I * (Integer(1) + SigmaZ(self.name))/2

        if isinstance(other, SigmaPlus) and self.name == other.name:
            return I * (Integer(1) - SigmaZ(self.name))/2

        if isinstance(other, Mul):
            args1 = tuple(arg for arg in other.args if arg.is_commutative)
            args2 = tuple(arg for arg in other.args if not arg.is_commutative)
            x = self
            for y in args2:
                x = x * y
            return Mul(*args1) * x

        return Mul(self, other)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, -I], [I, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaZ(SigmaOpBase):
    """Pauli sigma z operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympsi import represent
    >>> from sympsi.pauli import SigmaZ
    >>> sz = SigmaZ()
    >>> sz ** 3
    SigmaZ()
    >>> represent(sz)
    Matrix([
    [1,  0],
    [0, -1]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return 2 * I * SigmaY(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return - 2 * I * SigmaX(self.name)

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return Integer(0)

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return Integer(0)

    def _eval_adjoint(self):
        return self

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_z^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_z}'

    def _print_contents(self, printer, *args):
        return 'SigmaZ()'

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return SigmaZ(self.name).__pow__(int(e) % 2)

    def __mul__(self, other):

        if isinstance(other, SigmaOpBase) and self.name != other.name:
            # Pauli matrices with different labels commute; sort by name
            if self.name < other.name:
                return Mul(self, other)
            else:
                return Mul(other, self)

        if isinstance(other, SigmaX) and self.name == other.name:
            return I * SigmaY(self.name)

        if isinstance(other, SigmaY) and self.name == other.name:
            return - I * SigmaX(self.name)

        if isinstance(other, SigmaZ) and self.name == other.name:
            return Integer(1)

        if isinstance(other, SigmaMinus) and self.name == other.name:
            return - SigmaMinus(self.name)

        if isinstance(other, SigmaPlus) and self.name == other.name:
            return SigmaPlus(self.name)

        if isinstance(other, Mul):
            args1 = tuple(arg for arg in other.args if arg.is_commutative)
            args2 = tuple(arg for arg in other.args if not arg.is_commutative)
            x = self
            for y in args2:
                x = x * y
            return Mul(*args1) * x

        return Mul(self, other)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[1, 0], [0, -1]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaMinus(SigmaOpBase):
    """Pauli sigma minus operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympsi import represent, Dagger
    >>> from sympsi.pauli import SigmaMinus
    >>> sm = SigmaMinus()
    >>> sm
    SigmaMinus()
    >>> Dagger(sm)
    SigmaPlus()
    >>> represent(sm)
    Matrix([
    [0, 0],
    [1, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return -SigmaZ(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        return 2 * self

    def _eval_commutator_SigmaMinus(self, other, **hints):
        return SigmaZ(self.name)

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return Integer(0)

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return Integer(1)

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return - I * Integer(1)

    def _eval_anticommutator_SigmaPlus(self, other, **hints):
        return Integer(1)

    def _eval_adjoint(self):
        return SigmaPlus(self.name)

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return Integer(0)

    def __mul__(self, other):

        if isinstance(other, SigmaOpBase) and self.name != other.name:
            # Pauli matrices with different labels commute; sort by name
            if self.name < other.name:
                return Mul(self, other)
            else:
                return Mul(other, self)

        if isinstance(other, SigmaX) and self.name == other.name:
            return (Integer(1) - SigmaZ(self.name))/2

        if isinstance(other, SigmaY) and self.name == other.name:
            return - I * (Integer(1) - SigmaZ(self.name))/2

        if isinstance(other, SigmaZ) and self.name == other.name:
            # (SigmaX(self.name) - I * SigmaY(self.name))/2
            return SigmaMinus(self.name)

        if isinstance(other, SigmaMinus) and self.name == other.name:
            return Integer(0)

        if isinstance(other, SigmaPlus) and self.name == other.name:
            return Integer(1)/2 - SigmaZ(self.name)/2

        if isinstance(other, Mul):
            args1 = tuple(arg for arg in other.args if arg.is_commutative)
            args2 = tuple(arg for arg in other.args if not arg.is_commutative)
            x = self
            for y in args2:
                x = x * y
            return Mul(*args1) * x

        return Mul(self, other)

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_-^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_-}'

    def _print_contents(self, printer, *args):
        return 'SigmaMinus()'

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 0], [1, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaPlus(SigmaOpBase):
    """Pauli sigma plus operator

    Parameters
    ==========

    name : str
        An optional string that labels the operator. Pauli operators with
        different names commute.

    Examples
    ========

    >>> from sympsi import represent, Dagger
    >>> from sympsi.pauli import SigmaPlus
    >>> sp = SigmaPlus()
    >>> sp
    SigmaPlus()
    >>> Dagger(sp)
    SigmaMinus()
    >>> represent(sp)
    Matrix([
    [0, 1],
    [0, 0]])
    """

    def __new__(cls, *args, **hints):
        return SigmaOpBase.__new__(cls, *args)

    def _eval_commutator_SigmaX(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return SigmaZ(self.name)

    def _eval_commutator_SigmaY(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return I * SigmaZ(self.name)

    def _eval_commutator_SigmaZ(self, other, **hints):
        if self.name != other.name:
            return Integer(0)
        else:
            return -2 * self

    def _eval_commutator_SigmaMinus(self, other, **hints):
        return SigmaZ(self.name)

    def _eval_anticommutator_SigmaZ(self, other, **hints):
        return Integer(0)

    def _eval_anticommutator_SigmaX(self, other, **hints):
        return Integer(1)

    def _eval_anticommutator_SigmaY(self, other, **hints):
        return I * Integer(1)

    def _eval_anticommutator_SigmaMinus(self, other, **hints):
        return Integer(1)

    def _eval_adjoint(self):
        return SigmaMinus(self.name)

    def _eval_mul(self, other):
        return self * other

    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            return Integer(0)

    def __mul__(self, other):

        if other == Integer(1):
            return self

        if isinstance(other, SigmaOpBase) and self.name != other.name:
            # Pauli matrices with different labels commute; sort by name
            if self.name < other.name:
                return Mul(self, other)
            else:
                return Mul(other, self)

        if isinstance(other, SigmaX) and self.name == other.name:
            return (Integer(1) + SigmaZ(self.name))/2

        if isinstance(other, SigmaY) and self.name == other.name:
            return I * (Integer(1) + SigmaZ(self.name))/2

        if isinstance(other, SigmaZ) and self.name == other.name:
            #-(SigmaX(self.name) + I * SigmaY(self.name))/2
            return -SigmaPlus(self.name)

        if isinstance(other, SigmaMinus) and self.name == other.name:
            return (Integer(1) + SigmaZ(self.name))/2

        if isinstance(other, SigmaPlus) and self.name == other.name:
            return Integer(0)

        if isinstance(other, Mul):
            args1 = tuple(arg for arg in other.args if arg.is_commutative)
            args2 = tuple(arg for arg in other.args if not arg.is_commutative)
            x = self
            for y in args2:
                x = x * y
            return Mul(*args1) * x

        return Mul(self, other)

    def _print_contents_latex(self, printer, *args):
        if self.use_name:
            return r'{\sigma_+^{(%s)}}' % str(self.name)
        else:
            return r'{\sigma_+}'

    def _print_contents(self, printer, *args):
        return 'SigmaPlus()'

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[0, 1], [0, 0]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaZKet(Ket):
    """Ket for a two-level system quantum system.

    Parameters
    ==========

    n : Number
        The state number (0 or 1).

    """

    def __new__(cls, n):
        if n not in [0, 1]:
            raise ValueError("n must be 0 or 1")
        return Ket.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return SigmaZBra

    @classmethod
    def _eval_hilbert_space(cls, label):
        return ComplexSpace(2)

    def _eval_innerproduct_SigmaZBra(self, bra, **hints):
        return KroneckerDelta(self.n, bra.n)

    def _apply_operator_SigmaZ(self, op, **options):
        if self.n == 0:
            return self
        else:
            return Integer(-1) * self

    def _apply_operator_SigmaX(self, op, **options):
        return SigmaZKet(1) if self.n == 0 else SigmaZKet(0)

    def _apply_operator_SigmaY(self, op, **options):
        return I * SigmaZKet(1) if self.n == 0 else (-I) * SigmaZKet(0)

    def _apply_operator_SigmaMinus(self, op, **options):
        if self.n == 0:
            return SigmaZKet(1)
        else:
            return Integer(0)

    def _apply_operator_SigmaPlus(self, op, **options):
        if self.n == 0:
            return Integer(0)
        else:
            return SigmaZKet(0)

    def _represent_default_basis(self, **options):
        format = options.get('format', 'sympy')
        if format == 'sympy':
            return Matrix([[1], [0]]) if self.n == 0 else Matrix([[0], [1]])
        else:
            raise NotImplementedError('Representation in format ' +
                                      format + ' not implemented.')


class SigmaZBra(Bra):
    """Bra for a two-level quantum system.

    Parameters
    ==========

    n : Number
        The state number (0 or 1).

    """

    def __new__(cls, n):
        if n not in [0, 1]:
            raise ValueError("n must be 0 or 1")
        return Bra.__new__(cls, n)

    @property
    def n(self):
        return self.label[0]

    @classmethod
    def dual_class(self):
        return SigmaZKet


def _qsimplify_pauli_product(a, b):
    """
    Internal helper function for simplifying products of Pauli operators.
    """
    if not (isinstance(a, SigmaOpBase) and isinstance(b, SigmaOpBase)):
        return Mul(a, b)

    if a.name != b.name:
        # Pauli matrices with different labels commute; sort by name
        if a.name < b.name:
            return Mul(a, b)
        else:
            return Mul(b, a)

    elif isinstance(a, SigmaX):

        if isinstance(b, SigmaX):
            return Integer(1)

        if isinstance(b, SigmaY):
            return I * SigmaZ(a.name)

        if isinstance(b, SigmaZ):
            return - I * SigmaY(a.name)

        if isinstance(b, SigmaMinus):
            return (Integer(1)/2 + SigmaZ(a.name)/2)

        if isinstance(b, SigmaPlus):
            return (Integer(1)/2 - SigmaZ(a.name)/2)

    elif isinstance(a, SigmaY):

        if isinstance(b, SigmaX):
            return - I * SigmaZ(a.name)

        if isinstance(b, SigmaY):
            return Integer(1)

        if isinstance(b, SigmaZ):
            return I * SigmaX(a.name)

        if isinstance(b, SigmaMinus):
            return -I * (Integer(1) + SigmaZ(a.name))/2

        if isinstance(b, SigmaPlus):
            return I * (Integer(1) - SigmaZ(a.name))/2

    elif isinstance(a, SigmaZ):

        if isinstance(b, SigmaX):
            return I * SigmaY(a.name)

        if isinstance(b, SigmaY):
            return - I * SigmaX(a.name)

        if isinstance(b, SigmaZ):
            return Integer(1)

        if isinstance(b, SigmaMinus):
            return - SigmaMinus(a.name)

        if isinstance(b, SigmaPlus):
            return SigmaPlus(a.name)

    elif isinstance(a, SigmaMinus):

        if isinstance(b, SigmaX):
            return (Integer(1) - SigmaZ(a.name))/2

        if isinstance(b, SigmaY):
            return - I * (Integer(1) - SigmaZ(a.name))/2

        if isinstance(b, SigmaZ):
            # (SigmaX(a.name) - I * SigmaY(a.name))/2
            return SigmaMinus(b.name)

        if isinstance(b, SigmaMinus):
            return Integer(0)

        if isinstance(b, SigmaPlus):
            return Integer(1)/2 - SigmaZ(a.name)/2

    elif isinstance(a, SigmaPlus):

        if isinstance(b, SigmaX):
            return (Integer(1) + SigmaZ(a.name))/2

        if isinstance(b, SigmaY):
            return I * (Integer(1) + SigmaZ(a.name))/2

        if isinstance(b, SigmaZ):
            #-(SigmaX(a.name) + I * SigmaY(a.name))/2
            return -SigmaPlus(a.name)

        if isinstance(b, SigmaMinus):
            return (Integer(1) + SigmaZ(a.name))/2

        if isinstance(b, SigmaPlus):
            return Integer(0)

    else:
        return a * b


def qsimplify_pauli(e):
    """
    Simplify an expression that includes products of pauli operators.

    Parameters
    ==========

    e : expression
        An expression that contains products of Pauli operators that is
        to be simplified.

    Examples
    ========

    >>> from sympy.physics.quantum.pauli import SigmaX, SigmaY
    >>> from sympy.physics.quantum.pauli import qsimplify_pauli
    >>> sx, sy = SigmaX(), SigmaY()
    >>> sx * sy
    SigmaX()*SigmaY()
    >>> qsimplify_pauli(sx * sy)
    I*SigmaZ()
    """
    if isinstance(e, Operator):
        return e

    if isinstance(e, (Add, Pow, exp)):
        t = type(e)
        return t(*(qsimplify_pauli(arg) for arg in e.args))

    if isinstance(e, Mul):

        c, nc = e.args_cnc()

        nc_s = []
        while nc:
            curr = nc.pop(0)

            while (len(nc) and
                   isinstance(curr, SigmaOpBase) and
                   isinstance(nc[0], SigmaOpBase) and
                   curr.name == nc[0].name):

                x = nc.pop(0)
                y = _qsimplify_pauli_product(curr, x)
                c1, nc1 = y.args_cnc()
                curr = Mul(*nc1)
                c = c + c1

            nc_s.append(curr)

        return Mul(*c) * Mul(*nc_s)

    return e

