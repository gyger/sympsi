"""Quantum mechanical operators.

TODO:

* Fix early 0 in apply_operators.
* Debug and test apply_operators.
* Get cse working with classes in this file.
* Doctests and documentation of special methods for InnerProduct, Commutator,
  AntiCommutator, represent, apply_operators.
"""

from __future__ import print_function, division

from sympy import Derivative, Expr, Integer, oo, Mul, Symbol, latex, Pow, Add
from sympy.matrices import eye
from sympy.printing.pretty.stringpict import prettyForm

from sympy.physics.quantum import Operator, Dagger
from sympy.physics.quantum.qexpr import QExpr, dispatch_method

__all__ = [
    'OperatorFunction'
]

#-----------------------------------------------------------------------------
# OperatorFunction
#-----------------------------------------------------------------------------

def _replace_op_func(e, variable):

    if isinstance(e, Operator):
        return OperatorFunction(e, variable)
    
    if e.is_Number:
        return e
    
    if isinstance(e, Pow):
        return Pow(_replace_op_func(e.base, variable), e.exp)
    
    new_args = [_replace_op_func(arg, variable) for arg in e.args]
    
    if isinstance(e, Add):
        return Add(*new_args)
    
    elif isinstance(e, Mul):
        return Mul(*new_args)
    
    else:
        return e
        
class OperatorFunction(Operator):

    @property
    def operator(self):
        return self.args[0]

    @property
    def variable(self):
        return self.args[1]

    @property
    def free_symbols(self):
        return self.operator.free_symbols.union(self.variable.free_symbols)

    @classmethod
    def default_args(self):
        return (Operator("a"), Symbol("t"))

    def __call__(self, value):
        return OperatorFunction(self.operator, value)
    
    def __new__(cls, *args, **hints):
        if not len(args) in [2]:
            raise ValueError('2 parameters expected, got %s' % str(args))

        return Operator.__new__(cls, *args)

    def __mul__(self, other):

        if (isinstance(other, OperatorFunction) and
                str(self.variable) == str(other.variable)):                
            factors = (self.operator * other.operator).expand()

            factors_t = _replace_op_func(factors, self.variable)
            
            return factors_t

        return Mul(self, other)


    def _eval_power(self, e):
        if e.is_Integer and e.is_positive:
            x = self.operator._eval_power(e)
            if x:
                if isinstance(x, Operator):
                    return OperatorFunction(x, self.variable)
                else:
                    return x
            else:
                return None

    def _eval_commutator_OperatorFunction(self, other, **hints):
        from sympy.physics.quantum import Commutator

        if self.operator.args[0] == other.operator.args[0]:
            if str(self.variable) == str(other.variable):
                return Commutator(self.operator, other.operator).doit()

        return None

    def _eval_adjoint(self):
        return OperatorFunction(Dagger(self.operator), self.variable)
    
    def _print_contents_latex(self, printer, *args):
        return r'{{%s}(%s)}' % (latex(self.operator), latex(self.variable))
