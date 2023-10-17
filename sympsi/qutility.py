"""
Utitility functions for working with operator transformations in
sympsi.
"""

__all__ = [
    'show_first_few_terms',
    'html_table',
    'exchange_integral_order',
    'pull_outwards',
    'push_inwards',
    'integral_pow_expand',
    'sum_pow_expand',
    'replace_dirac_delta',
    'replace_kronecker_delta',
    'expression_tree_transform',
    'qsimplify',
    'pauli_represent_minus_plus',
    'pauli_represent_x_y',
    'split_coeff_operator',
    'extract_operators',
    'extract_operator_products',
    'extract_all_operators',
    'operator_order',
    'operator_sort_by_order',
    'drop_terms_containing',
    'drop_c_number_terms',
    'subs_single',
    'SymEq',
    'recursive_commutator',
    'ncollect',
    'get_coefficient',
    'bch_expansion',
    'unitary_transformation',
    'hamiltonian_transformation',
    'lindblad_dissipator',
    'master_equation',
    'operator_lindblad_dissipator',
    'operator_master_equation',
    'semi_classical_eqm',
    'semi_classical_eqm_matrix_form'
    ]

import warnings
from collections import namedtuple
from sympy import (Add, Mul, Pow, Number, exp, latex, Integral, Sum, Integer, Symbol,
                   I, pi, simplify, oo, DiracDelta, KroneckerDelta, collect,
                   factorial, diff, Function, Derivative, Eq, symbols, S,
                   Matrix, Equality, MatMul, Dummy, rcollect, Poly, O, linsolve)

from sympy.core.sympify import _sympify
from sympy.core.relational import Relational
from sympy import (sin, cos, sinh, cosh)

from sympy.physics.quantum import Operator, Commutator, Dagger
from sympy.physics.quantum.pauli import (SigmaX, SigmaY, SigmaMinus, SigmaPlus)

# TODO: figure out whether we can simply use the sympy version!!
#from sympy.physics.quantum.operatorordering import normal_ordered_form
from sympy.physics.quantum.operatorordering import normal_ordered_form
from sympy.physics.quantum.expectation import Expectation
from sympsi.support import acollect # this collect ignores commutations.

debug = False

# Hack to get past Sympy's Dagger implementation breaking instance checking
daggercls = Dagger(Dummy()).__class__

# -----------------------------------------------------------------------------
# IPython notebook related functions
#
#from IPython.display import display_latex
from IPython.display import Latex, HTML


def show_first_few_terms(e, n=10):
    if isinstance(e, Add):
        e_args_trunc = e.args[0:n]
        e = Add(*(e_args_trunc))

    return Latex("$" + latex(e).replace("dag", "dagger") + r"+ \dots$")


def html_table(data):
    t_table = "<table>\n%s\n</table>"
    t_row = "<tr>%s</tr>"
    t_col = "<td>%s</td>"
    table_code = t_table % "".join(
        [t_row % "".join([t_col % ("$%s$" %
                                   latex(col).replace(r'\dag', r'\dagger'))
                          for col in row])
         for row in data])

    return HTML(table_code)


# -----------------------------------------------------------------------------
# Simplification of integrals
#
def exchange_integral_order(e):
    """
    exchanging integral order. Works in this way:
    ∫(∫ ... (∫(∫    dx_0)dx_1)... dx_n-1)dx_n -->
    ∫(∫ ... (∫(∫  dx_1)dx_2)... dx_n)dx_0
    """
    if isinstance(e, Add):
        return Add(*[exchange_integral_order(arg) for arg in e.args])
    if isinstance(e, Mul):
        return Mul(*[exchange_integral_order(arg) for arg in e.args])
    if isinstance(e, Integral):
        i = push_inwards(e)
        func, lims = i.function, i.limits
        if len(lims) > 1:
            args = [func]
            for idx in range(1, len(lims)):
                args.append(lims[idx])
            args.append(lims[0])
            return(Integral(*args))
    else:
        return e


def _pull_outwards_sum(e, add=True, _n=0):
    f = pull_outwards(e.function, add=add, _n=_n+1)
    dvar = e.variables
    if add and isinstance(f.expand(), Add):
        f = f.expand()
        add_args = []
        for term in f.args:
            args = [term]
            for lim in e.limits:
                args.append(lim)
            add_args.append(Sum(*args))
        ne = Add(*add_args)
        return pull_outwards(ne, add=add, _n=_n+1)
    if isinstance(f, Mul):
        c = [arg for arg in f.args if not isinstance(arg, Sum)]
        s_in = [arg for arg in f.args if isinstance(arg, Sum)]

        const = [arg for arg in c if dvar[0] not in arg.free_symbols]
        nconst = [arg for arg in c if dvar[0] in arg.free_symbols]

        if len(dvar) == 1:
            return Mul(*const) * Sum(Mul(*nconst) * Mul(*s_in), e.limits[0])
        else:
            args = [Mul(*const) * Sum(Mul(*nconst) * Mul(*s_in), e.limits[0])]
            for lim in e.limits[1:]:
                args.append(lim)
            return Sum(*args)
    return pull_outwards(Sum(f, e.limits), add=add, _n=_n+1)


def _pull_outwards_integral(e, add=True, _n=0):
    f = e.function
    if isinstance(f, Sum):  # ∫ ∑ [...]  --> ∑ ∫ [...]
        return pull_outwards(Sum(Integral(f.function, e.limits), f.limits),
                             add=add, _n=_n+1)

    f = pull_outwards(e.function, add=add, _n=_n+1)
    dvar = e.variables
    if add and isinstance(f.expand(), Add):
        f = f.expand()
        add_args = []
        for term in f.args:
            args = [term]
            for lim in e.limits:
                args.append(lim)
            add_args.append(Integral(*args))
        return pull_outwards(Add(*add_args), add=add, _n=_n+1)
    if isinstance(f, Mul):
        c = [arg for arg in f.args if not (isinstance(arg, Integral)
                                           or isinstance(arg, Sum))]
        i_in = [arg for arg in f.args if isinstance(arg, Integral)]
        s_in = [arg for arg in f.args if isinstance(arg, Sum)]

        if not s_in == []:  # First, take summations out of the integrand
            nfunc = Mul(*c) * s_in[0].function * Mul(*s_in[1:]) * Mul(*i_in)
            return pull_outwards(Sum(Integral(nfunc, e.limits),
                                     s_in[0].limits), add=add, _n=_n+1)

        const = [arg for arg in c if dvar[0] not in arg.free_symbols]
        nconst = [arg for arg in c if dvar[0] in arg.free_symbols]

        if len(dvar) == 1:
            return Mul(*const) * Integral(Mul(*nconst) * Mul(*i_in),
                                          e.limits[0])
        else:
            args = [Mul(*const) * Integral(Mul(*nconst) * Mul(*i_in),
                    e.limits[0])]
            for lim in e.limits[1:]:
                args.append(lim)
            return pull_outwards(Integral(*args), add=add, _n=_n+1)
    return e


def pull_outwards(e, add=True, _n=0):
    """
    Trick to maximally pull out constant elements and summation from the
    integrand or the summand.
    """
    if _n > 30:
        warnings.warn("Too high level or recursion, aborting")
        return e

    if add and isinstance(e, Add):
        return Add(*[pull_outwards(arg, add=add, _n=_n+1) for arg in e.args]).expand()
    if isinstance(e, Mul):
        if add:
            return Mul(*[pull_outwards(arg, add=add,
                                       _n=_n+1) for arg in e.args]).expand()
        else:
            return Mul(*[pull_outwards(arg, add=add,
                                       _n=_n+1) for arg in e.args])
    if isinstance(e, Sum):
        return _pull_outwards_sum(e, add=add, _n=_n+1)
    if isinstance(e, Integral):
        return _pull_outwards_integral(e, add=add,  _n=_n+1)
    return e


def push_inwards(e, _n=0):
    """
    Trick to push every factors into integrand or summand
    """
    if _n > 30:
        warnings.warn("Too high level or recursion, aborting")
        return e

    if isinstance(e, Add):
        return Add(*[push_inwards(arg, _n=_n+1) for arg in e.args])

    if isinstance(e, Mul):
        c = Mul(*[arg for arg in e.args if not (isinstance(arg, Integral)
                                                or isinstance(arg, Sum))])
        i_in = [arg for arg in e.args if isinstance(arg, Integral)]
        s_in = [arg for arg in e.args if isinstance(arg, Sum)]

        if not s_in == []:
            func_in = s_in[0].function
            args = [c * func_in * Mul(*s_in[1:]) * Mul(*i_in)]
            for lim_in in s_in[0].limits:
                args.append(lim_in)
            return push_inwards(Sum(*args).expand(), _n=_n+1)

        if not i_in == []:
            func_in = i_in[0].function
            args = [c * func_in * Mul(*s_in) * Mul(*i_in[1:])]
            for lim_in in i_in[0].limits:
                args.append(lim_in)
            return push_inwards(Integral(*args).expand(), _n=_n+1)
        return e

    if isinstance(e, Sum):
        func = e.function
        nfunc = push_inwards(func.expand(), _n=_n+1)

        args = [nfunc]
        for lim in e.limits:
            args.append(lim)
        return Sum(*args)

    if isinstance(e, Integral):
        func = e.function
        nfunc = push_inwards(func.expand(), _n=_n+1)

        args = [nfunc]
        for lim in e.limits:
            args.append(lim)
        return Integral(*args)


def integral_pow_expand(e, _n=0):
    """
    replace powers of an Integral (integer order) with multiple integral
    containing dummy variables(')
    """
    if _n > 20:
        warnings.warn("Too high level or recursion, aborting")
        return e
    if isinstance(e, Add):
        return Add(*(integral_pow_expand(arg, _n=_n+1) for arg in e.args))
    if isinstance(e, Mul):
        return Mul(*(integral_pow_expand(arg, _n=_n+1) for arg in e.args))
    if isinstance(e, Integral):
        func, lims = e.function, e.limits
        return Integral(integral_pow_expand(func, _n=_n+1), lims)
    if isinstance(e, Pow):
        b = e.base
        ex = e.exp
        if isinstance(b, Integral) and isinstance(ex, Integer):
            i = b.function
            dvar = b.limits[0][0]
            if len(b.variables) == 1:
                dvars = [Symbol(str(dvar) + "'"*j, **dvar.assumptions0)
                         for j in range(ex)]
                if len(b.limits[0]) == 1:
                    nlim = [(dvars[j]) for j in range(ex)]
                elif len(b.limits[0]) == 2:
                    nlim = [(dvars[j], b.limits[0][1]) for j in range(ex)]
                else:
                    nlim = [(dvars[j], b.limits[0][1], b.limits[0][2])
                            for j in range(ex)]
                inew = 1
                for j in range(ex):
                    inew = Integral(i.replace(dvars[0], dvars[j]) * inew,
                                    nlim[j])
                return inew
    return e


def sum_pow_expand(e, _n=0):
    """
    replace powers of Sum (integer order) with multiple Sum
    containing dummy variables(')
    """
    if _n > 20:
        warnings.warn("Too high level or recursion, aborting")
        return e
    if isinstance(e, Add):
        return Add(*(sum_pow_expand(arg, _n=_n+1) for arg in e.args))
    if isinstance(e, Mul):
        return Mul(*(sum_pow_expand(arg, _n=_n+1) for arg in e.args))
    if isinstance(e, Integral):
        nargs = [sum_pow_expand(e.function, _n=_n+1)]
        for lim in e.limits:
            nargs.append(lim)
        return Integral(*nargs)
    if isinstance(e, Pow):
        b = e.base
        ex = e.exp
        if isinstance(b, Sum) and isinstance(ex, Integer):
            i = b.function
            dvar = b.limits[0][0]
            if len(b.variables) == 1:
                dvars = [Symbol(str(dvar) + "'"*j, **dvar.assumptions0)
                         for j in range(ex)]
                if len(b.limits[0]) == 1:
                    nlim = [(dvars[j]) for j in range(ex)]
                elif len(b.limits[0]) == 2:
                    nlim = [(dvars[j], b.limits[0][1]) for j in range(ex)]
                else:
                    nlim = [(dvars[j], b.limits[0][1], b.limits[0][2])
                            for j in range(ex)]
                inew = 1
                for j in range(ex):
                    inew = Sum(i.replace(dvars[0], dvars[j]) * inew, nlim[j])
                return inew
    return e


def replace_dirac_delta(e, _n=0):
    """
    Look for Integral of the form ∫ exp(I*k*x) dx
    and replace with 2*pi*DiracDelta(k)
    """
    if _n > 20:
        warnings.warn("Too high level or recursion, aborting")
        return e
    if isinstance(e, Add):
        return Add(*[replace_dirac_delta(arg, _n=_n+1) for arg in e.args])
    if isinstance(e, Mul):
        return Mul(*[replace_dirac_delta(arg, _n=_n+1) for arg in e.args])
    if isinstance(e, Sum):
        nargs = [replace_dirac_delta(e.function, _n=_n+1)]
        for lim in e.limits:
            nargs.append(lim)
        return Sum(*nargs)
    if isinstance(e, Integral):
        func = simplify(e.function)
        lims = e.limits
        if isinstance(func, exp) and len(lims[0]) == 3:  # works only for definite integrals
            ex_s = simplify(func.exp)
            dvar, xa, xb = lims[0]
            if (isinstance(ex_s, Mul)
                    and all([x in ex_s.args for x in [I, dvar]])
                    and (xa, xb) == (-oo, oo)):
                nvar = ex_s/(I*dvar)
                new_func = 2 * pi * DiracDelta(nvar)
                if len(lims) == 1:
                    return new_func
                else:
                    nargs = [new_func]
                    for i in range(1, len(lims)):
                        nargs.append(lims[i])
                    return Integral(*nargs)
        else:
            nargs = [replace_dirac_delta(e.function, _n=_n+1)]
            for lim in e.limits:
                nargs.append(lim)
            return Integral(*nargs)
    return e


def replace_kronecker_delta(e, L, _n=0):
    """
    Look for Integral of the form 
        L
        ∫ sin(n*pi*x/L) * sin(m*pi*x/L) dx
        0
                    or
        L
        ∫ cos(n*pi*x/L) * cos(m*pi*x/L) dx
        0
    and replace with L/2 * KroneckerDelta(n, m)
    if both n and m are positive integers.
    
    In addition, look for Integral of the form
        L
        ∫ sin(n*pi*x/L) * cos(m*pi*x/L) dx
        0
    and replace with 0 if both n and m are
    positive integers.
    """
    if _n > 20:
        warnings.warn("Too high level or recursion, aborting")
        return e
    if isinstance(e, Add):
        return Add(*[replace_kronecker_delta(arg, L=L, _n=_n+1) for arg in e.args])
    if isinstance(e, Mul):
        return Mul(*[replace_kronecker_delta(arg, L=L, _n=_n+1) for arg in e.args])
    if isinstance(e, Sum):
        nargs = [replace_kronecker_delta(e.function, L=L, _n=_n+1)]
        for lim in e.limits:
            nargs.append(lim)
        return Sum(*nargs)
        
    if isinstance(e, Integral):
        func = e.function
        lims = e.limits
        if len(lims)==1 and (isinstance(func, Mul) and len(func.args)==2
            and len(lims[0])==3): # works only for definite integrals
            funcs = func.args
            dvar, xa, xb = lims[0]
            if (xa, xb) == (0, L):
                if ((all([isinstance(f, sin) for f in funcs])
                    or all([isinstance(f, cos) for f in funcs]))
                    and all([dvar in f.args[0].args for f in funcs])):
                    n = [(f.args[0]*L/(dvar*pi)) for f in funcs]
                    if all([m.is_integer and m.is_positive for m in n]):
                        return L * KroneckerDelta(n[0], n[1]) / 2
                
                if (((isinstance(funcs[0], sin) and isinstance(funcs[1], cos))
                    or (isinstance(funcs[0], cos) and isinstance(funcs[1], sin)))
                    and all([dvar in f.args[0].args for f in funcs])):
                    n = [(f.args[0]*L/(dvar*pi)) for f in funcs]
                    if all([m.is_integer and m.is_positive for m in n]):
                        return 0
        else:
            nargs = [replace_kronecker_delta(e.function, L=L, _n=_n+1)]
            for lim in e.limits:
                nargs.append(lim)
            return Integral(*nargs)
    return e

# -----------------------------------------------------------------------------
# Simplification of quantum expressions
#
def expression_tree_transform(e, transformations):
    """
    Traverse and exressions tree  (or list thereof) and conditionally apply a
    transform on the nodes in the tree.
    """
    if isinstance(e, list):
        return [expression_tree_transform(ee, transformations) for ee in e]

    for cond_func, trans_func in transformations:
        if cond_func(e):
            return trans_func(e)

    if isinstance(e, (Add, Mul, Pow, exp)):
        t = type(e)
        return t(*(expression_tree_transform(arg, transformations)
                   for arg in e.args))
    elif isinstance(e, (Sum, Integral)):
        t = type(e)
        f = e.function
        l = e.limits
        nargs = [expression_tree_transform(f, transformations)]
        for lim in l:
            nargs.append(lim)
        return t(*nargs)
    else:
        return e


def qsimplify(e_orig, _n=0):
    """
    Simplify an expression containing operators.
    """
    if _n > 15:
        warnings.warn("Too high level or recursion, aborting")
        return e_orig

    
    e = normal_ordered_form(e_orig)

    if isinstance(e, Add):
        return Add(*(qsimplify(arg, _n=_n+1) for arg in e.args))

    elif isinstance(e, Pow):
        return Pow(*(qsimplify(arg, _n=_n+1) for arg in e.args))

    elif isinstance(e, exp):
        return exp(*(qsimplify(arg, _n=_n+1) for arg in e.args))

    elif isinstance(e, Mul):
        args1 = tuple(arg for arg in e.args if arg.is_commutative)
        args2 = tuple(arg for arg in e.args if not arg.is_commutative)
        #x = 1
        #for y in args2:
        #    x = x * y

        x = 1
        for y in reversed(args2):
            x = y * x

        if isinstance(x, Mul):
            args2 = x.args
            x = 1
            for y in args2:
                x = x * y

        e_new = simplify(Mul(*args1)) * x

        if e_new == e:
            return e
        else:
            return qsimplify(e_new.expand(), _n=_n+1)

    if e == e_orig:
        return e
    else:
        return qsimplify(e, _n=_n+1).expand()


def pauli_represent_minus_plus(e):
    """
    Traverse an expression and change all instances of SigmaX and SigmaY
    to the corresponding expressions using SigmaMinus and SigmaPlus.
    """
    # XXX: todo, make sure that new operators inherit labels
    return expression_tree_transform(
        e, [(lambda e: isinstance(e, SigmaX),
             lambda e: SigmaMinus() + SigmaPlus()),
            (lambda e: isinstance(e, SigmaY),
             lambda e: I * SigmaMinus() - I * SigmaPlus())]
        )


def pauli_represent_x_y(e):
    """
    Traverse an expression and change all instances of SigmaMinus and SigmaPlus
    to the corresponding expressions using SigmaX and SigmaY.
    """
    # XXX: todo, make sure that new operators inherit labels
    return expression_tree_transform(
        e, [(lambda e: isinstance(e, SigmaMinus),
             lambda e: SigmaX() / 2 - I * SigmaY() / 2),
            (lambda e: isinstance(e, SigmaPlus),
             lambda e: SigmaX() / 2 + I * SigmaY() / 2)]
        )


# -----------------------------------------------------------------------------
# Utility functions for manipulating operator expressions
#
def split_coeff_operator(e):
    """
    Split a product of coefficients, commuting variables and quantum
    operators into two factors containing the commuting factors and the
    quantum operators, resepectively.

    Returns:
    c_factor, o_factors:
        Commuting factors and noncommuting (operator) factors
    """
    if isinstance(e, Symbol):
        return e, 1

    if isinstance(e, Add):
        return [split_coeff_operator(arg) for arg in e.args]

    if isinstance(e, Number):
        return e, 1

    if isinstance(e, Operator):
        return 1, e

    if isinstance(e, daggercls):
        if e.is_commutative:
            return e, 1 
        else:
            return 1, e
    
    if isinstance(e, Mul):
        c_args = []
        o_args = []

        for arg in e.args:
            if isinstance(arg, Operator):
                o_args.append(arg)
            elif isinstance(arg, daggercls):
                if arg.is_commutative:
                    c_args.append(arg)
                else:
                    o_args.append(arg)
            elif isinstance(arg, Pow):
                c, o = split_coeff_operator(arg.base)

                if c and c != 1:
                    c_args.append(c ** arg.exp)
                if o and o != 1:
                    o_args.append(o ** arg.exp)
            elif isinstance(arg, Add):
                if arg.is_commutative:
                    c_args.append(arg)
                else:
                    o_args.append(arg)
            else:
                c_args.append(arg)

        return Mul(*c_args), Mul(*o_args)
        
    print("Warning: Unrecognized type of e: %s" % type(e))

    return None, None


def extract_operators(e, independent=False):
    """
    Return a list of unique quantum operator products in the
    expression e.
    """
    ops = []
    if isinstance(e, Operator):
        ops.append(e)

    elif isinstance(e, Add):
        for arg in e.args:
            ops += extract_operators(arg, independent=independent)

    elif isinstance(e, Mul):
        for arg in e.args:
            ops += extract_operators(arg, independent=independent)
    else:
        if debug:
            print("Unrecongized type: %s: %s" % (type(e), str(e)))

    return list(set(ops))


def extract_operator_products(e, independent=False):
    """
    Return a list of unique normal-ordered quantum operator products in the
    expression e.
    """
    ops = []

    if isinstance(e, Operator):
        ops.append(e)

    elif isinstance(e, Add):
        for arg in e.args:
            ops += extract_operator_products(arg, independent=independent)

    elif isinstance(e, Mul):
        c, o = split_coeff_operator(e)
        if o != 1:
            ops.append(o)
    else:
        if debug:
            print("Unrecongized type: %s: %s" % (type(e), str(e)))

    no_ops = []
    for op in ops:
        no_op = normal_ordered_form(op.expand(), independent=independent)
        if isinstance(no_op, (Mul, Operator, Pow)):
            no_ops.append(no_op)
        elif isinstance(no_op, Add):
            for sub_no_op in extract_operator_products(no_op, independent=independent):
                no_ops.append(sub_no_op)
        else:
            raise ValueError("Unsupported type in loop over ops: %s: %s" %
                             (type(no_op), no_op))

    return list(set(no_ops))


def extract_all_operators(e_orig):
    """
    Extract all unique operators in the normal ordered for of a given
    operator expression, including composite operators. The resulting list
    of operators are sorted in increasing order.
    """
    if debug:
        print("extract_all_operators: ", e_orig)

    if isinstance(e_orig, Operator):
        return [e_orig]

    e = drop_c_number_terms(normal_ordered_form(e_orig.expand(),
                                                independent=True))

    if isinstance(e, Pow) and isinstance(e.base, Operator):
        return [e]

    ops = []

    if isinstance(e, Add):
        for arg in e.args:
            ops += extract_all_operators(arg)

    if isinstance(e, Mul):
        op_f = [f for f in e.args if (isinstance(f, Operator) or
                                      (isinstance(f, Pow) and
                                       isinstance(f.base, Operator)))]
        ops.append(Mul(*op_f))
        ops += op_f

    unique_ops = list(set(ops))

    sorted_unique_ops = sorted(unique_ops, key=operator_order)

    return sorted_unique_ops


def operator_order(op):
    if isinstance(op, Operator):
        return 1

    if isinstance(op, Mul):
        return sum([operator_order(arg) for arg in op.args])

    if isinstance(op, Pow):
        return operator_order(op.base) * op.exp

    return 0


def operator_sort_by_order(ops):
    return sorted(ops, key=operator_order)


def drop_terms_containing(e, e_drops):
    """
    Drop terms contaning factors in the list e_drops
    """
    if isinstance(e, Add):
        # fix this
        #e = Add(*(arg for arg in e.args if not any([e_drop in arg.args
        #                                            for e_drop in e_drops])))

        new_args = []

        for term in e.args:

            keep = True
            for e_drop in e_drops:
                if e_drop in term.args:
                    keep = False

                if isinstance(e_drop, Mul):
                    if all([(f in term.args) for f in e_drop.args]):
                        keep = False

            if keep:
        #        new_args.append(arg)
                new_args.append(term)
        e = Add(*new_args)
        #e = Add(*(arg.subs({key: 0 for key in e_drops}) for arg in e.args))

    return e


def drop_c_number_terms(e):
    """
    Drop commuting terms from the expression e
    """
    if isinstance(e, Add):
        return Add(*(arg for arg in e.args if not arg.is_commutative))

    return e


def subs_single(O, subs_map):

    if isinstance(O, Operator):
        if O in subs_map:
            return subs_map[O]
        else:
            print("warning: unresolved operator: ", O)
            return O
    elif isinstance(O, Add):
        new_args = []
        for arg in O.args:
            new_args.append(subs_single(arg, subs_map))
        return Add(*new_args)

    elif isinstance(O, Mul):
        new_args = []
        for arg in O.args:
            new_args.append(subs_single(arg, subs_map))
        return Mul(*new_args)

    elif isinstance(O, Pow):
        return Pow(subs_single(O.base, subs_map), O.exp)

    else:
        return O


class SymEq(Relational):
    """A symbolic equality that is never automatically evaluated.
    """
    rel_op = '=='

    __slots__ = []

    is_Equality = True

    def __new__(cls, lhs, rhs=0, **options):
        lhs = _sympify(lhs)
        rhs = _sympify(rhs)

        return Relational.__new__(cls, lhs, rhs, **options)

    @classmethod
    def _eval_relation(cls, lhs, rhs):
        return None

# -----------------------------------------------------------------------------
# Commutators and BCH expansions
#
def recursive_commutator(a, b, n=1):
    """
    Generate a recursive commutator of order n:

    [a, b]_1 = [a, b]
    [a, b]_2 = [a, [a, b]]
    [a, b]_3 = [a, [a, b]_2] = [a, [a, [a, b]]]
    ...

    """
    if n == 1:
        return Commutator(a, b)
    else:
        return Commutator(a, recursive_commutator(a, b, n-1))


def _bch_expansion(A, B, N=10):
    """
    Baker–Campbell–Hausdorff formula:

    e^{A} B e^{-A} = B + 1/(1!)[A, B] +
                     1/(2!)[A, [A, B]] + 1/(3!)[A, [A, [A, B]]] + ...
                   = B + Sum_n^N 1/(n!)[A, B]^n

    Truncate the sum at N terms.
    """
    e = B
    for n in range(1, N):
        e += recursive_commutator(A, B, n=n) / factorial(n)
    return e


def _order(e):
    fs = list(e.free_symbols)
    if isinstance(e, Pow) and e.base == fs[0]:
        return e.exp
    elif isinstance(e, Mul):
        o = sum([_order(arg) for arg in e.args])
        return o
    elif isinstance(e, Add):
        o = max([_order(arg) for arg in e.args])
        return o
    elif e.is_Symbol:
        return 1
    else:
        return 0
    
def _lowest_order_term(e):
    """Get the term (in a summation) which has the lowest order (of whatever symbol)
    Example:
        - a*a + b -> b, 1
        - a*a*b -> a*a*b, 3
    """
    if isinstance(e, Add):
        min_order = _order(e.args[0])
        min_expr = e.args[0]
        for arg in e.args:
            arg_order  = _order(arg)
            if arg_order < min_order:
                min_order = arg_order
                min_expr = arg
        return min_expr, min_order
    else:
        return e, _order(e)


_s = Dummy()
_fncs = [cos, sin, 
         exp, lambda x: exp(-x),
         lambda x: exp(I*x), lambda x: exp(-I*x),
         sinh, cosh]
_series = [fnc(_s).series(_s, n=20) for fnc in _fncs]
def _expansion_search(expr, rep_list, lib=None):
    """
    Search for and substitute terms that match a series expansion of
    fundamental math functions.

    e: expression

    rep_list: list containing dummy variables

    """
    if debug:
        print("_expansion_search: ", expr)

    try:
        if isinstance(expr, Mul):
            exprs = [expr]
        elif isinstance(expr, Add):
            exprs = expr.args
        else:
            return expr

        if lib is not None:
            fncs, series = lib
        else:
            fncs, series = _fncs, _series 
            
        newargs = []
        for expr in exprs:
            # Try to simplify the expression
            if isinstance(expr, Mul):
                # Split the expression into a LIST of commutative and a LIST of non-commutative factor (i.e. operators).
                c, nc = expr.args_cnc()
                if nc and c:
                    # Construct an expression of commutative elements ONLY.
                    expr = Mul(*c).expand() 
                    #out = find_expansion(expr, *rep_list)
                    
                    eq = Poly(expr, *rep_list)
                    bdict = eq.as_dict()
                    
                    if any([n > 1 for n in [sum([i!=0 for i in k]) for k in bdict.keys()]]):
                        raise ValueError("Expressions cross-products of symbols are not supported (yet?)..")
                    
                    # Construct the list of 'vectors'
                    vecl = []; fncl = []
                    for x in rep_list:
                        for n in range(3): #range(eq.degree()-1):
                            X = pow(x,n)
                            vecl.extend([((X*p.subs(_s, x)).expand()+O(pow(x, eq.degree()+1))).removeO() for p in series])
                            fncl.extend([X*f(x) for f in fncs])
                        vecl.extend([pow(x,n) for n in range(max(eq.degree(), 3))])
                        fncl.extend([pow(x,n) for n in range(max(eq.degree(), 3))])    
                
                    # Create vector coefficients
                    q = symbols(f'q:{len(vecl)}')
                    # Take inner product of 'vector' list and vector coefficients
                    A = Add(*[x*y for x,y in zip(q, vecl)]).as_poly(*rep_list)
                
                    terms = []
                    for k, eq in A.as_dict().items():
                        terms.append(eq - bdict.get(k, 0))
                    H = linsolve(terms, *q)
                    h = H.subs({k: 0 for k in q})
                
                    out = Add(*[c*fnc for fnc, c in zip(fncl, list(h)[0])])
                    
                    newargs.append(out * Mul(*nc))
                else:
                    newargs.append(expr)
            else:
                newargs.append(expr)
                
        return Add(*newargs)

    except Exception as e:
        print("Failed to identify series expansions: " + str(e))
        return e

def get_coefficient(expr, ops, right=True):
    """ Search for the coefficients to either the left or right of ops"""
    if expr == S.Zero:
        return expr
    
    if isinstance(expr, Add):
        out = Add()
        for term in expr.args:
            out += get_coefficient(term, ops, right=right) 
        if out != S.Zero:  
            return out
    elif isinstance(expr, Mul):
        items = expr.args if right else reversed(expr.args)
        for e in items:
            if isinstance(e, Operator) or isinstance(Dagger(e), Operator) or isinstance(e, Add):
                break
        if e == ops:
            terms = list(expr.args)
            terms.remove(e)
            return Mul(*terms)
    elif isinstance(expr, Operator):
        return Add(1)
    elif isinstance(expr, daggercls):
        return Add(1)
    return S.Zero

def ncollect(expr, take='left'):
    """ A version of collect that 'works' with non-commuting sybols"""
    if expr == S.Zero:
        return expr

    terms = split_coeff_operator(expr.expand())
    if isinstance(terms, tuple):
        terms = [terms]

    uniq = {}
    scalar = Add()
    for coeff, e in terms:
        if isinstance(e, Mul):
            if take == 'right':
                d = uniq.setdefault(e.args[-1], Add())
                uniq[e.args[-1]] += Mul(coeff, *e.args[0:-1])
            else:
                d = uniq.setdefault(e.args[0], Add())
                uniq[e.args[0]] += Mul(coeff, *e.args[1:])
        elif isinstance(e, Operator):
            d = uniq.setdefault(e, Add())
            uniq[e] += coeff         
        elif isinstance(e, daggercls):
            if e.is_commutative:
                scalar += coeff
            else:
                d = uniq.setdefault(e, Add())
                uniq[e] += coeff  
        else:
            scalar += coeff
    if take == 'right':
        return Add(*[e*k for k,e in uniq.items()]) + scalar
    else:
        return Add(*[k*e for k,e in uniq.items()]) + scalar

def bch_expansion(A, B, N=6, collect_operators=None, independent=False,
                  expansion_search=True):
    """ Apply the BCH expansion using the exponent A and the operator B """
    
    # Use BCH expansion of order N
    if debug:
        print("bch_expansion: ", A, B)

    # Split the terms (of the summation) into tuples of variables and operators
    cno = split_coeff_operator(A)
    if isinstance(cno, list):
        nvar = len(cno)
        c_list = []
        o_list = []
        for n in range(nvar):
            c_list.append(cno[n][0])
            o_list.append(cno[n][1])
    else:
        nvar = 1
        c_list, o_list = [cno[0]], [cno[1]]

    if debug:
        print("A coefficient: ", c_list)

    rep_list = []
    var_list = []

    # Extract the parts of the exponent expression that contains symbols only...
    # 3*theta*I*sx + b*I*phi*sy => [theta, b*phi]
    # b*exp(-phi*sy) => [b*exp(-phi*SigmaY())]
    for n in range(nvar):
        rep_list.append(Dummy())
        
        # Split multiplication into Coefficient (Number) and an equation of Symbols (only)
        coeff, sym = c_list[n].as_coeff_Mul()
        if isinstance(sym, Mul):
            sym_ = simplify(sym)
            if I in sym_.args: 
                var_list.append(sym_/I)
            elif any([isinstance(arg, exp) for arg in sym_.args]):
                # In case there are exponentials in symbols equation, split it into a part with 
                # and a part without exponentials
                nexps = Mul(*[arg for arg in sym_.args
                              if not isinstance(arg, exp)])
                exps = Mul(*[arg for arg in sym_.args if isinstance(arg, exp)])

                # I don't follow this....
                if I in simplify(exps).exp.args:
                    var_list.append(nexps)
                else:
                    var_list.append(sym_)
        else:
            var_list.append(sym)

    A_rep = A.subs({var_list[n]: rep_list[n] for n in range(nvar)})

    e_bch_rep = _bch_expansion(A_rep, B, N=N).doit(independent=independent)

    if debug:
        print("simplify: ")

    e = qsimplify(normal_ordered_form(e_bch_rep.expand(),
                                      recursive_limit=25,
                                      independent=independent).expand())
    if debug:
        print("extract operators: ")

    if collect_operators:
        ops = collect_operators
    else:
        ops = extract_operator_products(e, independent=independent)
        # make sure that product operators comes first in the list
        ops = list(reversed(sorted(ops, key=lambda x: len(str(x)))))

    if debug:
        print("operators in expression: ", ops)

    # Apply hack to pass aroung collect to handling non-commuting symbols....
    #e_collected = ncollect(e, *ops)
    
    reg = {}
    for op in ops:
        dummy = Dummy()
        e = e.subs(op, dummy)
        reg[dummy] = op
    
    # Use broken collect version. https://github.com/sympy/sympy/issues/16713
    e_collected = acollect(e, reg.keys())    
    e_collected = e_collected.subs(reg)
    
    if debug:
        print("search for series expansions: ", expansion_search)

    if debug:
        print("e_collected: ", e_collected)

    if expansion_search and c_list:
        e_collected = _expansion_search(e_collected, rep_list)
        e_collected = e_collected.subs({rep_list[n]: var_list[n]
                                        for n in range(nvar)})

        return e_collected
    else:
        return e_collected.subs(
            {rep_list[n]: var_list[n] for n in range(nvar)})

# -----------------------------------------------------------------------------
# Transformations
#
def unitary_transformation(U, O, N=6, collect_operators=None,
                           independent=False, allinone=False,
                           expansion_search=True):
    """
    Perform a unitary transformation

        O = U O U^\dagger

    and automatically try to identify series expansions in the resulting
    operator expression.
    """
    if not isinstance(U, exp):
        raise ValueError("U must be a unitary operator on the form "
                         "U = exp(A)")

    A = U.exp

    if debug:
        print("unitary_transformation: using A = ", A)
        
    if allinone:
        return bch_expansion(A, O, N=N, collect_operators=collect_operators,
                             independent=independent,
                             expansion_search=expansion_search)
    else:
        ops = extract_operators(O.expand())
        ops_subs = {op: bch_expansion(A, op, N=N,
                                      collect_operators=collect_operators,
                                      independent=independent,
                                      expansion_search=expansion_search)
                    for op in ops}
        
        expr = subs_single(O, ops_subs)
        
        nexpr = Add()
        for expr in expr.args:
            c, nc = expr.args_cnc()
            if nc and c:
                expr = Mul(*c).expand() 
                expr = expr.simplify() * Mul(*nc) 
            nexpr += expr
        return nexpr


def hamiltonian_transformation(U, H, N=6, collect_operators=None,
                               independent=False, expansion_search=True, 
                               allinone = False):
    """
    Apply an unitary basis transformation to the Hamiltonian H:

        H = U H U^\dagger -i U d/dt(U^\dagger)

    """
    t = [s for s in U.exp.free_symbols if str(s) == 't']
    if t:
        t = t[0]
        H_td = - I * U * diff(exp(-U.exp), t)
    else:
        H_td = 0

    # H_td = I * diff(U, t) * exp(- U.exp)  # hack: Dagger(U) = exp(-U.exp)
    H_st = unitary_transformation(U, H, N=N,
                                  collect_operators=collect_operators,
                                  independent=independent,
                                  expansion_search=expansion_search, 
                                  allinone=allinone)
    return H_st + H_td


# ----------------------------------------------------------------------------
# Master equations and adjoint master equations
#
def lindblad_dissipator(a, rho):
    """
    Lindblad dissipator
    """
    return (a * rho * Dagger(a) - rho * Dagger(a) * a / 2
            - Dagger(a) * a * rho / 2)


def master_equation(rho_t, t, H, a_ops, use_eq=True):
    """
    Lindblad master equation
    """
    #t = [s for s in rho_t.free_symbols if isinstance(s, Symbol)][0]

    rhs = diff(rho_t, t)
    lhs = (-I * Commutator(H, rho_t) +
           sum([lindblad_dissipator(a, rho_t) for a in a_ops]))

    return Eq(rhs, lhs) if use_eq else (rhs, lhs)


def operator_lindblad_dissipator(a, rho):
    """
    Lindblad operator dissipator
    """
    return (Dagger(a) * rho * a - rho * Dagger(a) * a / 2
            - Dagger(a) * a * rho / 2)


def operator_master_equation(op_t, t, H, a_ops, use_eq=True):
    """
    Adjoint master equation
    """
    rhs = diff(op_t, t)
    lhs = (I * Commutator(H, op_t) +
           sum([operator_lindblad_dissipator(a, op_t) for a in a_ops]))

    if use_eq:
        return Eq(rhs, lhs)
    else:
        return rhs, lhs


# -----------------------------------------------------------------------------
# Semiclassical equations of motion
#

def _operator_to_func(e, op_func_map):

    if isinstance(e, Expectation):
        if e.expression in op_func_map:
            return op_func_map[e.expression]
        else:
            return e.expression

    if isinstance(e, Add):
        return Add(*(_operator_to_func(term, op_func_map) for term in e.args))

    if isinstance(e, Mul):
        return Mul(*(_operator_to_func(factor, op_func_map)
                     for factor in e.args))

    return e


def _sceqm_factor_op(op, ops):

    if isinstance(op, Pow):
        for n in range(1, op.exp):
            if Pow(op.base, op.exp - n) in ops and Pow(op.base, n) in ops:
                return op.base, Pow(op.base, op.exp - 1)

        raise Exception("Failed to find factorization of %r" % op)

    if isinstance(op, Mul):
        args = []
        for arg in op.args:
            if isinstance(arg, Pow):
                for n in range(arg.exp):
                    args.append(arg.base)
            else:
                args.append(arg)

        for n in range(1, len(op.args)):
            if Mul(*(args[:n])) in ops and Mul(*(args[n:])) in ops:
                return Mul(*(args[:n])), Mul(*(args[n:]))

        raise Exception("Failed to find factorization of %r" % op)

    return op.args[0], Mul(*(op.args[1:]))


def semi_classical_eqm(H, c_ops, N=20, discard_unresolved=True):
    """
    Generate a set of semiclassical equations of motion from a Hamiltonian
    and set of collapse operators. Equations of motion for all operators that
    are included in either the Hamiltonian or the list of collapse operators
    will be generated, as well as any operators that are included in the
    equations of motion for the orignal operators. If the system of equations
    for the averages of operators does not close, the highest order operators
    will be truncated and removed from the equations.
    """
    op_eqm = {}

    ops = extract_all_operators(H + sum(c_ops))

    if debug:
        print("Hamiltonian operators: ", ops)

    t = symbols("t", positive=True)

    n = 0
    while ops:

        if n > N:
            break

        n += 1

        _, idx = min((val, idx)
                     for (idx, val) in enumerate([operator_order(op)
                                                  for op in ops]))

        op = ops.pop(idx)

        lhs, rhs = operator_master_equation(op, t, H, c_ops, use_eq=False)

        op_eqm[op] = qsimplify(normal_ordered_form(
            rhs.doit(independent=True).expand(), independent=True))

        new_ops = extract_all_operators(op_eqm[op])

        for new_op in new_ops:
            if ((not new_op.is_Number) and
                    new_op not in op_eqm.keys() and new_op not in ops):
                if debug:
                    print(new_op, "not included, adding")
                ops.append(new_op)

    if debug:
        print("unresolved ops: ", ops)


    if discard_unresolved:
        for op, eqm in op_eqm.items():
            op_eqm[op] = drop_terms_containing(op_eqm[op], ops)
        ops_unresolved = []
    else:
        ops_unresolved = ops

    #for op, eqm in op_eqm.items():
    #    for o in extract_all_operators(eqm):
    #        if o not in op_eqm.keys():
    #            print("unresolved operator: %r" % o)

    op_factorization = {}
    sc_eqm = {}
    for op, eqm in op_eqm.items():
        ops = extract_all_operators(eqm)
        sc_eqm[op] = Expectation(eqm).expand(expectation=True)

        for op2 in ops_unresolved:
            if op2 not in op_eqm.keys():
                # need to factor op2
                sub_ops = _sceqm_factor_op(op2, op_eqm.keys())
                factored_expt = Mul(*(Expectation(o) for o in sub_ops))
                op_factorization[Expectation(op2)] = factored_expt
                sc_eqm[op] = sc_eqm[op].subs(Expectation(op2), factored_expt)

    op_func_map = {}
    op_index_map = {}
    for n, op in enumerate(op_eqm):
        op_func_map[op] = Function("A%d" % n)(t)
        op_index_map[op] = n

    if debug:
        print("Operator -> Function map: ", op_func_map)

    sc_ode = {}
    for op, eqm in sc_eqm.items():
        sc_ode[op] = Eq(Derivative(_operator_to_func(Expectation(op), op_func_map), t),
                        _operator_to_func(eqm, op_func_map))

    ops = operator_sort_by_order(op_func_map.keys())

    #for eqm in op_eqm:
    #    eqm_ops = extract_all_operators(op_eqm[op])

    SemiClassicalEQM = namedtuple('SemiClassicalEQM',
                                  ['operators',
                                   'operators_unresolved',
                                   'operator_eqs',
                                   'operator_factorization',
                                   'sc_eqs',
                                   'sc_ode',
                                   'op_func_map',
                                   'op_index_map',
                                   't'
                                   ])

    return SemiClassicalEQM(ops, ops_unresolved,
                            op_eqm, op_factorization, sc_eqm, sc_ode,
                            op_func_map, op_index_map, t)


def semi_classical_eqm_matrix_form(sc_eqm):
    """
    Convert a set of semiclassical equations of motion to matrix form.
    """
    ops = operator_sort_by_order(sc_eqm.op_func_map.keys())
    As = [sc_eqm.op_func_map[op] for op in ops]
    A = Matrix(As)
    b = Matrix([[sc_eqm.sc_ode[op].rhs.subs({A: 0 for A in As})] for op in ops])

    M = Matrix([[((sc_eqm.sc_ode[op1].rhs - b[m]).subs({A: 0 for A in (set(As) - set([sc_eqm.op_func_map[op2]]))}) / sc_eqm.op_func_map[op2]).expand()
                 for m, op1 in enumerate(ops)]
                for n, op2 in enumerate(ops)]).T

    return Equality(-Derivative(A, sc_eqm.t),  b + MatMul(M, A)), A, M, b
