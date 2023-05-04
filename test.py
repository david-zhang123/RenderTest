
from flask import Flask, request, jsonify
from flask_cors import CORS

from sympy import *
from latex2sympy2 import latex2sympy
import re
import mpld3
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)
CORS(app)


def MatrixCorrection(latex_code):
    # replace the string using regular expressions
    output_str = re.sub(
        r"\\left\[\\begin{matrix}", r"\\begin{bmatrix}", latex_code)
    output_str = re.sub(r"\\end{matrix}\\right\]",
                        r"\\end{bmatrix}", output_str)
    output_str = re.sub(r"\\operatorname{re}",
                        r"\\Re", output_str)
    output_str = re.sub(r"\\limits",
                        r"", output_str)

    # return the corrected LaTeX code
    return str(output_str)


def classify_equation(equation):
    # Check if the equation contains partial derivatives re.search(r'\\frac{\\partial\^', equation) or
    if re.search(r'\\partial', equation):
        return 'PDE'
    # Check if the equation contains derivatives
    elif re.search(r'\\frac{d', equation):
        return 'ODE'
    # Otherwise, assume it's an algebraic equation
    else:
        return 'Algebraic'


def Solve(latex_code):
    # Parse the LaTeX string into a sympy expression
    sympy_expr = latex2sympy(latex_code)
    # Check if the expression is an algebraic equation
    sympy_expr_str = str(sympy_expr)
    if "Eq(" in sympy_expr_str:
        return "\\left\\{" + MatrixCorrection(latex(sympy_expr, ln_notation=True))[7:-7] + "\\right\\}"

    else:
        unsimplified_expr = simplify(sympy_expr).doit()
        simplified_expr = simplify(unsimplified_expr)

        if len(str(unsimplified_expr)) < len(str(simplified_expr)):
            return MatrixCorrection(latex(unsimplified_expr, ln_notation=True))
        else:
            return MatrixCorrection(latex(simplified_expr, ln_notation=True))


def Graph(fx):

    latex = fx
    fx = latex2sympy(latex).doit()
    print(fx)
    x = np.arange(-256, 256, .005)

    fig, ax = plt.subplots()

    # Convert the SymPy expression to a NumPy function
    x_sym = sorted(fx.free_symbols)[0]
    func = lambdify(x_sym, fx, 'numpy')

    # Evaluate the function for the input data
    y = func(x)

    # Remove spikes from the data
    pos = np.where(np.abs(np.diff(y)) >= 0.5)[0]
    x[pos] = np.nan
    x[pos-1] = np.round(x[pos-1], 2)
    x[pos+1] = np.round(x[pos+1], 2)

    y[pos] = np.nan
    y[pos-1] = np.round(y[pos-1], 2)
    y[pos+1] = np.round(y[pos+1], 2)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.plot(x, y)
    ax.grid()
    return (mpld3.fig_to_html(fig))


@app.route('/CalcProcessing.py', methods=['POST'])
def run_solve():
    data = request.get_json()
    args = data['args']
    result = str(Solve(args))
    print(Solve(args))
    response_body = {'result': result}
    return jsonify(response_body)


@app.route('/GraphProcessing.py', methods=['POST'])
def run_graph():
    data = request.get_json()
    args = data['args']
    result = Graph(args)
    response_body = {'result': result}
    return jsonify(response_body)


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
