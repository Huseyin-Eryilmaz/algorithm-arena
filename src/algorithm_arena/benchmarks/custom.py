from __future__ import annotations

import re

import numpy as np
import sympy
from sympy import Symbol, symbols
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from algorithm_arena.optimizers.base import ObjectiveFn

_ALLOWED_FUNCTIONS = {
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "exp": sympy.exp,
    "log": sympy.log,
    "sqrt": sympy.sqrt,
    "Abs": sympy.Abs,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "Symbol": Symbol,  # auto_symbol dönüşümünün ürettiği kod bunu çağırıyor
}

# auto_number dönüşümünü standard_transformations'dan çıkarıyoruz: bu
# dönüşüm sayı literallerini (örn. "2") Integer(2)/Float(2) çağrılarına
# çeviriyor, bu da global_dict={"__builtins__": {}} ile çakışıp
# "name 'Integer' is not defined" hatasına yol açıyordu. Dönüşümü
# kaldırınca sayı literalleri düz Python int/float olarak kalıyor,
# ki bu zaten x**2 gibi ifadeler için yeterli ve hatta daha az whitelist
# bakımı gerektiriyor.
_TRANSFORMATIONS = tuple(
    t for t in standard_transformations if t.__name__ != "auto_number"
) + (convert_xor,)

_ALLOWED_CHARS_PATTERN = re.compile(r"^[0-9a-zA-Z\s+\-*/.(),]+$")


class InvalidExpressionError(Exception):
    """Kullanıcının girdiği matematiksel ifade parse edilemediğinde veya
    izin verilmeyen bir yapı içerdiğinde fırlatılır."""


def parse_expression(expression_str: str) -> tuple[ObjectiveFn, sympy.Expr]:
    """
    Kullanıcının yazdığı string ifadeyi güvenli şekilde parse eder.

    Güvenlik katmanları:
      1. Karakter whitelist'i (regex) — dunder/özel karakterleri baştan eler.
      2. local_dict'i sadece x, y ve izin verilen matematik fonksiyonlarıyla
         sınırlamak — tanımsız her isim NameError fırlatır.
      3. global_dict'te __builtins__'i açıkça boşaltmak — sympy'nin eval
         fallback'inin gerçek Python builtins'ine erişmesini engeller.
      4. auto_number dönüşümünü kapatmak — sayı literalleri için ekstra
         whitelist bakımı gerektirmez, saldırı yüzeyini küçültür.
    """
    if not _ALLOWED_CHARS_PATTERN.match(expression_str):
        raise InvalidExpressionError(
            f"İfade izin verilmeyen karakterler içeriyor: '{expression_str}'. "
            f"Sadece sayılar, x, y, +, -, *, /, ., (), , ve matematik fonksiyon "
            f"isimleri (sin, cos, exp, sqrt, log, ...) kullanılabilir."
        )

    x, y = symbols("x y", real=True)
    local_dict = {"x": x, "y": y, **_ALLOWED_FUNCTIONS}

    try:
        expr = parse_expr(
            expression_str,
            local_dict=local_dict,
            global_dict={"__builtins__": {}},
            transformations=_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as e:
        raise InvalidExpressionError(
            f"İfade parse edilemedi: '{expression_str}'. Hata: {e}"
        ) from e

    # auto_number kapalı olduğu için "5" gibi salt sayısal ifadeler artık
    # sympy.Expr değil, çıplak Python int/float olarak dönebilir.
    # Aşağı akıştaki free_symbols ve lambdify çağrıları sympy.Expr bekliyor,
    # o yüzden burada bilinçli şekilde sympy tipine sarıyoruz.
    if isinstance(expr, (int, float)):
        expr = sympy.Integer(expr) if isinstance(expr, int) else sympy.Float(expr)

    if not isinstance(expr, sympy.Expr):
        raise InvalidExpressionError(
            f"'{expression_str}' geçerli bir matematiksel ifade değil."
        )
    free_symbols = expr.free_symbols
    allowed_symbols = {x, y}
    unexpected = free_symbols - allowed_symbols
    if unexpected:
        raise InvalidExpressionError(
            f"Sadece 'x' ve 'y' değişkenleri desteklenir. "
            f"Beklenmeyen sembol(ler): {unexpected}"
        )

    numeric_fn = sympy.lambdify((x, y), expr, modules="numpy")

    def objective_fn(positions: np.ndarray) -> np.ndarray:
        x_vals = positions[:, 0]
        y_vals = positions[:, 1]
        result = numeric_fn(x_vals, y_vals)
        if np.isscalar(result):
            result = np.full(positions.shape[0], result, dtype=float)
        return np.asarray(result, dtype=float)

    return objective_fn, expr
