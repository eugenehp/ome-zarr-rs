//! JSON text exactly as Python's `json.dumps` writes it (as zarr-python does
//! for metadata documents): `repr()` floats (`1e-05`, `1e+16`, `NaN`),
//! `ensure_ascii` escaping, and Python's separators and indentation.

use serde_json::Value;

/// `json.dumps(value, indent=indent)`.
pub fn dumps(value: &Value, indent: Option<usize>) -> String {
    let mut out = String::new();
    write_value(&mut out, value, indent, 0);
    out
}

fn newline(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        out.push('\n');
        for _ in 0..n * level {
            out.push(' ');
        }
    }
}

fn write_value(out: &mut String, v: &Value, indent: Option<usize>, level: usize) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                out.push_str(&float_repr(n.as_f64().unwrap_or(f64::NAN)));
            }
        }
        Value::String(s) => write_str(out, s),
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(out, indent, level + 1);
                write_value(out, x, indent, level + 1);
            }
            newline(out, indent, level);
            out.push(']');
        }
        Value::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                newline(out, indent, level + 1);
                write_str(out, k);
                out.push_str(": ");
                write_value(out, x, indent, level + 1);
            }
            newline(out, indent, level);
            out.push('}');
        }
    }
}

fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `repr(float)`: the shortest round-tripping digits, in fixed
/// notation for decimal exponents in `[-4, 16)` and scientific otherwise.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Rust's `{:e}` gives the shortest round-trip digits, e.g. "1.5e-7"
    let e = format!("{x:e}");
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let s = if exp >= 0 {
            if n <= exp + 1 {
                format!("{digits}{}.0", "0".repeat((exp + 1 - n) as usize))
            } else {
                let (a, b) = digits.split_at((exp + 1) as usize);
                format!("{a}.{b}")
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() == 1 {
            digits.clone()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let es = if exp < 0 { '-' } else { '+' };
        format!("{sign}{m}e{es}{:02}", exp.abs())
    }
}

/// Python `repr()` of a JSON value as the equivalent Python object
/// (`{'a': [1, 2.5, None, True]}`), as it appears in exception messages.
pub fn repr(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True" } else { "False" }.into(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f.is_nan() {
                    "nan".into()
                } else if f.is_infinite() {
                    if f > 0.0 { "inf".into() } else { "-inf".into() }
                } else {
                    float_repr(f)
                }
            }
        }
        Value::String(s) => repr_str(s),
        Value::Array(a) => format!("[{}]", a.iter().map(repr).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}: {}", repr_str(k), repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python `repr()` of a string.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::new();
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `str()` of a value: strings as-is, everything else as `repr()`
/// (what an f-string interpolates).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => repr(other),
    }
}

/// `str(Exception(fmt, *args))` for exceptions constructed with several
/// arguments (`ValueError("%s is ...", x)`): the tuple repr of the arguments.
pub fn args_msg(fmt: &str, args: &[String]) -> String {
    let mut parts = vec![repr_str(fmt)];
    parts.extend(args.iter().cloned());
    format!("({})", parts.join(", "))
}

/// Python `repr()` of a set of strings (in the given order).
pub fn set_repr(items: &[String]) -> String {
    if items.is_empty() {
        return "set()".into();
    }
    format!(
        "{{{}}}",
        items
            .iter()
            .map(|s| repr_str(s))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Python `repr()` of a list of strings.
pub fn str_list_repr(items: &[String]) -> String {
    format!(
        "[{}]",
        items
            .iter()
            .map(|s| repr_str(s))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// numpy 2 `repr()` of a scalar of `dtype`, e.g. `np.uint8(7)`,
/// `np.float32(0.1)`, `np.True_`.
pub fn np_scalar_repr(dtype: crate::array::DType, v: f64) -> String {
    use crate::array::DType;
    match dtype {
        DType::Bool => if v != 0.0 { "np.True_" } else { "np.False_" }.into(),
        DType::Float16 => format!("np.float16({})", float16_repr(half::f16::from_f64(v))),
        DType::Float32 => format!("np.float32({})", float32_repr(v as f32)),
        DType::Complex64 | DType::Complex128 => np_complex_repr(dtype, v, 0.0),
        DType::Float64 => format!("np.float64({})", np_float_str(v)),
        d => format!("np.{}({})", d.name(), v as i64),
    }
}

fn np_float_str(v: f64) -> String {
    if v.is_nan() {
        "nan".into()
    } else if v.is_infinite() {
        if v > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        float_repr(v)
    }
}

/// Shortest repr of a float16 in Python's style (as numpy prints it).
pub fn float16_repr(x: half::f16) -> String {
    let v = x.to_f64();
    if !v.is_finite() {
        return np_float_str(v);
    }
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // the fewest significant digits that read back as the same float16
    for p in 1..=17 {
        let e = format!("{:.*e}", p - 1, v);
        if e.parse::<f64>().map(half::f16::from_f64) == Ok(x) {
            let (mant, exp) = e.split_once('e').unwrap();
            let exp: i32 = exp.parse().unwrap();
            let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
            let digits = digits.trim_end_matches('0');
            let digits = if digits.is_empty() { "0" } else { digits };
            let sign = if mant.starts_with('-') { "-" } else { "" };
            return format_digits(sign, digits, exp);
        }
    }
    float_repr(v)
}

/// numpy 2 `repr()` of a complex scalar: `np.complex64(2.5+1j)`,
/// `np.complex128(10j)` (the parts as Python prints a complex number).
pub fn np_complex_repr(dtype: crate::array::DType, re: f64, im: f64) -> String {
    let part = |v: f64| -> String {
        let s = if !v.is_finite() {
            np_float_str(v)
        } else if dtype == crate::array::DType::Complex64 {
            float32_repr(v as f32)
        } else {
            float_repr(v)
        };
        s.strip_suffix(".0").map(String::from).unwrap_or(s)
    };
    let body = if re == 0.0 && !re.is_sign_negative() {
        format!("{}j", part(im))
    } else {
        let im_s = part(im);
        let sign = if im_s.starts_with('-') { "" } else { "+" };
        format!("{}{sign}{im_s}j", part(re))
    };
    format!("np.{}({body})", dtype.name())
}

/// Shortest repr of a float32 in Python's style (as numpy prints it).
pub fn float32_repr(x: f32) -> String {
    if !x.is_finite() {
        return np_float_str(x as f64);
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // `{:e}` on f32 gives the shortest digits that round-trip as f32
    let e = format!("{x:e}");
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let sign = if mant.starts_with('-') { "-" } else { "" };
    format_digits(sign, &digits, exp)
}

fn format_digits(sign: &str, digits: &str, exp: i32) -> String {
    if (-4..16).contains(&exp) {
        let n = digits.len() as i32;
        let s = if exp >= 0 {
            if n <= exp + 1 {
                format!("{digits}{}.0", "0".repeat((exp + 1 - n) as usize))
            } else {
                let (a, b) = digits.split_at((exp + 1) as usize);
                format!("{a}.{b}")
            }
        } else {
            format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() == 1 {
            digits.to_string()
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let es = if exp < 0 { '-' } else { '+' };
        format!("{sign}{m}e{es}{:02}", exp.abs())
    }
}

/// Python / numpy float floor division `a // b` (CPython `float_floor_div`,
/// numpy `npy_floor_divide`), which is not always `floor(a / b)`.
pub fn py_floordiv(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        return f64::NAN;
    }
    let m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let mut fd = div.floor();
        if div - fd > 0.5 {
            fd += 1.0;
        }
        fd
    } else {
        0.0f64.copysign(a / b)
    }
}

/// Python `repr()` of a shape tuple, e.g. `(3, 4)` or `(5,)`.
pub fn shape_repr(shape: &[usize]) -> String {
    match shape {
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python `str()` / `repr()` of a value used as a graph node or similar.
pub trait PyDisplay {
    fn py_str(&self) -> String {
        self.py_repr()
    }
    fn py_repr(&self) -> String;
}

impl PyDisplay for String {
    fn py_str(&self) -> String {
        self.clone()
    }
    fn py_repr(&self) -> String {
        repr_str(self)
    }
}

impl PyDisplay for &str {
    fn py_str(&self) -> String {
        self.to_string()
    }
    fn py_repr(&self) -> String {
        repr_str(self)
    }
}

macro_rules! py_display_int {
    ($($t:ty),*) => {$(
        impl PyDisplay for $t {
            fn py_repr(&self) -> String {
                self.to_string()
            }
        }
    )*};
}
py_display_int!(i32, i64, u32, u64, usize);

impl<A: PyDisplay, B: PyDisplay> PyDisplay for (A, B) {
    fn py_repr(&self) -> String {
        format!("({}, {})", self.0.py_repr(), self.1.py_repr())
    }
}

/// numpy's element strings for a float array (common precision and padding).
fn np_format_items(v: &[f64]) -> Vec<String> {
    let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    let abs_nz: Vec<f64> = finite
        .iter()
        .map(|x| x.abs())
        .filter(|&x| x != 0.0)
        .collect();
    let max = abs_nz.iter().cloned().fold(0.0, f64::max);
    let min = abs_nz.iter().cloned().fold(f64::INFINITY, f64::min);
    let exp = !abs_nz.is_empty() && (max >= 1e8 || min < 1e-4 || max / min > 1e3);
    let nonfinite = |x: f64| {
        if x.is_nan() {
            "nan".to_string()
        } else if x > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        }
    };
    let items: Vec<String> = if exp {
        // shortest mantissa, at most 8 fraction digits, zero-padded to the longest
        let parts: Vec<(String, String, i32)> = finite
            .iter()
            .map(|&x| {
                let mut s = format!("{x:e}");
                let (m, _) = s.split_once('e').unwrap();
                if m.split_once('.').is_some_and(|(_, f)| f.len() > 8) {
                    s = format!("{x:.8e}");
                }
                let (m, e) = s.split_once('e').unwrap();
                let m = if m.contains('.') {
                    m.trim_end_matches('0').to_string()
                } else {
                    m.to_string()
                };
                let (ip, fp) = m
                    .split_once('.')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .unwrap_or((m.clone(), String::new()));
                (ip, fp, e.parse().unwrap())
            })
            .collect();
        let pad_left = parts.iter().map(|p| p.0.len()).max().unwrap_or(0);
        let prec = parts.iter().map(|p| p.1.len()).max().unwrap_or(0);
        let exp_size = parts
            .iter()
            .map(|p| p.2.unsigned_abs().to_string().len())
            .max()
            .unwrap_or(0)
            .max(2);
        let mut fi = parts.into_iter();
        v.iter()
            .map(|&x| {
                if !x.is_finite() {
                    return format!("{:>w$}", nonfinite(x), w = pad_left + prec + 3 + exp_size);
                }
                let (ip, fp, e) = fi.next().unwrap();
                format!(
                    "{ip:>pad_left$}.{fp:0<prec$}e{}{:0>exp_size$}",
                    if e < 0 { '-' } else { '+' },
                    e.unsigned_abs()
                )
            })
            .collect()
    } else {
        let parts: Vec<(String, String)> = finite
            .iter()
            .map(|&x| {
                let mut s = float_repr(x);
                if s.contains('e') || s.split_once('.').is_some_and(|(_, f)| f.len() > 8) {
                    s = format!("{x:.8}");
                }
                let (ip, fp) = s
                    .split_once('.')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
                    .unwrap_or((s.clone(), String::new()));
                let fp = fp.trim_end_matches('0').to_string();
                (ip, fp)
            })
            .collect();
        let pad_left = parts.iter().map(|p| p.0.len()).max().unwrap_or(0);
        let pad_right = parts.iter().map(|p| p.1.len()).max().unwrap_or(0);
        let mut fi = parts.into_iter();
        v.iter()
            .map(|&x| {
                if !x.is_finite() {
                    return format!("{:>w$}", nonfinite(x), w = pad_left + pad_right + 1);
                }
                let (ip, fp) = fi.next().unwrap();
                format!("{ip:>pad_left$}.{fp:<pad_right$}")
            })
            .collect()
    };
    items
}

/// numpy `str()` of a 1-D float array (`np.array2string` defaults:
/// `floatmode='maxprec'`, `precision=8`), e.g. `[0.5 0.  1. ]`.
pub fn np_array_str(v: &[f64]) -> String {
    format!("[{}]", np_format_items(v).join(" "))
}

/// numpy `str()` of a 2-D float array, e.g. `[[1. 1.]\n [0. 1.]]`.
pub fn np_array2_str(rows: &[Vec<f64>]) -> String {
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let items = np_format_items(&flat);
    let mut k = 0;
    let lines: Vec<String> = rows
        .iter()
        .map(|r| {
            let row = items[k..k + r.len()].join(" ");
            k += r.len();
            format!("[{row}]")
        })
        .collect();
    format!("[{}]", lines.join("\n "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn floats() {
        for (x, s) in [
            (1.0, "1.0"),
            (0.5, "0.5"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (123456789012345.6, "123456789012345.6"),
            (2.00990099009901, "2.00990099009901"),
            (-0.25, "-0.25"),
            (1.5e300, "1.5e+300"),
            (0.1 + 0.2, "0.30000000000000004"),
            (100.0, "100.0"),
            (f64::NAN, "NaN"),
        ] {
            assert_eq!(float_repr(x), s);
        }
    }

    #[test]
    fn indent_and_escape() {
        let v = json!({"a": [1, 2.5, {}], "b": [], "é": "x\"\n"});
        assert_eq!(
            dumps(&v, Some(2)),
            "{\n  \"a\": [\n    1,\n    2.5,\n    {}\n  ],\n  \"b\": [],\n  \"\\u00e9\": \"x\\\"\\n\"\n}"
        );
        assert_eq!(
            dumps(&v, None),
            "{\"a\": [1, 2.5, {}], \"b\": [], \"\\u00e9\": \"x\\\"\\n\"}"
        );
    }
}

#[cfg(test)]
mod repr_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reprs() {
        assert_eq!(
            repr(&json!({"path": "A/1", "n": [1, 2.5, null, true]})),
            "{'path': 'A/1', 'n': [1, 2.5, None, True]}"
        );
        assert_eq!(repr_str("it's"), "\"it's\"");
    }
}

#[cfg(test)]
mod np_tests {
    use super::*;

    #[test]
    fn array_str() {
        assert_eq!(np_array_str(&[0.0, 0.0, 1.0]), "[0. 0. 1.]");
        assert_eq!(np_array_str(&[0.5, 0.0, 1.0]), "[0.5 0.  1. ]");
        assert_eq!(np_array_str(&[-12.25, 1e-3]), "[-1.225e+01  1.000e-03]");
        assert_eq!(np_array_str(&[3.0, 100.0, 2.0]), "[  3. 100.   2.]");
        assert_eq!(
            np_array_str(&[1.0 / 3.0, -2.0]),
            "[ 0.33333333 -2.        ]"
        );
        // numpy 2.x outputs
        assert_eq!(
            np_array_str(&[0.3333333333333333, -2.0]),
            "[ 0.33333333 -2.        ]"
        );
        assert_eq!(np_array_str(&[1e-05, 1.0]), "[1.e-05 1.e+00]");
        assert_eq!(np_array_str(&[100000000.0, 1.0]), "[1.e+08 1.e+00]");
        assert_eq!(
            np_array_str(&[0.1, 0.25, 1000.0]),
            "[1.0e-01 2.5e-01 1.0e+03]"
        );
        assert_eq!(np_array_str(&[1.5, f64::NAN]), "[1.5 nan]");
        assert_eq!(
            np_array_str(&[123456.789, 0.5]),
            "[1.23456789e+05 5.00000000e-01]"
        );
        assert_eq!(np_array_str(&[-0.0, 1.0]), "[-0.  1.]");
        assert_eq!(np_array_str(&[0.0025, 7.0]), "[2.5e-03 7.0e+00]");
        assert_eq!(np_array_str(&[1e+300, -1.0]), "[ 1.e+300 -1.e+000]");
    }
}
