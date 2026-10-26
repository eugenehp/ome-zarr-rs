//! pydantic validation of the ome-zarr-models classes, with pydantic's
//! errors: an interpreter of the pydantic-core schemas of the models
//! (`pydantic_schemas.json`, exported by scripts/gen_pydantic_fixtures.py)
//! in Python ("lax") mode, with the models' validators, producing
//! `ValidationError` messages exactly as pydantic formats them.

use std::sync::OnceLock;

use serde_json::Value;

use crate::error::Error;
use crate::pyjson::{float_repr, repr_str};

static SCHEMAS_JSON: &str = include_str!("pydantic_schemas.json");

fn schemas() -> &'static serde_json::Map<String, Value> {
    static S: OnceLock<serde_json::Map<String, Value>> = OnceLock::new();
    S.get_or_init(|| serde_json::from_str(SCHEMAS_JSON).expect("pydantic_schemas.json"))
}

// ---------------------------------------------------------------- Python values

/// A Python value: a JSON input, or a validated (coerced) value.
#[derive(Debug, Clone, PartialEq)]
pub enum Py {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<Py>),
    Tuple(Vec<Py>),
    Dict(Vec<(String, Py)>),
    /// A model instance: its fields (in definition order) and extra fields.
    Model {
        cls: String,
        fields: Vec<(String, Py)>,
        extra: Vec<(String, Py)>,
    },
}

impl Py {
    pub fn from_json(v: &Value) -> Py {
        match v {
            Value::Null => Py::None,
            Value::Bool(b) => Py::Bool(*b),
            Value::Number(n) => match n.as_i64() {
                Some(i) => Py::Int(i),
                None => Py::Float(n.as_f64().unwrap_or(f64::NAN)),
            },
            Value::String(s) => Py::Str(s.clone()),
            Value::Array(a) => Py::List(a.iter().map(Py::from_json).collect()),
            Value::Object(m) => Py::Dict(
                m.iter()
                    .map(|(k, v)| (k.clone(), Py::from_json(v)))
                    .collect(),
            ),
        }
    }

    /// `type(x).__name__`.
    fn type_name(&self) -> String {
        match self {
            Py::None => "NoneType".into(),
            Py::Bool(_) => "bool".into(),
            Py::Int(_) => "int".into(),
            Py::Float(_) => "float".into(),
            Py::Str(_) => "str".into(),
            Py::List(_) => "list".into(),
            Py::Tuple(_) => "tuple".into(),
            Py::Dict(_) => "dict".into(),
            Py::Model { cls, .. } => cls.clone(),
        }
    }

    /// `repr(x)`.
    pub fn repr(&self) -> String {
        match self {
            Py::None => "None".into(),
            Py::Bool(b) => if *b { "True" } else { "False" }.into(),
            Py::Int(i) => i.to_string(),
            Py::Float(f) => py_float(*f),
            Py::Str(s) => repr_str(s),
            Py::List(a) => format!(
                "[{}]",
                a.iter().map(Py::repr).collect::<Vec<_>>().join(", ")
            ),
            Py::Tuple(a) => tuple_repr(a),
            Py::Dict(m) => {
                format!(
                    "{{{}}}",
                    m.iter()
                        .map(|(k, v)| format!("{}: {}", repr_str(k), v.repr()))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Py::Model { cls, .. } => format!("{cls}({})", self.model_args().join(", ")),
        }
    }

    /// `str(x)`: strings unquoted, models as `a=1 b='x'`.
    pub fn str(&self) -> String {
        match self {
            Py::Str(s) => s.clone(),
            Py::Model { .. } => self.model_args().join(" "),
            other => other.repr(),
        }
    }

    fn model_args(&self) -> Vec<String> {
        match self {
            Py::Model { fields, extra, .. } => fields
                .iter()
                .chain(extra)
                .map(|(k, v)| format!("{k}={}", v.repr()))
                .collect(),
            _ => vec![],
        }
    }

    fn get(&self, k: &str) -> &Py {
        static NONE: Py = Py::None;
        match self {
            Py::Model { fields, extra, .. } => fields
                .iter()
                .chain(extra)
                .find(|(n, _)| n == k)
                .map(|(_, v)| v)
                .unwrap_or(&NONE),
            Py::Dict(m) => m
                .iter()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v)
                .unwrap_or(&NONE),
            _ => &NONE,
        }
    }

    fn cls(&self) -> &str {
        match self {
            Py::Model { cls, .. } => cls,
            _ => "",
        }
    }

    fn items(&self) -> &[Py] {
        match self {
            Py::List(a) | Py::Tuple(a) => a,
            _ => &[],
        }
    }

    fn is_none(&self) -> bool {
        matches!(self, Py::None)
    }

    fn as_f64(&self) -> Option<f64> {
        match self {
            Py::Int(i) => Some(*i as f64),
            Py::Float(f) => Some(*f),
            Py::Bool(b) => Some(*b as i64 as f64),
            _ => None,
        }
    }

    /// Python `==` (numbers compare by value across int / float / bool).
    fn py_eq(&self, other: &Py) -> bool {
        match (self.as_f64(), other.as_f64()) {
            (Some(a), Some(b)) => a == b,
            _ => self == other,
        }
    }
}

fn py_float(f: f64) -> String {
    if f.is_nan() {
        "nan".into()
    } else if f.is_infinite() {
        if f > 0.0 { "inf".into() } else { "-inf".into() }
    } else {
        float_repr(f)
    }
}

fn tuple_repr(a: &[Py]) -> String {
    match a {
        [x] => format!("({},)", x.repr()),
        _ => format!(
            "({})",
            a.iter().map(Py::repr).collect::<Vec<_>>().join(", ")
        ),
    }
}

// ---------------------------------------------------------------- errors

#[derive(Debug, Clone)]
enum Loc {
    Key(String),
    Idx(usize),
}

#[derive(Debug, Clone)]
struct LineError {
    kind: String,
    msg: String,
    loc: Vec<Loc>,
    input: Py,
}

/// A failed validation: pydantic errors, or an exception pydantic does not
/// catch (it propagates as such).
enum Fail {
    Errors(Vec<LineError>),
    Raise(Error),
}

type V = Result<Py, Fail>;

fn err(kind: &str, msg: impl Into<String>, input: &Py) -> Fail {
    Fail::Errors(vec![LineError {
        kind: kind.into(),
        msg: msg.into(),
        loc: vec![],
        input: input.clone(),
    }])
}

fn prefix(f: Fail, item: Loc) -> Fail {
    match f {
        Fail::Errors(mut es) => {
            for e in &mut es {
                e.loc.insert(0, item.clone());
            }
            Fail::Errors(es)
        }
        r => r,
    }
}

/// pydantic-core's `input_value` repr: longer than 50 bytes, it keeps the
/// first 25 and the last 24 bytes (at character boundaries).
fn truncate(s: &str) -> String {
    if s.len() <= 50 {
        return s.to_string();
    }
    let mut a = 25;
    while !s.is_char_boundary(a) {
        a -= 1;
    }
    let mut b = s.len() - 24;
    while !s.is_char_boundary(b) {
        b -= 1;
    }
    format!("{}...{}", &s[..a], &s[b..])
}

fn format_errors(title: &str, es: &[LineError]) -> String {
    let n = es.len();
    let mut out = format!(
        "{n} validation error{} for {title}",
        if n == 1 { "" } else { "s" }
    );
    for e in es {
        out.push('\n');
        if !e.loc.is_empty() {
            let loc: Vec<String> = e
                .loc
                .iter()
                .map(|l| match l {
                    Loc::Key(k) => k.clone(),
                    Loc::Idx(i) => i.to_string(),
                })
                .collect();
            out.push_str(&loc.join("."));
            out.push('\n');
        }
        out.push_str(&format!(
            "  {} [type={}, input_value={}, input_type={}]",
            e.msg,
            e.kind,
            truncate(&e.input.repr()),
            e.input.type_name()
        ));
        // custom errors (`PydanticCustomError`) have no documentation link
        if !matches!(e.kind.as_str(), "sequence_str" | "invalid-json-value") {
            out.push_str(&format!(
                "\n    For further information visit https://errors.pydantic.dev/2.12/v/{}",
                e.kind
            ));
        }
    }
    out
}

// ---------------------------------------------------------------- the interpreter

struct Ctx<'a> {
    defs: &'a serde_json::Map<String, Value>,
}

fn s<'a>(schema: &'a Value, k: &str) -> &'a Value {
    &schema[k]
}

/// pydantic-core's name of a schema (used in union error locations).
fn schema_name(schema: &Value, ctx: &Ctx) -> String {
    match schema["type"].as_str().unwrap_or("") {
        "literal" => format!(
            "literal[{}]",
            schema["expected"]
                .as_array()
                .map(|a| a
                    .iter()
                    .map(|v| Py::from_json(v).repr())
                    .collect::<Vec<_>>()
                    .join(","))
                .unwrap_or_default()
        ),
        "model" => schema["cls"].as_str().unwrap_or("").to_string(),
        "nullable" => format!("nullable[{}]", schema_name(&schema["schema"], ctx)),
        "default" => schema_name(&schema["schema"], ctx),
        "union" => format!(
            "union[{}]",
            schema["choices"]
                .as_array()
                .map(|c| c
                    .iter()
                    .map(|x| schema_name(x, ctx))
                    .collect::<Vec<_>>()
                    .join(","))
                .unwrap_or_default()
        ),
        "list" => format!("list[{}]", schema_name(&schema["items_schema"], ctx)),
        "tuple" => {
            let items: Vec<String> = schema["items_schema"]
                .as_array()
                .map(|c| c.iter().map(|x| schema_name(x, ctx)).collect())
                .unwrap_or_default();
            if schema["variadic_item_index"].is_u64() {
                format!("tuple[{}, ...]", items.join(","))
            } else {
                format!("tuple[{}]", items.join(", "))
            }
        }
        "definition-ref" => ctx
            .defs
            .get(schema["schema_ref"].as_str().unwrap_or(""))
            .map(|d| schema_name(d, ctx))
            .unwrap_or_default(),
        t => t.to_string(),
    }
}

fn validate(schema: &Value, input: &Py, strict: bool, ctx: &Ctx) -> V {
    let strict = strict || schema["strict"].as_bool() == Some(true);
    match schema["type"].as_str().unwrap_or("") {
        "any" => Ok(input.clone()),
        "none" => match input {
            Py::None => Ok(Py::None),
            _ => Err(err("none_required", "Input should be None", input)),
        },
        "nullable" => match input {
            Py::None => Ok(Py::None),
            _ => validate(s(schema, "schema"), input, strict, ctx),
        },
        "default" => validate(s(schema, "schema"), input, strict, ctx),
        "definition-ref" => {
            let r = schema["schema_ref"].as_str().unwrap_or("");
            let d = ctx
                .defs
                .get(r)
                .unwrap_or_else(|| panic!("pydantic schema: unknown ref {r}"));
            validate(d, input, strict, ctx)
        }
        "json-or-python" => validate(s(schema, "python_schema"), input, strict, ctx),
        "str" => validate_str(schema, input),
        "int" => validate_int(schema, input, strict),
        "float" => validate_float(input, strict),
        "bool" => validate_bool(input, strict),
        "literal" => {
            let expected: Vec<Py> = schema["expected"]
                .as_array()
                .map(|a| a.iter().map(Py::from_json).collect())
                .unwrap_or_default();
            if let Some(e) = expected.iter().find(|e| e.py_eq(input)) {
                return Ok(e.clone());
            }
            let reprs: Vec<String> = expected.iter().map(Py::repr).collect();
            let msg = match reprs.len() {
                0 => String::new(),
                1 => reprs[0].clone(),
                n => format!("{} or {}", reprs[..n - 1].join(", "), reprs[n - 1]),
            };
            Err(err(
                "literal_error",
                format!("Input should be {msg}"),
                input,
            ))
        }
        "union" => {
            let choices = schema["choices"].as_array().cloned().unwrap_or_default();
            for c in &choices {
                if let Ok(v) = validate(c, input, true, ctx) {
                    return Ok(v);
                }
            }
            let mut errors = vec![];
            for c in &choices {
                match validate(c, input, strict, ctx) {
                    Ok(v) => return Ok(v),
                    Err(Fail::Raise(e)) => return Err(Fail::Raise(e)),
                    Err(f) => {
                        if let Fail::Errors(es) = prefix(f, Loc::Key(schema_name(c, ctx))) {
                            errors.extend(es);
                        }
                    }
                }
            }
            Err(Fail::Errors(errors))
        }
        "tagged-union" => validate_tagged_union(schema, input, strict, ctx),
        "list" | "tuple" => validate_sequence(schema, input, strict, ctx),
        "dict" => match input {
            Py::Dict(m) => {
                let mut out = vec![];
                let mut errors = vec![];
                for (k, v) in m {
                    match validate(s(schema, "values_schema"), v, strict, ctx) {
                        Ok(v) => out.push((k.clone(), v)),
                        Err(Fail::Raise(e)) => return Err(Fail::Raise(e)),
                        Err(f) => {
                            if let Fail::Errors(es) = prefix(f, Loc::Key(k.clone())) {
                                errors.extend(es);
                            }
                        }
                    }
                }
                if errors.is_empty() {
                    Ok(Py::Dict(out))
                } else {
                    Err(Fail::Errors(errors))
                }
            }
            _ => Err(err(
                "dict_type",
                "Input should be a valid dictionary",
                input,
            )),
        },
        "model" => {
            let cls = schema["cls"].as_str().unwrap_or("");
            let Py::Dict(m) = input else {
                return Err(err(
                    "model_type",
                    format!("Input should be a valid dictionary or instance of {cls}"),
                    input,
                ));
            };
            validate_fields(cls, s(schema, "schema"), m, input, strict, ctx)
        }
        "function-after" => {
            let v = validate(s(schema, "schema"), input, strict, ctx)?;
            let name = schema["function"].as_str().unwrap_or("");
            match call_validator(name, v) {
                Ok(v) => Ok(v),
                Err(Raised::Value(msg)) => {
                    Err(err("value_error", format!("Value error, {msg}"), input))
                }
                Err(Raised::Other(e)) => Err(Fail::Raise(e)),
                Err(Raised::Nested(es)) => Err(Fail::Errors(es)),
            }
        }
        "chain" => {
            let mut v = input.clone();
            for step in schema["steps"].as_array().cloned().unwrap_or_default() {
                v = validate(&step, &v, strict, ctx)?;
            }
            Ok(v)
        }
        "is-instance" => {
            // the only class checked is `collections.abc.Sequence`
            match input {
                Py::List(_) | Py::Tuple(_) | Py::Str(_) => Ok(input.clone()),
                _ => {
                    let cls = schema["cls_repr"].as_str().unwrap_or("");
                    Err(err(
                        "is_instance_of",
                        format!("Input should be an instance of {cls}"),
                        input,
                    ))
                }
            }
        }
        "function-wrap" => {
            // pydantic's `sequence_validator`: no strings, then the list schema
            if matches!(input, Py::Str(_)) {
                return Err(err(
                    "sequence_str",
                    "'str' instances are not allowed as a Sequence value",
                    input,
                ));
            }
            validate(s(schema, "schema"), input, strict, ctx)
        }
        "function-before" => {
            let name = schema["function"].as_str().unwrap_or("");
            let v = match if name.ends_with("Dataset._ensure_scale_translation") {
                ensure_scale_translation(input)
            } else {
                call_validator(name, input.clone())
            } {
                Ok(v) => v,
                Err(Raised::Value(msg)) => {
                    return Err(err("value_error", format!("Value error, {msg}"), input));
                }
                Err(Raised::Other(e)) => return Err(Fail::Raise(e)),
                Err(Raised::Nested(es)) => return Err(Fail::Errors(es)),
            };
            validate(s(schema, "schema"), &v, strict, ctx)
        }
        t => Err(Fail::Raise(Error::NotImplemented(format!(
            "pydantic schema type {t}"
        )))),
    }
}

fn validate_str(schema: &Value, input: &Py) -> V {
    let Py::Str(v) = input else {
        return Err(err("string_type", "Input should be a valid string", input));
    };
    if let Some(n) = schema["min_length"].as_u64()
        && (v.chars().count() as u64) < n
    {
        return Err(err(
            "string_too_short",
            format!(
                "String should have at least {n} character{}",
                if n == 1 { "" } else { "s" }
            ),
            input,
        ));
    }
    if let Some(p) = schema["pattern"].as_str()
        && !pattern_search(p, v)
    {
        return Err(err(
            "string_pattern_mismatch",
            format!("String should match pattern '{p}'"),
            input,
        ));
    }
    Ok(input.clone())
}

/// `re.search(pattern, s)` for the patterns the models use.
fn pattern_search(pattern: &str, s: &str) -> bool {
    match pattern {
        "[0-9a-fA-F]{6}" => {
            let c: Vec<char> = s.chars().collect();
            c.windows(6)
                .any(|w| w.iter().all(|x| x.is_ascii_hexdigit()))
        }
        _ => true,
    }
}

fn validate_int(schema: &Value, input: &Py, strict: bool) -> V {
    let v = match input {
        Py::Int(i) => *i,
        Py::Bool(b) if !strict => *b as i64,
        Py::Float(f) if !strict => {
            if !f.is_finite() {
                return Err(err(
                    "finite_number",
                    "Input should be a finite number",
                    input,
                ));
            }
            if f.fract() != 0.0 {
                return Err(err(
                    "int_from_float",
                    "Input should be a valid integer, got a number with a fractional part",
                    input,
                ));
            }
            *f as i64
        }
        Py::Str(t) if !strict => match parse_py_int(t) {
            Some(i) => i,
            None => {
                return Err(err(
                    "int_parsing",
                    "Input should be a valid integer, unable to parse string as an integer",
                    input,
                ));
            }
        },
        _ => return Err(err("int_type", "Input should be a valid integer", input)),
    };
    if let Some(ge) = schema["ge"].as_i64()
        && v < ge
    {
        return Err(err(
            "greater_than_equal",
            format!("Input should be greater than or equal to {ge}"),
            input,
        ));
    }
    if let Some(le) = schema["le"].as_i64()
        && v > le
    {
        return Err(err(
            "less_than_equal",
            format!("Input should be less than or equal to {le}"),
            input,
        ));
    }
    Ok(Py::Int(v))
}

fn parse_py_int(t: &str) -> Option<i64> {
    let t = t.trim();
    let t = t.strip_prefix('+').unwrap_or(t);
    if t.is_empty() || t.starts_with('_') || t.ends_with('_') || t.contains("__") {
        return None;
    }
    t.replace('_', "").parse().ok()
}

fn validate_float(input: &Py, strict: bool) -> V {
    match input {
        Py::Float(f) => Ok(Py::Float(*f)),
        Py::Int(i) => Ok(Py::Float(*i as f64)),
        Py::Bool(b) if !strict => Ok(Py::Float(*b as i64 as f64)),
        Py::Str(t) if !strict => {
            let x = t.trim().to_lowercase().replace('_', "");
            let v = match x.as_str() {
                "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
                "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
                "nan" | "+nan" | "-nan" => Some(f64::NAN),
                _ => x.parse::<f64>().ok().filter(|_| !t.trim().is_empty()),
            };
            v.map(Py::Float).ok_or_else(|| {
                err(
                    "float_parsing",
                    "Input should be a valid number, unable to parse string as a number",
                    input,
                )
            })
        }
        _ => Err(err("float_type", "Input should be a valid number", input)),
    }
}

fn validate_bool(input: &Py, strict: bool) -> V {
    match input {
        Py::Bool(b) => Ok(Py::Bool(*b)),
        _ if strict => Err(err("bool_type", "Input should be a valid boolean", input)),
        Py::Int(0) => Ok(Py::Bool(false)),
        Py::Int(1) => Ok(Py::Bool(true)),
        Py::Float(f) if *f == 0.0 => Ok(Py::Bool(false)),
        Py::Float(f) if *f == 1.0 => Ok(Py::Bool(true)),
        Py::Str(t) => match t.to_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Ok(Py::Bool(false)),
            "1" | "on" | "t" | "true" | "y" | "yes" => Ok(Py::Bool(true)),
            _ => Err(err(
                "bool_parsing",
                "Input should be a valid boolean, unable to interpret input",
                input,
            )),
        },
        Py::Int(_) => Err(err(
            "bool_parsing",
            "Input should be a valid boolean, unable to interpret input",
            input,
        )),
        _ => Err(err("bool_type", "Input should be a valid boolean", input)),
    }
}

fn validate_tagged_union(schema: &Value, input: &Py, strict: bool, ctx: &Ctx) -> V {
    let choices = schema["choices"].as_object().cloned().unwrap_or_default();
    let disc = schema["discriminator"].as_str().unwrap_or("");
    if disc == "<python type name>" {
        // pydantic's JsonValue: by the input's Python type
        let key = match input {
            Py::Model { .. } => "dict".to_string(),
            other => other.type_name(),
        };
        return match choices.get(&key) {
            Some(c) => validate(c, input, strict, ctx),
            None => Err(err(
                "invalid-json-value",
                "input was not a valid JSON value",
                input,
            )),
        };
    }
    let Py::Dict(_) = input else {
        return Err(err(
            "model_attributes_type",
            "Input should be a valid dictionary or object to extract fields from",
            input,
        ));
    };
    let tag = input.get(disc);
    if !matches!(input, Py::Dict(m) if m.iter().any(|(k, _)| k == disc)) {
        return Err(err(
            "union_tag_not_found",
            format!("Unable to extract tag using discriminator '{disc}'"),
            input,
        ));
    }
    let tag_s = tag.str();
    match (tag, choices.get(&tag_s)) {
        (Py::Str(_), Some(c)) => {
            validate(c, input, strict, ctx).map_err(|f| prefix(f, Loc::Key(tag_s.clone())))
        }
        _ => {
            let expected: Vec<String> = choices.keys().map(|k| repr_str(k)).collect();
            Err(err(
                "union_tag_invalid",
                format!(
                    "Input tag '{tag_s}' found using '{disc}' does not match any of the expected tags: {}",
                    expected.join(", ")
                ),
                input,
            ))
        }
    }
}

fn validate_sequence(schema: &Value, input: &Py, strict: bool, ctx: &Ctx) -> V {
    let is_tuple = schema["type"] == "tuple";
    let items = match input {
        Py::List(a) | Py::Tuple(a) => a,
        _ => {
            return Err(if is_tuple {
                err("tuple_type", "Input should be a valid tuple", input)
            } else {
                err("list_type", "Input should be a valid list", input)
            });
        }
    };
    let positional: Vec<Value> = if is_tuple {
        schema["items_schema"]
            .as_array()
            .cloned()
            .unwrap_or_default()
    } else {
        vec![]
    };
    let variadic = if is_tuple {
        schema["variadic_item_index"].as_u64().map(|v| v as usize)
    } else {
        Some(0)
    };
    let item_schema = |i: usize| -> Option<&Value> {
        if !is_tuple {
            return Some(&schema["items_schema"]);
        }
        match variadic {
            Some(vi) if i >= vi => positional.get(vi),
            _ => positional.get(i),
        }
    };
    // a fixed-length tuple with too many items fails on its length alone
    if is_tuple && variadic.is_none() && items.len() > positional.len() {
        let n = positional.len();
        return Err(err(
            "too_long",
            format!(
                "Tuple should have at most {n} item{} after validation, not {}",
                if n == 1 { "" } else { "s" },
                items.len()
            ),
            input,
        ));
    }
    let mut out = vec![];
    let mut errors = vec![];
    for (i, it) in items.iter().enumerate() {
        let Some(sch) = item_schema(i) else { break };
        match validate(sch, it, strict, ctx) {
            Ok(v) => out.push(v),
            Err(Fail::Raise(e)) => return Err(Fail::Raise(e)),
            Err(f) => {
                if let Fail::Errors(es) = prefix(f, Loc::Idx(i)) {
                    errors.extend(es);
                }
            }
        }
    }
    if is_tuple && variadic.is_none() {
        for i in items.len()..positional.len() {
            errors.push(LineError {
                kind: "missing".into(),
                msg: "Field required".into(),
                loc: vec![Loc::Idx(i)],
                input: input.clone(),
            });
        }
    }
    // length constraints count the items that validated
    let kind = if is_tuple { "Tuple" } else { "List" };
    let plural = |n: u64| if n == 1 { "" } else { "s" };
    if let Some(n) = schema["min_length"].as_u64()
        && (out.len() as u64) < n
    {
        errors.push(LineError {
            kind: "too_short".into(),
            msg: format!(
                "{kind} should have at least {n} item{} after validation, not {}",
                plural(n),
                out.len()
            ),
            loc: vec![],
            input: input.clone(),
        });
    }
    if let Some(n) = schema["max_length"].as_u64()
        && (out.len() as u64) > n
    {
        errors.push(LineError {
            kind: "too_long".into(),
            msg: format!(
                "{kind} should have at most {n} item{} after validation, not {}",
                plural(n),
                out.len()
            ),
            loc: vec![],
            input: input.clone(),
        });
    }
    if !errors.is_empty() {
        return Err(Fail::Errors(errors));
    }
    Ok(if is_tuple {
        Py::Tuple(out)
    } else {
        Py::List(out)
    })
}

fn validate_fields(
    cls: &str,
    fields_schema: &Value,
    m: &[(String, Py)],
    input: &Py,
    strict: bool,
    ctx: &Ctx,
) -> V {
    let fields = fields_schema["fields"]
        .as_object()
        .cloned()
        .unwrap_or_default();
    let mut out = vec![];
    let mut errors = vec![];
    let mut used = vec![];
    for (name, f) in &fields {
        let key = f["validation_alias"].as_str().unwrap_or(name).to_string();
        let sch = &f["schema"];
        match m.iter().find(|(k, _)| *k == key) {
            Some((_, v)) => {
                used.push(key.clone());
                match validate(sch, v, strict, ctx) {
                    Ok(v) => out.push((name.clone(), v)),
                    Err(Fail::Raise(e)) => return Err(Fail::Raise(e)),
                    Err(fl) => {
                        if let Fail::Errors(es) = prefix(fl, Loc::Key(key)) {
                            errors.extend(es);
                        }
                    }
                }
            }
            None if sch["type"] == "default" => {
                out.push((name.clone(), Py::from_json(&sch["default"])))
            }
            None => errors.push(LineError {
                kind: "missing".into(),
                msg: "Field required".into(),
                loc: vec![Loc::Key(key)],
                input: input.clone(),
            }),
        }
    }
    if !errors.is_empty() {
        return Err(Fail::Errors(errors));
    }
    let extra = m
        .iter()
        .filter(|(k, _)| !used.contains(k))
        .cloned()
        .collect();
    Ok(Py::Model {
        cls: cls.into(),
        fields: out,
        extra,
    })
}

// ---------------------------------------------------------------- validators

enum Raised {
    /// `ValueError` / `AssertionError`: a validation error.
    Value(String),
    /// A `ValidationError` raised by the validator: its errors are merged
    /// into the outer ones (pydantic-core does not wrap them).
    Nested(Vec<LineError>),
    /// Anything else propagates.
    Other(Error),
}

type R<T> = Result<T, Raised>;

fn value(msg: impl Into<String>) -> Raised {
    Raised::Value(msg.into())
}

fn list_repr(v: &[Py]) -> String {
    Py::List(v.to_vec()).repr()
}

/// `unique_items_validator`.
fn unique_items(values: &[Py]) -> R<()> {
    for (i, v) in values.iter().enumerate() {
        if values[i + 1..].iter().any(|w| w.py_eq(v)) {
            return Err(value(format!(
                "Duplicate values found in {}.",
                list_repr(values)
            )));
        }
    }
    Ok(())
}

/// Python `set` repr of strings, in first-seen order (Python's own order
/// depends on its hash seed).
fn set_repr(items: &[Py]) -> String {
    let mut seen: Vec<&Py> = vec![];
    for i in items {
        if !seen.iter().any(|s| s.py_eq(i)) {
            seen.push(i);
        }
    }
    if seen.is_empty() {
        return "set()".into();
    }
    format!(
        "{{{}}}",
        seen.iter().map(|p| p.repr()).collect::<Vec<_>>().join(", ")
    )
}

/// `check_length(seq, valid_lengths=..., variable_name=...)`.
fn check_length(n: usize, valid: &[usize], valid_repr: &str, name: &str) -> R<()> {
    if valid.contains(&n) {
        Ok(())
    } else {
        Err(value(format!(
            "Length of {name} ({n}) not valid. Allowed lengths are {valid_repr}."
        )))
    }
}

/// `Affine.affine_matrix` / `Rotation.rotation_matrix`.
fn matrix<'a>(t: &'a Py, field: &str, what: &str) -> R<&'a Py> {
    let m = t.get(field);
    if !m.is_none() {
        Ok(m)
    } else if !t.get("path").is_none() {
        Err(Raised::Other(Error::NotImplemented(format!(
            "Loading {what} matrix from a Zarr array not yet implemented"
        ))))
    } else {
        Err(Raised::Other(Error::Runtime(format!(
            "Both self.{field} and self.path are None"
        ))))
    }
}

/// `transform.ndim`, or `None` where the class has no `ndim` (Python's
/// `hasattr`, which lets other exceptions through).
fn ndim(t: &Py) -> R<Option<usize>> {
    Ok(match t.cls() {
        "MapAxis" => Some(t.get("mapAxis").items().len()),
        "Translation" => Some(t.get("translation").items().len()),
        "Scale" => Some(t.get("scale").items().len()),
        "Affine" => Some(matrix(t, "affine", "affine")?.items().len()),
        "Rotation" => Some(matrix(t, "rotation", "rotation")?.items().len()),
        "Sequence" => {
            let first =
                t.get("transformations").items().first().ok_or_else(|| {
                    Raised::Other(Error::Index("tuple index out of range".into()))
                })?;
            match ndim(first)? {
                Some(n) => Some(n),
                None => {
                    return Err(Raised::Other(Error::Runtime(format!(
                        "{} does not have a dimensionality",
                        first.str()
                    ))));
                }
            }
        }
        "CoordinateSystem" => Some(t.get("axes").items().len()),
        _ => None,
    })
}

/// numpy `str()` of a float matrix (`np.array(rows)`), or numpy's error for
/// ragged rows.
fn np_matrix(rows: &[Py]) -> R<(Vec<Vec<f64>>, String)> {
    let m: Vec<Vec<f64>> = rows
        .iter()
        .map(|r| {
            r.items()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect()
        })
        .collect();
    if let Some(first) = m.first()
        && m.iter().any(|r| r.len() != first.len())
    {
        return Err(value(format!(
            "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. \
             The detected shape was ({},) + inhomogeneous part.",
            m.len()
        )));
    }
    Ok((m.clone(), crate::pyjson::np_array2_str(&m)))
}

/// The validator's key: `Class.method` (v0.6 / common models), or
/// `v04.Class.method` / `v05.Class.method`.
fn validator_key(name: &str) -> String {
    if name.ends_with(".sequence_validator") {
        return "sequence_validator".into();
    }
    let parts: Vec<&str> = name.split('.').collect();
    let short = parts[parts.len().saturating_sub(2)..].join(".");
    for v in ["v04", "v05"] {
        if name.starts_with(&format!("ome_zarr_models.{v}.")) {
            return format!("{v}.{short}");
        }
    }
    short
}

fn call_validator(name: &str, v: Py) -> R<Py> {
    let key = validator_key(name);
    match key.as_str() {
        "CoordinateSystem._ensure_unique_axis_names" => {
            let names: Vec<Py> = v.items().iter().map(|a| a.get("name").clone()).collect();
            unique_items(&names)?;
        }
        "CoordinateSystemIdentifier._ensure_either_name_or_path" => {
            if v.get("name").is_none() && v.get("path").is_none() {
                return Err(value("At least one of 'name' or 'path' must be set."));
            }
        }
        "Transform._ensure_consistent_input_output" => {
            let (i, o) = (v.get("input"), v.get("output"));
            if i.is_none() != o.is_none() {
                return Err(value(format!(
                    "Either both input and output must be defined or both must be omitted. Got input={} and output={}.",
                    i.str(),
                    o.str()
                )));
            }
        }
        "MapAxis.check_all_axes" => {
            let a = v.items();
            let mut seen: Vec<i64> = a
                .iter()
                .filter_map(|x| x.as_f64())
                .map(|x| x as i64)
                .collect();
            seen.sort();
            seen.dedup();
            if seen != (0..a.len() as i64).collect::<Vec<_>>() {
                return Err(value(format!(
                    "Not all axes present from 0 to {}: {}",
                    a.len() as i64 - 1,
                    v.repr()
                )));
            }
        }
        "ProjectAxis._created_outputs_unique" | "ProjectAxis._dropped_inputs_unique" => {
            if !v.is_none() {
                unique_items(v.items())?;
            }
        }
        "ProjectAxis._ensure_either_created_or_dropped" => {
            if v.get("createdOutputs").is_none() && v.get("droppedInputs").is_none() {
                return Err(value(
                    "At least one of 'createdOutputs' or 'droppedInputs' must be set.",
                ));
            }
        }
        "Affine._validate_affine" => {
            if !v.is_none() {
                let lens: Vec<Py> = v
                    .items()
                    .iter()
                    .map(|r| Py::Int(r.items().len() as i64))
                    .collect();
                if lens.windows(2).any(|w| w[0] != w[1])
                    && lens.iter().skip(1).any(|l| *l != lens[0])
                {
                    return Err(value(format!(
                        "Row lengths in affine matrix ({}) are not all equal.",
                        list_repr(&lens)
                    )));
                }
            }
        }
        "Affine.check_metadata_set" => {
            if v.get("affine").is_none() && v.get("path").is_none() {
                return Err(value("One of 'affine' or 'path' must be given"));
            }
        }
        "Rotation.check_metadata_set" => {
            if v.get("rotation").is_none() && v.get("path").is_none() {
                return Err(value("One of 'rotation' or 'path' must be given"));
            }
        }
        "Rotation.check_rotation_matrix" => {
            let rows = matrix(&v, "rotation", "rotation")?.items().to_vec();
            let (m, s) = np_matrix(&rows)?;
            let n = rows.len();
            let k = m.first().map(|r| r.len()).unwrap_or(0);
            let mut close = true;
            for i in 0..n {
                for j in 0..n {
                    let dot: f64 = (0..k).map(|l| m[i][l] * m[j][l]).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    if (dot - want).abs() > 1e-8 + 1e-5 * want {
                        close = false;
                    }
                }
            }
            if !close {
                return Err(value(format!(
                    "Provided matrix is not a pure rotation matrix: {s}"
                )));
            }
        }
        "LabelBase._parse_colors" => {
            if !v.is_none() {
                let mut counts: Vec<(Py, usize)> = vec![];
                for c in v.items() {
                    let lv = c.get("label_value").clone();
                    match counts.iter_mut().find(|(k, _)| k.py_eq(&lv)) {
                        Some((_, n)) => *n += 1,
                        None => counts.push((lv, 1)),
                    }
                }
                let dupes: Vec<Py> = counts
                    .into_iter()
                    .filter(|(_, n)| *n > 1)
                    .map(|(k, _)| k)
                    .collect();
                if !dupes.is_empty() {
                    return Err(value(format!(
                        "Duplicated label-value: {}.label-values must be unique across elements of `colors`.",
                        tuple_repr(&dupes)
                    )));
                }
            }
        }
        "LabelBase._check_label_values" => {
            let (colors, props) = (v.get("colors"), v.get("properties"));
            if !colors.is_none() && !props.is_none() {
                let p: Vec<Py> = props
                    .items()
                    .iter()
                    .map(|x| x.get("label_value").clone())
                    .collect();
                let c: Vec<Py> = colors
                    .items()
                    .iter()
                    .map(|x| x.get("label_value").clone())
                    .collect();
                let same = p.iter().all(|x| c.iter().any(|y| y.py_eq(x)))
                    && c.iter().all(|x| p.iter().any(|y| y.py_eq(x)));
                if !same {
                    return Err(value(format!(
                        "Inconsistent `label_value` attributes in `colors` and `properties`.The `properties` attributes have \
                         `label_values` {}, The `colors` attributes have `label_values` {}, ",
                        list_repr(&p),
                        list_repr(&c)
                    )));
                }
            }
        }
        "BaseOMEAttrs._parse_version" => {
            return match &v {
                Py::Str(s) if s == "0.6" || s == "0.6.dev4" => Ok(Py::Str("0.6".into())),
                other => Err(value(format!(
                    "Invalid version: '{}'. Must be '0.6' or '0.6.dev4'.",
                    other.str()
                ))),
            };
        }
        "Dataset.check_cs_input" => {
            for t in v.get("coordinateTransformations").items() {
                let i = t.get("input");
                if i.is_none() {
                    return Err(value("Transformations in datasets must have an input."));
                }
                if i.get("path").is_none() {
                    return Err(value(
                        "Input for dataset transforms must have a path field.",
                    ));
                }
                if !i.get("path").py_eq(v.get("path")) {
                    return Err(value(format!(
                        "Input for a dataset transform must be the dataset array path: '{}'. Got '{}' instead.",
                        v.get("path").str(),
                        i.get("path").str()
                    )));
                }
            }
        }
        "Dataset.check_cs_output" => {
            for t in v.get("coordinateTransformations").items() {
                let o = t.get("output");
                if o.is_none() {
                    return Err(value("Transformations in datasets must have an output."));
                }
                if o.get("name").is_none() {
                    return Err(value("Output name for dataset transforms must be set."));
                }
            }
        }
        "Dataset._ensure_transform_dimensionality" => {
            let ts = v.items();
            let pair: Option<(&Py, &Py)> = if ts.len() == 2 {
                Some((&ts[0], &ts[1]))
            } else if ts.len() == 1
                && ts[0].cls() == "Sequence"
                && ts[0].get("transformations").items().len() == 2
            {
                let inner = ts[0].get("transformations").items();
                Some((&inner[0], &inner[1]))
            } else {
                None
            };
            if let Some((s, t)) = pair
                && s.cls() == "Scale"
                && t.cls() == "Translation"
            {
                let (a, b) = (
                    s.get("scale").items().len(),
                    t.get("translation").items().len(),
                );
                if a != b {
                    return Err(value(format!(
                        "The length of the scale and translation vectors must be the same. Got {a} and {b}."
                    )));
                }
            }
        }
        "Multiscale._check_unique_system_names" => {
            let names: Vec<Py> = v.items().iter().map(|c| c.get("name").clone()).collect();
            if unique_items(&names).is_err() {
                return Err(value(format!(
                    "Duplicate coordinate system names: {}",
                    list_repr(&names)
                )));
            }
        }
        "Multiscale._ensure_ordered_scales" => {
            let mut scales: Vec<Vec<Py>> = vec![];
            for ds in v.items() {
                let t = &ds.get("coordinateTransformations").items()[0];
                match t.cls() {
                    "Scale" => scales.push(t.get("scale").items().to_vec()),
                    "Identity" => {}
                    "Sequence" => {
                        let first = t.get("transformations").items().first().ok_or_else(|| {
                            Raised::Other(Error::Index("tuple index out of range".into()))
                        })?;
                        if first.cls() != "Scale" {
                            return Err(value(
                                "Sequence transformation inside a multiscales must start with a scale transform",
                            ));
                        }
                        scales.push(first.get("scale").items().to_vec());
                    }
                    _ => {
                        return Err(value(
                            "Transform inside a multiscales must be a scale or sequence",
                        ));
                    }
                }
            }
            for i in 0..scales.len().saturating_sub(1) {
                let (s1, s2) = (&scales[i], &scales[i + 1]);
                for j in 0..s1.len() {
                    let b = s2.get(j).ok_or_else(|| {
                        Raised::Other(Error::Index("tuple index out of range".into()))
                    })?;
                    // Python's `s1[j] <= s2[j]` (False with NaN)
                    let (x, y) = (
                        s1[j].as_f64().unwrap_or(f64::NAN),
                        b.as_f64().unwrap_or(f64::NAN),
                    );
                    let ordered = matches!(
                        x.partial_cmp(&y),
                        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                    );
                    if !ordered {
                        return Err(value(format!(
                            "Dataset {i} has a lower resolution (scales = {}) than dataset {} (scales = {}).",
                            tuple_repr(s1),
                            i + 1,
                            tuple_repr(s2)
                        )));
                    }
                }
            }
        }
        "Multiscale._ensure_same_output_cs_for_all_datasets" => {
            let names: Vec<Py> = v
                .get("datasets")
                .items()
                .iter()
                .map(|ds| {
                    ds.get("coordinateTransformations").items()[0]
                        .get("output")
                        .clone()
                })
                .filter(|o| !o.is_none())
                .map(|o| o.get("name").clone())
                .collect();
            let set = set_repr(&names);
            let mut uniq: Vec<&Py> = vec![];
            for n in &names {
                if !uniq.iter().any(|u| u.py_eq(n)) {
                    uniq.push(n);
                }
            }
            if uniq.len() > 1 {
                return Err(value(format!(
                    "All `Dataset` instances of a `Multiscale`  must have the same output coordinate system. Got {set}."
                )));
            }
        }
        "Multiscale._ensure_axes_top_transforms" | "Multiscale._ensure_axes_dataset_transforms" => {
            let top = name.ends_with("top_transforms");
            let groups: Vec<(Option<usize>, Vec<Py>)> = if top {
                match v.get("coordinateTransformations") {
                    Py::None => vec![],
                    t => vec![(None, t.items().to_vec())],
                }
            } else {
                v.get("datasets")
                    .items()
                    .iter()
                    .enumerate()
                    .map(|(i, ds)| {
                        (
                            Some(i),
                            ds.get("coordinateTransformations").items().to_vec(),
                        )
                    })
                    .collect()
            };
            for (idx, txs) in groups {
                for tx in txs {
                    if let Some(tn) = ndim(&tx)? {
                        let dn = multiscale_ndim(&v)?;
                        if dn != tn {
                            let ty = tx.get("type").str();
                            let place = match idx {
                                None => "coordinateTransformations".to_string(),
                                Some(i) => format!("datasets[{i}].coordinateTransformations"),
                            };
                            return Err(value(format!(
                                "The length of axes does not match the dimensionality of the {ty} transform in {place}. \
                                 Got {dn} axes, but the {ty} transform has dimensionality {tn}"
                            )));
                        }
                    }
                }
            }
        }
        "Multiscale.check_dataset_transform_output" => {
            let mut outputs: Vec<Py> = vec![];
            for ds in v.get("datasets").items() {
                for t in ds.get("coordinateTransformations").items() {
                    outputs.push(t.get("output").clone());
                }
            }
            let mut uniq: Vec<&Py> = vec![];
            for o in &outputs {
                if !uniq.contains(&o) {
                    uniq.push(o);
                }
            }
            if uniq.len() > 1 {
                return Err(value(format!(
                    "All coordinate transform outputs inside 'datasets' must be identical. Got {}",
                    list_repr(&outputs)
                )));
            }
        }
        "Multiscale._check_cs_input_output" => {
            if let Py::Tuple(ts) | Py::List(ts) = v.get("coordinateTransformations") {
                let names: Vec<Py> = v
                    .get("coordinateSystems")
                    .items()
                    .iter()
                    .map(|c| c.get("name").clone())
                    .collect();
                let cs = set_repr(&names);
                for t in ts {
                    let (i, o) = (t.get("input"), t.get("output"));
                    if i.is_none() {
                        return Err(value(
                            "Transformations in coordinateTransformations must have an input.",
                        ));
                    }
                    if o.is_none() {
                        return Err(value(
                            "Transformations in coordinateTransformations must have an output.",
                        ));
                    }
                    if i.get("name").is_none() {
                        return Err(value(
                            "Input for coordinate transformations must provide a coordinate system name",
                        ));
                    }
                    if o.get("name").is_none() {
                        return Err(value(
                            "Output for coordinate transformations must provide a coordinate system name",
                        ));
                    }
                    if !names.iter().any(|n| n.py_eq(i.get("name"))) {
                        return Err(value(format!(
                            "Invalid input in coordinate transformation '{}': {}. Must be one of {cs}.",
                            t.get("name").str(),
                            i.get("name").str()
                        )));
                    }
                    let path_empty = matches!(o.get("path"), Py::None)
                        || matches!(o.get("path"), Py::Str(s) if s.is_empty());
                    if path_empty && !names.iter().any(|n| n.py_eq(o.get("name"))) {
                        return Err(value(format!(
                            "Invalid output in coordinate transformation '{}': {}. Must be one of {cs} or the path field must be set.",
                            t.get("name").str(),
                            o.get("name").str()
                        )));
                    }
                    if let Py::Str(p) = o.get("path")
                        && p.contains("..")
                    {
                        return Err(value(format!(
                            "Output paths in coordinate transformations must be downpointing. Got '{p}'."
                        )));
                    }
                }
            }
        }
        "Multiscale._validate_axis_length" => {
            for c in v.get("coordinateSystems").items() {
                check_length(
                    c.get("axes").items().len(),
                    &[2, 3, 4, 5],
                    "(2, 3, 4, 5)",
                    "axes",
                )?;
            }
        }
        "Multiscale._ensure_axis_types" => {
            for c in v.get("coordinateSystems").items() {
                let types: Vec<Py> = c
                    .get("axes")
                    .items()
                    .iter()
                    .map(|a| a.get("type").clone())
                    .collect();
                let count = |t: &str| {
                    types
                        .iter()
                        .filter(|x| matches!(x, Py::Str(s) if s == t))
                        .count()
                };
                check_length(count("space"), &[2, 3], "[2, 3]", "space axes")?;
                check_length(count("time"), &[0, 1], "[0, 1]", "time axes")?;
                check_length(count("channel"), &[0, 1], "[0, 1]", "channel axes")?;
                let custom = types.iter().filter(|x| !matches!(x, Py::Str(s) if s == "space" || s == "time" || s == "channel")).count();
                check_length(custom, &[0, 1], "[0, 1]", "custom axes")?;
                let ns = count("space");
                let tail = if ns == 0 {
                    &types[..]
                } else {
                    &types[types.len() - ns..]
                };
                if !tail.iter().all(|x| matches!(x, Py::Str(s) if s == "space")) {
                    return Err(value(format!(
                        "All space axes must be at the end of the axes list. Got axes with order: {}.",
                        list_repr(&types)
                    )));
                }
                if count("time") == 1 && !matches!(types.first(), Some(Py::Str(s)) if s == "time") {
                    return Err(value("Time axis must be at the beginning of axis list."));
                }
            }
        }
        k if k.starts_with("v04.") || k.starts_with("v05.") => {
            return call_validator_v0405(&k[4..], v);
        }
        other => {
            return Err(Raised::Other(Error::NotImplemented(format!(
                "pydantic validator {other}"
            ))));
        }
    }
    Ok(v)
}

/// `Multiscale.ndim` (its intrinsic coordinate system's).
fn multiscale_ndim(m: &Py) -> R<usize> {
    let t = &m.get("datasets").items()[0]
        .get("coordinateTransformations")
        .items()[0];
    let out = t.get("output");
    if out.is_none() {
        return Err(value(format!(
            "Output coordinate system in {} not found. ",
            t.str()
        )));
    }
    for c in m.get("coordinateSystems").items() {
        if c.get("name").py_eq(out.get("name")) {
            return Ok(c.get("axes").items().len());
        }
    }
    Err(value(format!(
        "No coordinate systems have the name '{}'",
        out.get("name").str()
    )))
}

// ---------------------------------------------------------------- entry points

/// Validate `input` as the pydantic model `entry` (`CoordinateSystem`,
/// `Transform` (the `AnyTransform` tagged union), `Multiscale`, `Omero`,
/// `LabelBase`, `SceneAttrs`, `BaseSceneAttrs`): `Ok` with the validated
/// value, or pydantic's `ValidationError` (or the exception a validator
/// raised that pydantic lets through).
pub fn validate_model(entry: &str, input: &Value) -> crate::Result<Py> {
    let all = schemas();
    let e = all
        .get(entry)
        .unwrap_or_else(|| panic!("no pydantic schema for {entry}"));
    let empty = serde_json::Map::new();
    let ctx = Ctx {
        defs: e["definitions"].as_object().unwrap_or(&empty),
    };
    let title = e["title"].as_str().unwrap_or(entry);
    match validate(&e["schema"], &Py::from_json(input), false, &ctx) {
        Ok(v) => Ok(v),
        Err(Fail::Errors(es)) => Err(Error::Validation(format_errors(title, &es))),
        Err(Fail::Raise(e)) => Err(e),
    }
}

/// v04 / v05 `Dataset._ensure_scale_translation` (a "before" validator): the
/// input validated as `Transforms(transforms=...)` first (its
/// `ValidationError` becomes the value error), then the transform types.
fn ensure_scale_translation(input: &Py) -> R<Py> {
    let all = schemas();
    let e = &all["Transforms"];
    let empty = serde_json::Map::new();
    let ctx = Ctx {
        defs: e["definitions"].as_object().unwrap_or(&empty),
    };
    let wrapped = Py::Dict(vec![("transforms".into(), input.clone())]);
    let ts = match validate(&e["schema"], &wrapped, false, &ctx) {
        Ok(v) => v.get("transforms").clone(),
        Err(Fail::Errors(es)) => return Err(Raised::Nested(es)),
        Err(Fail::Raise(e)) => return Err(Raised::Other(e)),
    };
    let ts = ts.items();
    check_length(ts.len(), &[1, 2], "[1, 2]", "transforms")?;
    let types: Vec<Py> = ts.iter().map(|t| t.get("type").clone()).collect();
    let names: Vec<String> = types.iter().map(Py::str).collect();
    if names == ["scale"] || names == ["scale", "translation"] {
        return Ok(input.clone());
    }
    // the (translation, scale) swap of ome-zarr-models never applies: the
    // validated items are plain `Transform`s, not `VectorTranslation`s
    Err(value(format!(
        "Expected a scale or (scale, translation) transform. Got {} instead.",
        tuple_repr(&types)
    )))
}

/// `_ndim` of a vector transform (`VectorScale` / `VectorTranslation`).
fn vector_ndim(t: &Py) -> Option<usize> {
    match t.cls() {
        "VectorScale" => Some(t.get("scale").items().len()),
        "VectorTranslation" => Some(t.get("translation").items().len()),
        _ => None,
    }
}

/// The validators of the v0.4 / v0.5 `Multiscale` and `Dataset` models.
fn call_validator_v0405(key: &str, v: Py) -> R<Py> {
    match key {
        "Dataset._ensure_transform_dimensionality" => {
            let nd: Vec<Py> = v
                .items()
                .iter()
                .filter_map(vector_ndim)
                .map(|n| Py::Int(n as i64))
                .collect();
            if nd.iter().any(|x| *x != nd[0]) {
                return Err(value(format!(
                    "The transforms have inconsistent dimensionality. Got transforms with dimensionality = {}.",
                    tuple_repr(&nd)
                )));
            }
        }
        "Multiscale._ensure_axis_length" => {
            check_length(v.items().len(), &[2, 3, 4, 5], "(2, 3, 4, 5)", "axes")?;
        }
        "Multiscale._ensure_unique_axis_names" => {
            let names: Vec<Py> = v.items().iter().map(|a| a.get("name").clone()).collect();
            if unique_items(&names).is_err() {
                return Err(value(format!(
                    "Axis names must be unique. Got {}",
                    list_repr(&names)
                )));
            }
        }
        "Multiscale._ensure_axis_types" => {
            let types: Vec<Py> = v.items().iter().map(|a| a.get("type").clone()).collect();
            check_axis_types(&types)?;
        }
        "Multiscale._ensure_ordered_scales" => {
            let scales: Vec<Py> = v
                .items()
                .iter()
                .map(|d| d.get("coordinateTransformations").items()[0].clone())
                .filter(|t| t.cls() == "VectorScale")
                .map(|t| t.get("scale").clone())
                .collect();
            check_ordered_scales(&scales)?;
        }
        "Multiscale._ensure_axes_dataset_transforms" | "Multiscale._ensure_axes_top_transforms" => {
            let n = v.get("axes").items().len();
            let groups: Vec<(Option<usize>, Vec<Py>)> = if key.ends_with("top_transforms") {
                match v.get("coordinateTransformations") {
                    Py::None => vec![],
                    t => vec![(None, t.items().to_vec())],
                }
            } else {
                v.get("datasets")
                    .items()
                    .iter()
                    .enumerate()
                    .map(|(i, d)| (Some(i), d.get("coordinateTransformations").items().to_vec()))
                    .collect()
            };
            for (idx, txs) in groups {
                for tx in txs {
                    if let Some(tn) = vector_ndim(&tx)
                        && tn != n
                    {
                        let ty = tx.get("type").str();
                        let place = match idx {
                            None => "coordinateTransformations".to_string(),
                            Some(i) => format!("datasets[{i}].coordinateTransformations"),
                        };
                        return Err(value(format!(
                            "The length of axes does not match the dimensionality of the {ty} transform in {place}. \
                             Got {n} axes, but the {ty} transform has dimensionality {tn}"
                        )));
                    }
                }
            }
        }
        other => {
            return Err(Raised::Other(Error::NotImplemented(format!(
                "pydantic validator v0.4/v0.5 {other}"
            ))));
        }
    }
    Ok(v)
}

/// The axis-type rules of `_ensure_axis_types` (v0.4 / v0.5).
fn check_axis_types(types: &[Py]) -> R<()> {
    let count = |t: &str| {
        types
            .iter()
            .filter(|x| matches!(x, Py::Str(s) if s == t))
            .count()
    };
    check_length(count("space"), &[2, 3], "[2, 3]", "space axes")?;
    check_length(count("time"), &[0, 1], "[0, 1]", "time axes")?;
    check_length(count("channel"), &[0, 1], "[0, 1]", "channel axes")?;
    let custom = types
        .iter()
        .filter(|x| !matches!(x, Py::Str(s) if s == "space" || s == "time" || s == "channel"))
        .count();
    check_length(custom, &[0, 1], "[0, 1]", "custom axes")?;
    let ns = count("space");
    let tail = if ns == 0 {
        types
    } else {
        &types[types.len() - ns..]
    };
    if !tail.iter().all(|x| matches!(x, Py::Str(s) if s == "space")) {
        return Err(value(format!(
            "All space axes must be at the end of the axes list. Got axes with order: {}.",
            list_repr(types)
        )));
    }
    if count("time") == 1 && !matches!(types.first(), Some(Py::Str(s)) if s == "time") {
        return Err(value("Time axis must be at the beginning of axis list."));
    }
    Ok(())
}

/// `check_ordered_scales`.
fn check_ordered_scales(scales: &[Py]) -> R<()> {
    for i in 0..scales.len().saturating_sub(1) {
        let (p1, p2) = (&scales[i], &scales[i + 1]);
        let (s1, s2) = (p1.items(), p2.items());
        for j in 0..s1.len() {
            let b = s2
                .get(j)
                .ok_or_else(|| Raised::Other(Error::Index("tuple index out of range".into())))?;
            let (x, y) = (
                s1[j].as_f64().unwrap_or(f64::NAN),
                b.as_f64().unwrap_or(f64::NAN),
            );
            if !matches!(
                x.partial_cmp(&y),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            ) {
                return Err(value(format!(
                    "Dataset {i} has a lower resolution (scales = {}) than dataset {} (scales = {}).",
                    p1.repr(),
                    i + 1,
                    p2.repr()
                )));
            }
        }
    }
    Ok(())
}
