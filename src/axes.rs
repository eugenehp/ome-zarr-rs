//! Axes class for validating and transforming axes.

use log::warn;
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::format::{CURRENT_FORMAT, Format};
use crate::zarr::JsonMap;

/// Map from well-known axis names to axis types.
pub const KNOWN_AXES: [(&str, &str); 5] = [
    ("x", "space"),
    ("y", "space"),
    ("z", "space"),
    ("c", "channel"),
    ("t", "time"),
];

pub const KNOWN_SPATIAL_UNITS: [&str; 26] = [
    "angstrom",
    "attometer",
    "femtometer",
    "picometer",
    "nanometer",
    "millimeter",
    "micrometer",
    "centimeter",
    "decimeter",
    "meter",
    "kilometer",
    "megameter",
    "hectometer",
    "gigameter",
    "terameter",
    "petameter",
    "exameter",
    "inch",
    "foot",
    "mile",
    "yard",
    "parsec",
    "yoctometer",
    "yottameter",
    "zeptometer",
    "zettameter",
];

pub const KNOWN_TEMPORAL_UNITS: [&str; 23] = [
    "attosecond",
    "centisecond",
    "day",
    "decisecond",
    "exasecond",
    "femtosecond",
    "gigasecond",
    "hectosecond",
    "hour",
    "kilosecond",
    "megasecond",
    "microsecond",
    "millisecond",
    "minute",
    "nanosecond",
    "petasecond",
    "picosecond",
    "second",
    "terasecond",
    "yoctosecond",
    "yottasecond",
    "zeptosecond",
    "zettasecond",
];

/// The type of a well-known axis name.
pub fn known_axis_type(name: &str) -> Option<&'static str> {
    KNOWN_AXES.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
}

/// Input for axes: a string like `"tczyx"`, names, or axis dicts.
#[derive(Debug, Clone, PartialEq)]
pub enum AxesInput {
    Names(Vec<String>),
    Dicts(Vec<JsonMap>),
}

impl AxesInput {
    /// Axes from a string such as `"czyx"`, one char per axis.
    pub fn from_chars(s: &str) -> AxesInput {
        AxesInput::Names(s.chars().map(|c| c.to_string()).collect())
    }

    /// Parse from JSON (list of strings or list of objects).
    pub fn from_json(v: &Value) -> Result<AxesInput> {
        match v {
            Value::String(s) => Ok(AxesInput::from_chars(s)),
            Value::Array(items) => {
                if items.iter().all(|i| i.is_string()) {
                    Ok(AxesInput::Names(
                        items
                            .iter()
                            .map(|i| i.as_str().unwrap().to_string())
                            .collect(),
                    ))
                } else if items.iter().all(|i| i.is_object()) {
                    Ok(AxesInput::Dicts(
                        items
                            .iter()
                            .map(|i| i.as_object().unwrap().clone())
                            .collect(),
                    ))
                } else {
                    Err(Error::type_err(
                        "`axes` must be a list of strings or a list of dicts containing 'name'",
                    ))
                }
            }
            other => Err(Error::type_err(format!("invalid axes: {other}"))),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            AxesInput::Names(v) => v.len(),
            AxesInput::Dicts(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Axis names (dicts without 'name' yield an error).
    pub fn names(&self) -> Result<Vec<String>> {
        match self {
            AxesInput::Names(v) => Ok(v.clone()),
            AxesInput::Dicts(v) => v
                .iter()
                .map(|d| {
                    d.get("name")
                        .and_then(|n| n.as_str())
                        .map(String::from)
                        .ok_or_else(|| {
                            Error::type_err("`axes` must be a list of dicts containing 'name'")
                        })
                })
                .collect(),
        }
    }
}

impl From<&str> for AxesInput {
    fn from(s: &str) -> Self {
        AxesInput::from_chars(s)
    }
}

impl From<Vec<&str>> for AxesInput {
    fn from(v: Vec<&str>) -> Self {
        AxesInput::Names(v.into_iter().map(String::from).collect())
    }
}

impl From<Vec<String>> for AxesInput {
    fn from(v: Vec<String>) -> Self {
        AxesInput::Names(v)
    }
}

impl From<&[String]> for AxesInput {
    fn from(v: &[String]) -> Self {
        AxesInput::Names(v.to_vec())
    }
}

impl From<Vec<JsonMap>> for AxesInput {
    fn from(v: Vec<JsonMap>) -> Self {
        AxesInput::Dicts(v)
    }
}

/// Validated axes.
#[derive(Debug, Clone, PartialEq)]
pub struct Axes {
    pub axes: Vec<JsonMap>,
    pub fmt: Format,
}

impl Axes {
    /// Constructor, transforms axes and validates. Returns `ValueError` if not valid.
    pub fn new(
        axes: Option<&AxesInput>,
        axes_units: Option<&JsonMap>,
        fmt: Format,
    ) -> Result<Axes> {
        let mut dicts = match axes {
            Some(a) => Self::axes_to_dicts(a),
            None if matches!(fmt, Format::V01 | Format::V02) => {
                Self::axes_to_dicts(&AxesInput::from_chars("tczyx"))
            }
            None => return Err(Error::value("axes must be provided")),
        };
        if let Some(units) = axes_units {
            for ax in dicts.iter_mut() {
                let Some(name) = ax.get("name").and_then(|n| n.as_str()).map(String::from) else {
                    continue;
                };
                if let Some(unit) = units.get(&name) {
                    ax.insert("unit".into(), unit.clone());
                    let u = unit.as_str().unwrap_or("");
                    let t = ax.get("type").and_then(|t| t.as_str());
                    if t == Some("space") && !KNOWN_SPATIAL_UNITS.contains(&u) {
                        warn!("Unit {u} for axis {name} is not a known spatial unit");
                    }
                    if t == Some("time") && !KNOWN_TEMPORAL_UNITS.contains(&u) {
                        warn!("Unit {u} for axis {name} is not a known temporal unit");
                    }
                }
            }
        }
        let a = Axes { axes: dicts, fmt };
        a.validate()?;
        Ok(a)
    }

    /// Validates with the default (current) format.
    pub fn with_default_format(axes: &AxesInput) -> Result<Axes> {
        Axes::new(Some(axes), None, CURRENT_FORMAT)
    }

    /// Raises ValueError if not valid.
    pub fn validate(&self) -> Result<()> {
        match self.fmt {
            Format::V01 | Format::V02 => Ok(()),
            Format::V03 => self.validate_03(),
            _ => self.validate_axes_types(),
        }
    }

    /// The axes as JSON for the given format: names for 0.3, dicts otherwise.
    pub fn to_list(&self, fmt: Format) -> Result<Value> {
        if fmt == Format::V03 {
            return Ok(json!(self.get_names()?));
        }
        Ok(Value::Array(
            self.axes.iter().map(|a| Value::Object(a.clone())).collect(),
        ))
    }

    fn axes_to_dicts(axes: &AxesInput) -> Vec<JsonMap> {
        match axes {
            AxesInput::Names(names) => names
                .iter()
                .map(|n| {
                    let mut d = JsonMap::new();
                    d.insert("name".into(), json!(n));
                    if let Some(t) = known_axis_type(n) {
                        d.insert("type".into(), json!(t));
                    }
                    d
                })
                .collect(),
            AxesInput::Dicts(d) => d.clone(),
        }
    }

    fn validate_axes_types(&self) -> Result<()> {
        let types: Vec<Option<String>> = self
            .axes
            .iter()
            .map(|a| a.get("type").and_then(|t| t.as_str()).map(String::from))
            .collect();
        let known = ["space", "channel", "time"];
        let unknown: Vec<&Option<String>> = types
            .iter()
            .filter(|t| !t.as_deref().is_some_and(|t| known.contains(&t)))
            .collect();
        if unknown.len() > 1 {
            return Err(Error::value(format!(
                "Too many unknown axes types. 1 allowed, found: [{}]",
                unknown
                    .iter()
                    .map(|t| t
                        .as_deref()
                        .map(crate::pyjson::repr_str)
                        .unwrap_or_else(|| "None".into()))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let last_index = |item: &str| types.iter().rposition(|t| t.as_deref() == Some(item));
        if let Some(i) = last_index("time")
            && i > 0
        {
            return Err(Error::value("'time' axis must be first dimension only"));
        }
        if types
            .iter()
            .filter(|t| t.as_deref() == Some("channel"))
            .count()
            > 1
        {
            return Err(Error::value("Only 1 axis can be type 'channel'"));
        }
        if let Some(ci) = last_index("channel") {
            let first_space = types.iter().position(|t| t.as_deref() == Some("space"));
            match first_space {
                Some(si) if ci > si => {
                    return Err(Error::value("'space' axes must come after 'channel'"));
                }
                None => return Err(Error::value("'space' is not in list")),
                _ => {}
            }
        }
        Ok(())
    }

    /// Returns a list of axis names.
    pub fn get_names(&self) -> Result<Vec<String>> {
        self.axes
            .iter()
            .map(|a| {
                a.get("name")
                    .and_then(|n| n.as_str())
                    .map(String::from)
                    .ok_or_else(|| Error::value(format!("Axis Dict {a:?} has no 'name'")))
            })
            .collect()
    }

    fn validate_03(&self) -> Result<()> {
        let names = self.get_names()?;
        let v: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        match v.len() {
            2 => {
                if v != ["y", "x"] {
                    return Err(Error::value(format!(
                        "2D data must have axes ('y', 'x') {}",
                        tuple_repr(&v)
                    )));
                }
            }
            3 => {
                if ![["z", "y", "x"], ["c", "y", "x"], ["t", "y", "x"]]
                    .iter()
                    .any(|o| v == o)
                {
                    return Err(Error::value(format!(
                        "3D data must have axes ('z', 'y', 'x') or ('c', 'y', 'x') or ('t', 'y', 'x'), not {}",
                        tuple_repr(&v)
                    )));
                }
            }
            4 => {
                if ![
                    ["t", "z", "y", "x"],
                    ["c", "z", "y", "x"],
                    ["t", "c", "y", "x"],
                ]
                .iter()
                .any(|o| v == o)
                {
                    return Err(Error::value("4D data must have axes tzyx or czyx or tcyx"));
                }
            }
            _ => {
                if v != ["t", "c", "z", "y", "x"] {
                    return Err(Error::value(
                        "5D data must have axes ('t', 'c', 'z', 'y', 'x')",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Python `repr()` of a tuple of strings.
fn tuple_repr(v: &[&str]) -> String {
    let items: Vec<String> = v.iter().map(|s| crate::pyjson::repr_str(s)).collect();
    if items.len() == 1 {
        format!("({},)", items[0])
    } else {
        format!("({})", items.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_axes() {
        let a = Axes::new(Some(&"tczyx".into()), None, Format::V04).unwrap();
        assert_eq!(a.get_names().unwrap(), vec!["t", "c", "z", "y", "x"]);
        assert_eq!(a.axes[1]["type"], "channel");
        Axes::new(Some(&"cyx".into()), None, Format::V05).unwrap();
    }

    #[test]
    fn invalid_axes() {
        // time not first
        assert!(Axes::new(Some(&"ztyx".into()), None, Format::V04).is_err());
        // channel after space
        assert!(Axes::new(Some(&"yxc".into()), None, Format::V04).is_err());
        // two unknown
        assert!(Axes::new(Some(&vec!["a", "b", "y", "x"].into()), None, Format::V04).is_err());
        // two channels
        assert!(Axes::new(Some(&"ccyx".into()), None, Format::V04).is_err());
        // v0.3 strict
        assert!(Axes::new(Some(&"xy".into()), None, Format::V03).is_err());
        Axes::new(Some(&"tcyx".into()), None, Format::V03).unwrap();
    }

    #[test]
    fn units() {
        let mut u = JsonMap::new();
        u.insert("x".into(), json!("micrometer"));
        let a = Axes::new(Some(&"yx".into()), Some(&u), Format::V04).unwrap();
        assert_eq!(a.axes[1]["unit"], "micrometer");
        assert_eq!(a.to_list(Format::V03).unwrap(), json!(["y", "x"]));
    }
}
