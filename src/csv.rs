//! Add values from a CSV file to label properties (`ome_zarr.csv`).

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::zarr::{Group, JsonMap, Mode};

/// Column types: `d` (double), `l` (long), `s` (string), `b` (bool).
pub const COLUMN_TYPES: [&str; 4] = ["d", "l", "s", "b"];

/// Parse string value from csv, according to [`COLUMN_TYPES`].
pub fn parse_csv_value(value: &str, col_type: &str) -> Value {
    match col_type {
        "d" => match value.trim().parse::<f64>() {
            Ok(f) => json!(f),
            Err(_) => json!(value),
        },
        "l" => match value.trim().parse::<f64>() {
            Ok(f) => json!(f.round_ties_even() as i64),
            Err(_) => json!(value),
        },
        // Python's bool(str): true for any non-empty string
        "b" => json!(!value.is_empty()),
        _ => json!(value),
    }
}

/// Python `str()` of a JSON value.
pub fn py_str(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "None".into(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => if *b { "True" } else { "False" }.into(),
        Some(Value::Number(n)) => {
            if let Some(f) = n.as_f64().filter(|_| n.is_f64()) {
                if f.fract() == 0.0 && f.abs() < 1e16 {
                    format!("{f:.1}")
                } else {
                    f.to_string()
                }
            } else {
                n.to_string()
            }
        }
        Some(other) => other.to_string(),
    }
}

/// Add keys:values from a CSV file to the label properties of a Plate or Image.
///
/// Column types can be specified with `#d` (float), `#l` (int) or `#b`
/// (boolean), e.g. `"col1#d,col2#l,col5"`.
pub fn csv_to_zarr(
    csv_path: &str,
    csv_id: &str,
    csv_keys: &str,
    zarr_path: &str,
    zarr_id: &str,
) -> Result<()> {
    let mut cols_types: BTreeMap<String, String> = BTreeMap::new();
    for col in csv_keys.split(',') {
        match col.rsplit_once('#') {
            Some((name, t)) => {
                let t = if COLUMN_TYPES.contains(&t) { t } else { "s" };
                cols_types.insert(name.to_string(), t.to_string());
            }
            None => {
                cols_types.insert(col.to_string(), "s".to_string());
            }
        }
    }
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_path(csv_path)
        .map_err(|e| Error::Io(std::io::Error::other(e)))?;
    let mut columns: Option<Vec<String>> = None;
    let mut id_column = 0;
    let mut props_by_id: BTreeMap<String, JsonMap> = BTreeMap::new();
    for row in reader.records() {
        let row = row.map_err(|e| Error::value(e.to_string()))?;
        let row: Vec<String> = row.iter().map(String::from).collect();
        match &columns {
            None => {
                id_column = row.iter().position(|c| c == csv_id).ok_or_else(|| {
                    Error::value(format!(
                        "csv_id '{csv_id}' should match acsv column name: {row:?}"
                    ))
                })?;
                columns = Some(row);
            }
            Some(cols) => {
                let row_id = row.get(id_column).cloned().unwrap_or_default();
                let mut props = JsonMap::new();
                for (name, value) in cols.iter().zip(&row) {
                    if let Some(t) = cols_types.get(name) {
                        props.insert(name.clone(), parse_csv_value(value, t));
                    }
                }
                props_by_id.insert(row_id, props);
            }
        }
    }
    dict_to_zarr(&props_by_id, zarr_path, zarr_id)
}

/// Add keys:values to the label properties of an ome-zarr Plate or Image.
///
/// For each labels properties dict, the value of the property named
/// `zarr_id` selects the entry of `props_to_add` whose keys and values are
/// added.
pub fn dict_to_zarr(
    props_to_add: &BTreeMap<String, JsonMap>,
    zarr_path: &str,
    zarr_id: &str,
) -> Result<()> {
    let root = Group::open_url(zarr_path, Mode::Append, None)?;
    // As ome-zarr-py, only the top-level attributes are inspected (so zarr v3
    // stores, whose metadata is under "ome", are rejected).
    let attrs = root.attrs()?;
    let plate = attrs.get("plate").filter(|p| !p.is_null()).cloned();
    let multiscales = attrs.contains_key("multiscales");
    if plate.is_none() && !multiscales {
        return Err(Error::type_err(
            "zarr_path must be to plate.zarr or image.zarr",
        ));
    }
    let labels_paths: Vec<String> = match &plate {
        Some(p) => p
            .get("wells")
            .and_then(|w| w.as_array())
            .cloned()
            .unwrap_or_default()
            .iter()
            .filter_map(|w| w.get("path").and_then(|p| p.as_str()))
            .map(|wp| {
                Path::new(zarr_path)
                    .join(wp)
                    .join("0")
                    .join("labels")
                    .join("0")
                    .to_string_lossy()
                    .to_string()
            })
            .collect(),
        None => vec![
            Path::new(zarr_path)
                .join("labels")
                .join("0")
                .to_string_lossy()
                .to_string(),
        ],
    };
    for path in labels_paths {
        let label_group = Group::open_url(&path, Mode::Append, None)?;
        let mut attrs = label_group.attrs()?;
        let v3 = attrs.contains_key("ome");
        let container: &mut JsonMap = if v3 {
            attrs
                .get_mut("ome")
                .and_then(|o| o.as_object_mut())
                .unwrap()
        } else {
            &mut attrs
        };
        let Some(properties) = container
            .get_mut("image-label")
            .and_then(|il| il.get_mut("properties"))
            .and_then(|p| p.as_array_mut())
        else {
            continue;
        };
        for props in properties.iter_mut() {
            let Some(pd) = props.as_object_mut() else {
                continue;
            };
            let id = py_str(pd.get(zarr_id));
            if let Some(add) = props_to_add.get(&id) {
                for (k, v) in add {
                    pd.insert(k.clone(), v.clone());
                }
            }
        }
        label_group.update_attrs(&attrs)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_values() {
        assert_eq!(parse_csv_value("1.5", "d"), json!(1.5));
        assert_eq!(parse_csv_value("2.5", "l"), json!(2));
        assert_eq!(parse_csv_value("abc", "l"), json!("abc"));
        assert_eq!(parse_csv_value("False", "b"), json!(true));
        assert_eq!(py_str(Some(&json!(3))), "3");
        assert_eq!(py_str(Some(&json!(3.0))), "3.0");
    }
}
