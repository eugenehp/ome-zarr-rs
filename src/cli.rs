//! Entrypoint for the `ome_zarr` command-line tool.

use std::sync::atomic::{AtomicI32, Ordering};

use log::LevelFilter;

use crate::argparse::{Arg, Exit, Parser, Value};
use crate::data::{SampleData, create_zarr};
use crate::error::{Error, Result};
use crate::format::{CURRENT_FORMAT, format_from_version};
use crate::scale::{Methods, ScaleFactors};
use crate::store::Store;
use crate::writer::{ImageOptions, write_image};
use crate::zarr::{Group, Mode, ZArray};

#[derive(Debug, Clone)]
pub struct Cli {
    /// increase loglevel for each use, e.g. -vvv
    pub verbose: u8,
    /// decrease loglevel for each use, e.q. -qqq
    pub quiet: u8,
    pub command: Command,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CreateMethod {
    Coins,
    Astronaut,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleMethod {
    Nearest,
    Resize,
    Laplacian,
    LocalMean,
    Zoom,
}

#[derive(Debug, Clone)]
pub enum Command {
    /// Print information about an OME-Zarr fileset
    Info {
        /// Path to image.zarr
        path: String,
        /// Print the minimum/maximum of all arrays (reads all data)
        stats: bool,
    },
    /// Download an OME-Zarr fileset
    Download { path: String, output: String },
    /// View an image: natively with threers (default), or in ome-ngff-validator with --web
    View {
        /// Path to image.zarr
        path: String,
        /// Port to serve the data (default: 8000), with --web
        port: u16,
        /// Force open in browser. Don't check for OME-Zarr data first.
        force: bool,
        /// Serve the data and open it in ome-ngff-validator in a web browser
        web: bool,
    },
    /// Open a directory of images in BioFile Finder in a browser
    Finder {
        /// Directory to open in BioFile Finder
        path: String,
        /// Port to serve the data (default: 8000)
        port: u16,
    },
    /// Create sample data
    Create {
        method: CreateMethod,
        path: String,
        /// OME-Zarr version to create. e.g. '0.4'
        format: Option<String>,
    },
    /// Write a zarr array as a multiscale OME-Zarr image
    Scale {
        input_array: String,
        output_directory: String,
        /// Dimensions of input data, i.e. 'zyx' or 'tczyx'.
        axes: String,
        /// copies the array metadata to the new group
        copy_metadata: bool,
        method: ScaleMethod,
        /// if true, don't write the base array
        in_place: bool,
        downscale: u64,
        max_layer: u32,
    },
    /// Add csv data to labels properties
    CsvToLabels {
        /// path to csv file
        csv_path: String,
        /// csv column name containing ID for identifying label properties to update
        csv_id: String,
        /// Comma-separated list of columns to read from csv to zarr
        csv_keys: String,
        /// path to local zarr plate or image
        zarr_path: String,
        /// Labels properties key. Values should match csv_id column values
        zarr_id: String,
    },
    /// Render an image (with labels) to a PNG file using the GPU (threers)
    Render {
        path: String,
        /// Output PNG file
        output: String,
        width: u32,
        height: u32,
        /// z plane (default: middle)
        z: Option<u64>,
        /// time point
        t: u64,
        /// pyramid level (default: fit the output size)
        level: Option<usize>,
        /// hide labels
        no_labels: bool,
    },
}

/// The CLI's logger, formatted as Python's `logging` prints: after
/// `logging.basicConfig`, `LEVEL:name:message` at the configured level; before
/// it (commands that never configure logging), Python's last-resort handler:
/// the bare message, WARNING and above.
struct PyLogger;

/// Python's numeric threshold (`logging.WARNING` = 30), or -1 while unconfigured.
static PY_LEVEL: AtomicI32 = AtomicI32::new(-1);

fn py_level(l: log::Level) -> (i32, &'static str) {
    match l {
        log::Level::Error => (40, "ERROR"),
        log::Level::Warn => (30, "WARNING"),
        log::Level::Info => (20, "INFO"),
        log::Level::Debug => (10, "DEBUG"),
        log::Level::Trace => (5, "TRACE"),
    }
}

impl log::Log for PyLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        let (n, _) = py_level(m.level());
        let threshold = PY_LEVEL.load(Ordering::Relaxed);
        // other crates' logs (Python's root logger shows those of zarr,
        // fsspec, ...) are kept to warnings
        let ours = m.target().starts_with("ome_zarr");
        if threshold < 0 || !ours {
            n >= 30
        } else {
            n >= threshold
        }
    }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        if PY_LEVEL.load(Ordering::Relaxed) < 0 {
            eprintln!("{}", r.args());
        } else {
            let (_, name) = py_level(r.level());
            eprintln!("{name}:{}:{}", r.target().replace("::", "."), r.args());
        }
    }
    fn flush(&self) {}
}

fn install_logger() {
    static LOGGER: PyLogger = PyLogger;
    if log::set_logger(&LOGGER).is_ok() {
        log::set_max_level(LevelFilter::Trace);
    }
}

/// `config_logging(loglevel, args)` of ome-zarr-py: each `-v` lowers the level
/// by 10 and each `-q` raises it by 10, starting from `loglevel` (WARNING).
pub fn config_logging(verbose: u8, quiet: u8) {
    install_logger();
    let level = 30 - 10 * verbose as i32 + 10 * quiet as i32;
    PY_LEVEL.store(level.max(0), Ordering::Relaxed);
}

/// Run a parsed command.
pub fn run(cli: Cli) -> Result<()> {
    install_logger();
    // as ome-zarr-py, `scale` and `csv_to_labels` leave logging unconfigured
    if !matches!(
        cli.command,
        Command::Scale { .. } | Command::CsvToLabels { .. }
    ) {
        config_logging(cli.verbose, cli.quiet);
    }
    match cli.command {
        Command::Info { path, stats } => {
            crate::utils::info(&path, stats)?;
        }
        Command::Download { path, output } => crate::utils::download(&path, &output)?,
        Command::View {
            path,
            port,
            force,
            web,
        } => {
            if web {
                crate::utils::view(&path, port, false, force)?;
            } else {
                #[cfg(feature = "render")]
                crate::render::viewer::run(&path)?;
                #[cfg(not(feature = "render"))]
                return Err(crate::Error::NotImplemented(
                    "the native viewer needs the `render` feature; use --web".into(),
                ));
            }
        }
        Command::Finder { path, port } => {
            crate::utils::finder(&path, port, false)?;
        }
        Command::Create {
            method,
            path,
            format,
        } => {
            let fmt = match format {
                Some(v) => format_from_version(&v)?,
                None => CURRENT_FORMAT,
            };
            let m = match method {
                CreateMethod::Coins => SampleData::Coins,
                CreateMethod::Astronaut => SampleData::Astronaut,
            };
            create_zarr(&path, m, fmt)?;
        }
        Command::Scale {
            input_array,
            output_directory,
            axes,
            copy_metadata,
            method,
            downscale,
            max_layer,
            ..
        } => {
            let method = match method {
                ScaleMethod::Nearest => Methods::Nearest,
                ScaleMethod::Resize => Methods::Resize,
                ScaleMethod::LocalMean => Methods::LocalMean,
                ScaleMethod::Zoom => Methods::Zoom,
                ScaleMethod::Laplacian => {
                    return Err(Error::value("'laplacian' is not a valid Methods"));
                }
            };
            let store = Store::from_url(&input_array, true)?;
            let base = ZArray::open(&store, "")?;
            let factors = (1..=max_layer).map(|i| downscale.pow(i)).collect();
            let opts = ImageOptions {
                axes: Some(axes.as_str().into()),
                method: Some(method),
                scale_factors: Some(ScaleFactors::Ints(factors)),
                ..Default::default()
            };
            write_image(&base.to_lazy(), output_directory.as_str(), &opts)?;
            if copy_metadata {
                let attrs = base.attrs()?;
                let keys: Vec<String> = attrs.keys().cloned().collect();
                println!(
                    "copying attribute keys: {}",
                    crate::pyjson::str_list_repr(&keys)
                );
                Group::open_url(&output_directory, Mode::Append, None)?.update_attrs(&attrs)?;
            }
        }
        Command::CsvToLabels {
            csv_path,
            csv_id,
            csv_keys,
            zarr_path,
            zarr_id,
        } => {
            println!("csv_to_labels {csv_path} {zarr_path}");
            crate::csv::csv_to_zarr(&csv_path, &csv_id, &csv_keys, &zarr_path, &zarr_id)?;
        }
        Command::Render {
            path,
            output,
            width,
            height,
            z,
            t,
            level,
            no_labels,
        } => {
            #[cfg(feature = "render")]
            {
                let opts = crate::render::RenderOptions {
                    width,
                    height,
                    z,
                    t,
                    level,
                    show_labels: !no_labels,
                    ..Default::default()
                };
                crate::render::render_to_png(path.as_str(), std::path::Path::new(&output), &opts)?;
                println!("wrote {output}");
            }
            #[cfg(not(feature = "render"))]
            {
                let _ = (path, output, width, height, z, t, level, no_labels);
                return Err(crate::Error::NotImplemented(
                    "rendering needs the `render` feature".into(),
                ));
            }
        }
    }
    Ok(())
}

/// Parse `args` (without the program name) and run, as `ome_zarr.cli.main`.
pub fn main_with_args<I, T>(args: I) -> Result<()>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let argv: Vec<String> = args
        .into_iter()
        .map(|a| a.into().to_string_lossy().into_owned())
        .collect();
    let cli = parse_args(&argv).map_err(|e| Error::value(format!("{}{}", e.stdout, e.stderr)))?;
    run(cli)
}

/// The `ome_zarr` argument parser, as ome-zarr-py's `argparse` definitions.
/// With `extensions`, the additions of this crate: `view --web` and `render`.
pub fn parser(extensions: bool) -> Parser {
    let sub = |name: &str| Parser::new(&format!("ome_zarr {name}"));
    let info = sub("info")
        .arg(Arg::positional("path").help("Path to image.zarr"))
        .arg(Arg::option(&["--stats"]).flag());
    let download = sub("download")
        .arg(Arg::positional("path"))
        .arg(Arg::option(&["--output"]).default(Value::Str(".".into())));
    let mut view = sub("view")
        .arg(Arg::positional("path").help("Path to image.zarr to open in ome-ngff-validator"))
        .arg(
            Arg::option(&["--port"])
                .int()
                .default(Value::Int(8000))
                .help("Port to serve the data (default: 8000)"),
        )
        .arg(
            Arg::option(&["--force", "-f"])
                .flag()
                .help("Force open in browser. Don't check for OME-Zarr data first."),
        );
    if extensions {
        view = view.arg(Arg::option(&["--web"]).flag().help(
            "Open in ome-ngff-validator in a web browser (default: the native threers viewer)",
        ));
    }
    let finder = sub("finder")
        .arg(Arg::positional("path").help("Directory to open in BioFile Finder"))
        .arg(
            Arg::option(&["--port"])
                .int()
                .default(Value::Int(8000))
                .help("Port to serve the data (default: 8000)"),
        );
    let create = sub("create")
        .arg(
            Arg::option(&["--method"])
                .choices(&["coins", "astronaut"])
                .default(Value::Str("coins".into())),
        )
        .arg(Arg::positional("path"))
        .arg(Arg::option(&["--format"]).help("OME-Zarr version to create. e.g. '0.4'"));
    let scale = sub("scale")
        .arg(Arg::positional("input_array"))
        .arg(Arg::positional("output_directory"))
        .arg(Arg::positional("axes").help("Dimensions of input data, i.e. 'zyx' or 'tczyx'."))
        .arg(
            Arg::option(&["--copy-metadata"])
                .flag()
                .help("copies the array metadata to the new group"),
        )
        .arg(
            Arg::option(&["--method"])
                .choices(&["nearest", "resize", "laplacian", "local_mean", "zoom"])
                .default(Value::Str("resize".into())),
        )
        .arg(
            Arg::option(&["--in-place"])
                .flag()
                .help("if true, don't write the base array"),
        )
        .arg(Arg::option(&["--downscale"]).int().default(Value::Int(2)))
        .arg(Arg::option(&["--max_layer"]).int().default(Value::Int(4)));
    let csv_to_labels = sub("csv_to_labels")
        .arg(Arg::positional("csv_path").help("path to csv file"))
        .arg(
            Arg::positional("csv_id")
                .help("csv column name containing ID for identifying label properties to update"),
        )
        .arg(
            Arg::positional("csv_keys")
                .help("Comma-separated list of columns to read from csv to zarr"),
        )
        .arg(Arg::positional("zarr_path").help("path to local zarr plate or image"))
        .arg(
            Arg::positional("zarr_id")
                .help("Labels properties key. Values should match csv_id column values"),
        );
    let mut subs = vec![
        ("info".to_string(), info),
        ("download".to_string(), download),
        ("view".to_string(), view),
        ("finder".to_string(), finder),
        ("create".to_string(), create),
        ("scale".to_string(), scale),
        ("csv_to_labels".to_string(), csv_to_labels),
    ];
    if extensions {
        let render = sub("render")
            .arg(Arg::positional("path").help("Path to image.zarr (local or online)"))
            .arg(
                Arg::option(&["--output", "-o"])
                    .default(Value::Str("ome_zarr.png".into()))
                    .help("Output PNG file (default: ome_zarr.png)"),
            )
            .arg(Arg::option(&["--width"]).int().default(Value::Int(512)))
            .arg(Arg::option(&["--height"]).int().default(Value::Int(512)))
            .arg(
                Arg::option(&["--z"])
                    .int()
                    .help("z plane (default: middle)"),
            )
            .arg(
                Arg::option(&["--t"])
                    .int()
                    .default(Value::Int(0))
                    .help("time point"),
            )
            .arg(
                Arg::option(&["--level"])
                    .int()
                    .help("pyramid level (default: fit the output size)"),
            )
            .arg(Arg::option(&["--no-labels"]).flag().help("hide labels"));
        subs.push(("render".to_string(), render));
    }
    Parser::new("ome_zarr")
        .arg(
            Arg::option(&["-v", "--verbose"])
                .count()
                .help("increase loglevel for each use, e.g. -vvv"),
        )
        .arg(
            Arg::option(&["-q", "--quiet"])
                .count()
                .help("decrease loglevel for each use, e.q. -qqq"),
        )
        .subcommands("command", subs)
}

/// Parse the command line (without the program name), as ome-zarr-py's
/// `main` does; `Err` holds what argparse prints and its exit code.
pub fn parse_args(argv: &[String]) -> std::result::Result<Cli, Exit> {
    let ns = parser(true).parse(argv)?;
    let s = |k: &str| ns.str(k).unwrap_or_default();
    let int = |k: &str| ns.int(k);
    let command = match ns.command.as_deref().unwrap_or("") {
        "info" => Command::Info {
            path: s("path"),
            stats: ns.bool("stats"),
        },
        "download" => Command::Download {
            path: s("path"),
            output: s("output"),
        },
        "view" => Command::View {
            path: s("path"),
            port: int("port").unwrap_or(8000) as u16,
            force: ns.bool("force"),
            web: ns.bool("web"),
        },
        "finder" => Command::Finder {
            path: s("path"),
            port: int("port").unwrap_or(8000) as u16,
        },
        "create" => Command::Create {
            method: if s("method") == "astronaut" {
                CreateMethod::Astronaut
            } else {
                CreateMethod::Coins
            },
            path: s("path"),
            format: ns.str("format"),
        },
        "scale" => Command::Scale {
            input_array: s("input_array"),
            output_directory: s("output_directory"),
            axes: s("axes"),
            copy_metadata: ns.bool("copy_metadata"),
            method: match s("method").as_str() {
                "nearest" => ScaleMethod::Nearest,
                "laplacian" => ScaleMethod::Laplacian,
                "local_mean" => ScaleMethod::LocalMean,
                "zoom" => ScaleMethod::Zoom,
                _ => ScaleMethod::Resize,
            },
            in_place: ns.bool("in_place"),
            downscale: int("downscale").unwrap_or(2) as u64,
            max_layer: int("max_layer").unwrap_or(4) as u32,
        },
        "csv_to_labels" => Command::CsvToLabels {
            csv_path: s("csv_path"),
            csv_id: s("csv_id"),
            csv_keys: s("csv_keys"),
            zarr_path: s("zarr_path"),
            zarr_id: s("zarr_id"),
        },
        _ => Command::Render {
            path: s("path"),
            output: s("output"),
            width: int("width").unwrap_or(512) as u32,
            height: int("height").unwrap_or(512) as u32,
            z: int("z").map(|v| v as u64),
            t: int("t").unwrap_or(0) as u64,
            level: int("level").map(|v| v as usize),
            no_labels: ns.bool("no_labels"),
        },
    };
    Ok(Cli {
        verbose: int("verbose").unwrap_or(0) as u8,
        quiet: int("quiet").unwrap_or(0) as u8,
        command,
    })
}
