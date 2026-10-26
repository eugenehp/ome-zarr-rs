#[cfg(not(target_arch = "wasm32"))]
mod native {
    use ome_zarr::Error;
    use ome_zarr::cli::{parse_args, run};

    pub fn main() {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let cli = match parse_args(&argv) {
            Ok(cli) => cli,
            Err(exit) => {
                print!("{}", exit.stdout);
                eprint!("{}", exit.stderr);
                std::process::exit(exit.code);
            }
        };
        if let Err(e) = run(cli) {
            match e {
                Error::Assertion(msg) => {
                    // logging.getLogger("ome_zarr.cli").error(error)
                    log::error!(target: "ome_zarr::cli", "{msg}");
                    std::process::exit(2);
                }
                other => {
                    eprintln!("Error: {other}");
                    std::process::exit(1);
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::main()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
