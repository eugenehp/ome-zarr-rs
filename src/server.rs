//! The static file server of `view --web` and `finder`: ome-zarr-py's
//! `CORSRequestHandler` (RangeHTTPServer's `RangeRequestHandler` on Python's
//! `SimpleHTTPRequestHandler`, plus `Access-Control-Allow-Origin: *`), with
//! the same responses: HTTP/1.0, directory redirects and listings, the HTML
//! error pages, range requests, `If-Modified-Since`, and the request log on
//! stderr.

use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};

use log::debug;

use crate::error::Result;

/// A response: status line, headers, body and the lines the server logs.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    pub status: u16,
    pub reason: String,
    pub headers: Vec<(String, String)>,
    /// The body, or the file range to send.
    pub body: Body,
    /// `log_error` messages (`code 404, message File not found`).
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Body {
    Bytes(Vec<u8>),
    File { path: PathBuf, start: u64, len: u64 },
}

/// `http.server.BaseHTTPRequestHandler.responses`: (reason, explanation).
fn status_text(code: u16) -> (&'static str, &'static str) {
    match code {
        200 => ("OK", "Request fulfilled, document follows"),
        206 => ("Partial Content", "Partial content follows"),
        301 => (
            "Moved Permanently",
            "Object moved permanently -- see URI list",
        ),
        304 => ("Not Modified", "Document has not changed since given time"),
        400 => ("Bad Request", "Bad request syntax or unsupported method"),
        404 => ("Not Found", "Nothing matches the given URI"),
        416 => (
            "Requested Range Not Satisfiable",
            "Cannot satisfy request range",
        ),
        501 => ("Not Implemented", "Server does not support this operation"),
        _ => ("", ""),
    }
}

/// `html.escape(s, quote=False)`.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// `send_error(code, message)`.
fn error(code: u16, message: &str) -> Response {
    let (_, explain) = status_text(code);
    let body = format!(
        "<!DOCTYPE HTML>\n<html lang=\"en\">\n    <head>\n        <meta charset=\"utf-8\">\n        \
         <title>Error response</title>\n    </head>\n    <body>\n        <h1>Error response</h1>\n        \
         <p>Error code: {code}</p>\n        <p>Message: {}.</p>\n        \
         <p>Error code explanation: {code} - {}.</p>\n    </body>\n</html>\n",
        html_escape(message),
        html_escape(explain)
    );
    Response {
        status: code,
        reason: message.to_string(),
        headers: vec![
            ("Connection".into(), "close".into()),
            ("Content-Type".into(), "text/html;charset=utf-8".into()),
            ("Content-Length".into(), body.len().to_string()),
        ],
        body: Body::Bytes(body.into_bytes()),
        errors: vec![format!("code {code}, message {message}")],
    }
}

fn ok(code: u16, headers: Vec<(String, String)>, body: Body) -> Response {
    Response {
        status: code,
        reason: status_text(code).0.into(),
        headers,
        body,
        errors: vec![],
    }
}

/// `mimetypes.guess_type` for the files a fileset contains (Python's table).
fn guess_type(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "json" => "application/json",
        "csv" => "text/csv",
        "txt" => "text/plain",
        "html" | "htm" => "text/html",
        "xml" => "text/xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "tif" | "tiff" => "image/tiff",
        "js" => "text/javascript",
        "css" => "text/css",
        "svg" => "image/svg+xml",
        "zip" => "application/zip",
        "gz" => "application/gzip",
        _ => "application/octet-stream",
    }
}

/// `posixpath.normpath` of an absolute URL path.
fn normpath(p: &str) -> String {
    let mut out: Vec<&str> = vec![];
    for seg in p.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                out.pop();
            }
            s => out.push(s),
        }
    }
    format!("/{}", out.join("/"))
}

/// `SimpleHTTPRequestHandler.translate_path`.
fn translate_path(root: &Path, url_path: &str) -> PathBuf {
    let path = url_path.split('?').next().unwrap_or("");
    let path = path.split('#').next().unwrap_or("");
    let trailing_slash = path.trim_end().ends_with('/');
    let decoded = percent_encoding::percent_decode_str(path)
        .decode_utf8_lossy()
        .to_string();
    let mut out = root.to_path_buf();
    for word in normpath(&decoded).split('/').filter(|w| !w.is_empty()) {
        if word.contains('/') || word == "." || word == ".." {
            continue;
        }
        out.push(word);
    }
    if trailing_slash {
        let mut s = out.into_os_string();
        s.push("/");
        out = PathBuf::from(s);
    }
    out
}

/// `email.utils.formatdate(t, usegmt=True)`.
fn http_date(t: std::time::SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Utc> = t.into();
    dt.format("%a, %d %b %Y %H:%M:%S GMT").to_string()
}

/// `urllib.parse.quote(name)`: keep alphanumerics and `_.-~/`.
fn quote(s: &str) -> String {
    const SET: &percent_encoding::AsciiSet = &percent_encoding::NON_ALPHANUMERIC
        .remove(b'_')
        .remove(b'.')
        .remove(b'-')
        .remove(b'~')
        .remove(b'/');
    percent_encoding::utf8_percent_encode(s, SET).to_string()
}

/// `list_directory`.
fn list_directory(dir: &Path, url_path: &str) -> Response {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return error(404, "No permission to list directory");
    };
    let mut names: Vec<String> = rd
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    names.sort_by_key(|n| n.to_lowercase());
    let display = html_escape(&percent_encoding::percent_decode_str(url_path).decode_utf8_lossy());
    let title = format!("Directory listing for {display}");
    let mut html = format!(
        "<!DOCTYPE HTML>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<title>{title}</title>\n</head>\n\
         <body>\n<h1>{title}</h1>\n<hr>\n<ul>\n"
    );
    for name in names {
        let full = dir.join(&name);
        let (mut display, mut link) = (name.clone(), name.clone());
        let meta = std::fs::symlink_metadata(&full);
        if full.is_dir() {
            display.push('/');
            link.push('/');
        }
        if meta.map(|m| m.file_type().is_symlink()).unwrap_or(false) {
            display.push('@');
        }
        html.push_str(&format!(
            "<li><a href=\"{}\">{}</a></li>\n",
            quote(&link),
            html_escape(&display)
        ));
    }
    html.push_str("</ul>\n<hr>\n</body>\n</html>\n");
    ok(
        200,
        vec![
            ("Content-type".into(), "text/html; charset=utf-8".into()),
            ("Content-Length".into(), html.len().to_string()),
        ],
        Body::Bytes(html.into_bytes()),
    )
}

/// `parse_byte_range`: `bytes=first-[last]`.
fn parse_byte_range(h: &str) -> std::result::Result<(u64, Option<u64>), ()> {
    let spec = h.strip_prefix("bytes=").ok_or(())?;
    let (a, b) = spec.split_once('-').ok_or(())?;
    if a.is_empty()
        || !a.bytes().all(|c| c.is_ascii_digit())
        || !b.bytes().all(|c| c.is_ascii_digit())
    {
        return Err(());
    }
    let first: u64 = a.parse().map_err(|_| ())?;
    let last: Option<u64> = if b.is_empty() {
        None
    } else {
        Some(b.parse().map_err(|_| ())?)
    };
    if let Some(l) = last
        && l != 0
        && l < first
    {
        return Err(());
    }
    Ok((first, last))
}

/// The response to `method target` with `headers` (names as sent), serving `root`.
pub fn respond_to(
    root: &Path,
    method: &str,
    target: &str,
    headers: &[(String, String)],
) -> Response {
    let header = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    };
    let mut r = match method {
        "GET" | "HEAD" => send_head(
            root,
            target,
            header("Range"),
            header("If-Modified-Since"),
            header("If-None-Match"),
        ),
        m => error(501, &format!("Unsupported method ('{m}')")),
    };
    r.headers
        .push(("Access-Control-Allow-Origin".into(), "*".into()));
    if method == "HEAD" {
        r.body = Body::Bytes(vec![]);
    }
    r
}

fn send_head(
    root: &Path,
    target: &str,
    range: Option<&str>,
    ims: Option<&str>,
    inm: Option<&str>,
) -> Response {
    let path = translate_path(root, target);
    let ctype = guess_type(&path);
    if let Some(range) = range {
        // RangeRequestHandler
        let Ok((first, last)) = parse_byte_range(range) else {
            return error(400, "Invalid byte range");
        };
        let Ok(meta) = std::fs::metadata(&path).and_then(|m| {
            if m.is_dir() {
                Err(std::io::Error::other("dir"))
            } else {
                Ok(m)
            }
        }) else {
            return error(404, "File not found");
        };
        let len = meta.len();
        if first >= len {
            return error(416, "Requested Range Not Satisfiable");
        }
        let last = match last {
            Some(l) if l < len => l,
            _ => len - 1,
        };
        let n = last - first + 1;
        let mut h = vec![
            ("Content-type".into(), ctype.to_string()),
            (
                "Content-Range".into(),
                format!("bytes {first}-{last}/{len}"),
            ),
            ("Content-Length".into(), n.to_string()),
        ];
        if let Ok(t) = meta.modified() {
            h.push(("Last-Modified".into(), http_date(t)));
        }
        return ok(
            206,
            h,
            Body::File {
                path,
                start: first,
                len: n,
            },
        );
    }
    // SimpleHTTPRequestHandler.send_head
    if path.is_dir() {
        let url_path = target.split('?').next().unwrap_or("");
        if !url_path.ends_with('/') {
            let (p, q) = match target.split_once('?') {
                Some((p, q)) => (p.to_string(), format!("?{q}")),
                None => (target.to_string(), String::new()),
            };
            return ok(
                301,
                vec![
                    ("Location".into(), format!("{p}/{q}")),
                    ("Content-Length".into(), "0".into()),
                ],
                Body::Bytes(vec![]),
            );
        }
        let index = ["index.html", "index.htm"]
            .iter()
            .map(|i| path.join(i))
            .find(|p| p.is_file());
        match index {
            Some(i) => return send_file(&i, guess_type(&i), ims, inm),
            None => return list_directory(&path, target),
        }
    }
    if path.to_string_lossy().ends_with('/') {
        return error(404, "File not found");
    }
    send_file(&path, ctype, ims, inm)
}

fn send_file(path: &Path, ctype: &str, ims: Option<&str>, inm: Option<&str>) -> Response {
    let Ok(meta) = std::fs::metadata(path).and_then(|m| {
        if m.is_file() {
            Ok(m)
        } else {
            Err(std::io::Error::other("not a file"))
        }
    }) else {
        return error(404, "File not found");
    };
    let mtime = meta.modified().ok();
    if let (Some(ims), None, Some(mtime)) = (ims, inm, mtime)
        && let Ok(since) = chrono::DateTime::parse_from_rfc2822(ims)
    {
        let secs = mtime
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        if secs <= since.timestamp() {
            return ok(304, vec![], Body::Bytes(vec![]));
        }
    }
    let mut h = vec![
        ("Content-type".into(), ctype.to_string()),
        ("Content-Length".into(), meta.len().to_string()),
    ];
    if let Some(t) = mtime {
        h.push(("Last-Modified".into(), http_date(t)));
    }
    ok(
        200,
        h,
        Body::File {
            path: path.to_path_buf(),
            start: 0,
            len: meta.len(),
        },
    )
}

/// `log_date_time_string()`: local time as `26/Oct/2026 05:48:14`.
fn log_time() -> String {
    chrono::Local::now().format("%d/%b/%Y %H:%M:%S").to_string()
}

fn handle(mut stream: TcpStream, root: &Path) -> std::io::Result<()> {
    let peer = stream
        .peer_addr()
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|_| "-".into());
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(());
    }
    let request_line = line.trim_end_matches(['\r', '\n']).to_string();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    let mut headers = vec![];
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    debug!("{method} {target}");
    let r = respond_to(root, &method, &target, &headers);
    for e in &r.errors {
        eprintln!("{peer} - - [{}] {e}", log_time());
    }
    eprintln!(
        "{peer} - - [{}] \"{request_line}\" {} -",
        log_time(),
        r.status
    );
    let mut head = format!(
        "HTTP/1.0 {} {}\r\nServer: SimpleHTTP/0.6 ome-zarr-rs/{}\r\nDate: {}\r\n",
        r.status,
        r.reason,
        crate::VERSION,
        http_date(std::time::SystemTime::now())
    );
    for (k, v) in &r.headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    match r.body {
        Body::Bytes(b) => stream.write_all(&b)?,
        Body::File { path, start, len } => {
            let mut f = std::fs::File::open(path)?;
            f.seek(SeekFrom::Start(start))?;
            std::io::copy(&mut f.take(len), &mut stream)?;
        }
    }
    Ok(())
}

/// Serve `root` on `port` until the process is killed (Python's
/// `http.server.test`: dual-stack on `::`).
pub fn serve(root: &Path, port: u16) -> Result<()> {
    let (listener, host) = match TcpListener::bind(("::", port)) {
        Ok(l) => (l, "::"),
        Err(_) => (TcpListener::bind(("0.0.0.0", port))?, "0.0.0.0"),
    };
    let url_host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.to_string()
    };
    println!("Serving HTTP on {host} port {port} (http://{url_host}:{port}/) ...");
    let root = root.to_path_buf();
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let root = root.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, &root) {
                debug!("connection error: {e}");
            }
        });
    }
    Ok(())
}

/// Open a URL in the default web browser.
pub fn open_browser(url: &str) {
    let cmd = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    if let Err(e) = std::process::Command::new(cmd).arg(url).spawn() {
        log::warn!("could not open browser ({e}); please visit {url}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serves_over_tcp() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.bin"), b"0123456789").unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let root = dir.path().to_path_buf();
        std::thread::spawn(move || {
            let (s, _) = listener.accept().unwrap();
            handle(s, &root).unwrap();
        });
        let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
        s.write_all(b"GET /f.bin HTTP/1.1\r\nRange: bytes=2-4\r\n\r\n")
            .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        assert!(out.starts_with("HTTP/1.0 206 Partial Content"), "{out}");
        assert!(out.contains("Access-Control-Allow-Origin: *"));
        assert!(out.ends_with("234"));
    }
}
