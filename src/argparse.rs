//! A small emulation of Python's `argparse`, enough for the `ome_zarr` command
//! line to parse, report errors and print help exactly as ome-zarr-py does:
//! options anywhere, `--opt=value`, unique-prefix abbreviations, clustered
//! short flags, `--`, and argparse's usage / help formatting (Python 3.12,
//! 80 columns) and error messages.

use std::collections::BTreeMap;

/// How an argument consumes the command line.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// Takes one value (`--output OUTPUT`, or a positional).
    Store,
    /// A flag (`store_true`).
    StoreTrue,
    /// A counted flag (`-vvv`).
    Count,
    /// `-h, --help`.
    Help,
}

/// A value's type (argparse's `type=`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Type {
    Str,
    Int,
}

/// One argument (`add_argument`).
#[derive(Debug, Clone)]
pub struct Arg {
    /// Option strings in the order given (`["--force", "-f"]`); empty for positionals.
    pub flags: Vec<String>,
    pub dest: String,
    pub kind: Kind,
    pub ty: Type,
    pub choices: Option<Vec<String>>,
    pub default: Option<Value>,
    pub help: Option<String>,
}

impl Arg {
    pub fn positional(dest: &str) -> Arg {
        Arg {
            flags: vec![],
            dest: dest.into(),
            kind: Kind::Store,
            ty: Type::Str,
            choices: None,
            default: None,
            help: None,
        }
    }
    pub fn option(flags: &[&str]) -> Arg {
        let long = flags
            .iter()
            .find(|f| f.starts_with("--"))
            .unwrap_or(&flags[0]);
        let dest = long.trim_start_matches('-').replace('-', "_");
        Arg {
            flags: flags.iter().map(|s| s.to_string()).collect(),
            dest,
            kind: Kind::Store,
            ty: Type::Str,
            choices: None,
            default: None,
            help: None,
        }
    }
    pub fn flag(mut self) -> Arg {
        self.kind = Kind::StoreTrue;
        self.default = Some(Value::Bool(false));
        self
    }
    pub fn count(mut self) -> Arg {
        self.kind = Kind::Count;
        self.default = Some(Value::Int(0));
        self
    }
    pub fn int(mut self) -> Arg {
        self.ty = Type::Int;
        self
    }
    pub fn choices(mut self, c: &[&str]) -> Arg {
        self.choices = Some(c.iter().map(|s| s.to_string()).collect());
        self
    }
    pub fn default(mut self, v: Value) -> Arg {
        self.default = Some(v);
        self
    }
    pub fn help(mut self, h: &str) -> Arg {
        self.help = Some(h.into());
        self
    }

    fn is_positional(&self) -> bool {
        self.flags.is_empty()
    }

    fn metavar(&self) -> String {
        match &self.choices {
            Some(c) => format!("{{{}}}", c.join(",")),
            None if self.is_positional() => self.dest.clone(),
            None => self.dest.to_uppercase(),
        }
    }

    /// The usage part: `[-h]`, `[--output OUTPUT]`, `path`.
    fn usage_part(&self) -> String {
        if self.is_positional() {
            return self.metavar();
        }
        match self.kind {
            Kind::Store => format!("[{} {}]", self.flags[0], self.metavar()),
            _ => format!("[{}]", self.flags[0]),
        }
    }

    /// The help invocation: `-h, --help`, `--port PORT`, `path`.
    fn invocation(&self) -> String {
        if self.is_positional() {
            return self.metavar();
        }
        match self.kind {
            Kind::Store => self
                .flags
                .iter()
                .map(|f| format!("{f} {}", self.metavar()))
                .collect::<Vec<_>>()
                .join(", "),
            _ => self.flags.join(", "),
        }
    }
}

/// A parsed value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Bool(bool),
    None,
}

/// The parse result (argparse's `Namespace`), plus the chosen subcommand.
#[derive(Debug, Clone, Default)]
pub struct Namespace {
    pub values: BTreeMap<String, Value>,
    pub command: Option<String>,
}

impl Namespace {
    pub fn str(&self, k: &str) -> Option<String> {
        match self.values.get(k) {
            Some(Value::Str(s)) => Some(s.clone()),
            _ => None,
        }
    }
    pub fn int(&self, k: &str) -> Option<i64> {
        match self.values.get(k) {
            Some(Value::Int(i)) => Some(*i),
            _ => None,
        }
    }
    pub fn bool(&self, k: &str) -> bool {
        matches!(self.values.get(k), Some(Value::Bool(true)))
    }
}

/// What argparse prints before exiting (`--help`, or an error).
#[derive(Debug, Clone, PartialEq)]
pub struct Exit {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// A parser (`ArgumentParser`), with optional subcommands (`add_subparsers`).
#[derive(Debug, Clone)]
pub struct Parser {
    pub prog: String,
    pub args: Vec<Arg>,
    /// `(dest, [(name, parser)])` of the required subparsers action.
    pub subcommands: Option<(String, Vec<(String, Parser)>)>,
}

/// argparse's line width: `COLUMNS` (80) - 2.
const WIDTH: usize = 78;

impl Parser {
    pub fn new(prog: &str) -> Parser {
        let help = Arg::option(&["-h", "--help"]);
        let help = Arg {
            kind: Kind::Help,
            dest: "help".into(),
            help: Some("show this help message and exit".into()),
            ..help
        };
        Parser {
            prog: prog.into(),
            args: vec![help],
            subcommands: None,
        }
    }

    pub fn arg(mut self, a: Arg) -> Parser {
        self.args.push(a);
        self
    }

    pub fn subcommands(mut self, dest: &str, subs: Vec<(String, Parser)>) -> Parser {
        self.subcommands = Some((dest.into(), subs));
        self
    }

    fn sub_choices(&self) -> Option<String> {
        self.subcommands.as_ref().map(|(_, s)| {
            format!(
                "{{{}}}",
                s.iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
    }

    /// `format_usage()`, without the trailing newline.
    pub fn usage(&self) -> String {
        let prefix = "usage: ";
        let opt_parts: Vec<String> = self
            .args
            .iter()
            .filter(|a| !a.is_positional())
            .map(Arg::usage_part)
            .collect();
        let mut pos_parts: Vec<String> = self
            .args
            .iter()
            .filter(|a| a.is_positional())
            .map(Arg::usage_part)
            .collect();
        if let Some(c) = self.sub_choices() {
            pos_parts.push(c);
            pos_parts.push("...".into());
        }
        let all: Vec<&str> = std::iter::once(self.prog.as_str())
            .chain(opt_parts.iter().map(String::as_str))
            .chain(pos_parts.iter().map(String::as_str))
            .collect();
        let usage = all.join(" ");
        if prefix.len() + usage.len() <= WIDTH {
            return format!("{prefix}{usage}");
        }
        let get_lines = |parts: Vec<&str>, indent: &str, prefix: Option<&str>| -> Vec<String> {
            let mut lines = vec![];
            let mut line: Vec<&str> = vec![];
            let mut line_len = match prefix {
                Some(p) => p.len() as isize - 1,
                None => indent.len() as isize - 1,
            };
            for part in parts {
                if line_len + 1 + part.len() as isize > WIDTH as isize && !line.is_empty() {
                    lines.push(format!("{indent}{}", line.join(" ")));
                    line.clear();
                    line_len = indent.len() as isize - 1;
                }
                line.push(part);
                line_len += part.len() as isize + 1;
            }
            if !line.is_empty() {
                lines.push(format!("{indent}{}", line.join(" ")));
            }
            if prefix.is_some() {
                lines[0] = lines[0][indent.len()..].to_string();
            }
            lines
        };
        let indent = " ".repeat(prefix.len() + self.prog.len() + 1);
        let lines = if !opt_parts.is_empty() {
            let mut l = get_lines(
                std::iter::once(self.prog.as_str())
                    .chain(opt_parts.iter().map(String::as_str))
                    .collect(),
                &indent,
                Some(prefix),
            );
            l.extend(get_lines(
                pos_parts.iter().map(String::as_str).collect(),
                &indent,
                None,
            ));
            l
        } else if !pos_parts.is_empty() {
            get_lines(
                std::iter::once(self.prog.as_str())
                    .chain(pos_parts.iter().map(String::as_str))
                    .collect(),
                &indent,
                Some(prefix),
            )
        } else {
            vec![self.prog.clone()]
        };
        format!("{prefix}{}", lines.join("\n"))
    }

    /// `format_help()`.
    pub fn help(&self) -> String {
        let mut positionals: Vec<(String, Option<String>)> = self
            .args
            .iter()
            .filter(|a| a.is_positional())
            .map(|a| (a.invocation(), a.help.clone()))
            .collect();
        if let Some(c) = self.sub_choices() {
            positionals.push((c, None));
        }
        let optionals: Vec<(String, Option<String>)> = self
            .args
            .iter()
            .filter(|a| !a.is_positional())
            .map(|a| (a.invocation(), a.help.clone()))
            .collect();
        let max_len = positionals
            .iter()
            .chain(&optionals)
            .map(|(i, _)| i.len() + 2)
            .max()
            .unwrap_or(0);
        let help_position = (max_len + 2).min(24);
        let action_width = help_position - 4;
        let help_width = (WIDTH - help_position).max(11);
        let section = |title: &str, items: &[(String, Option<String>)]| -> String {
            let mut s = format!("{title}:\n");
            for (inv, help) in items {
                let Some(help) = help else {
                    s.push_str(&format!("  {inv}\n"));
                    continue;
                };
                let lines = wrap(help, help_width);
                if inv.len() <= action_width {
                    s.push_str(&format!("  {inv:<action_width$}  {}\n", lines[0]));
                } else {
                    s.push_str(&format!(
                        "  {inv}\n{}{}\n",
                        " ".repeat(help_position),
                        lines[0]
                    ));
                }
                for l in &lines[1..] {
                    s.push_str(&format!("{}{l}\n", " ".repeat(help_position)));
                }
            }
            s
        };
        let mut out = format!("{}\n\n", self.usage());
        if !positionals.is_empty() {
            out.push_str(&section("positional arguments", &positionals));
            out.push('\n');
        }
        out.push_str(&section("options", &optionals));
        out
    }

    fn error(&self, msg: &str) -> Exit {
        Exit {
            code: 2,
            stdout: String::new(),
            stderr: format!("{}\n{}: error: {msg}\n", self.usage(), self.prog),
        }
    }

    fn find_option(&self, token: &str) -> std::result::Result<Option<&Arg>, Exit> {
        if let Some(a) = self
            .args
            .iter()
            .find(|a| a.flags.iter().any(|f| f == token))
        {
            return Ok(Some(a));
        }
        // unique prefixes of long options
        if token.starts_with("--") && token.len() > 2 {
            let matches: Vec<(&Arg, &String)> = self
                .args
                .iter()
                .flat_map(|a| a.flags.iter().map(move |f| (a, f)))
                .filter(|(_, f)| f.starts_with("--") && f.starts_with(token))
                .collect();
            match matches.len() {
                0 => {}
                1 => return Ok(Some(matches[0].0)),
                _ => {
                    let names: Vec<&str> = matches.iter().map(|(_, f)| f.as_str()).collect();
                    return Err(self.error(&format!(
                        "ambiguous option: {token} could match {}",
                        names.join(", ")
                    )));
                }
            }
        }
        Ok(None)
    }

    fn convert(&self, a: &Arg, raw: &str) -> std::result::Result<Value, Exit> {
        let name = if a.is_positional() {
            a.dest.clone()
        } else {
            a.flags.join("/")
        };
        let v = match a.ty {
            Type::Str => Value::Str(raw.into()),
            Type::Int => match raw.trim().parse::<i64>() {
                Ok(i) => Value::Int(i),
                Err(_) => {
                    return Err(self.error(&format!("argument {name}: invalid int value: '{raw}'")));
                }
            },
        };
        if let (Some(c), Value::Str(s)) = (&a.choices, &v)
            && !c.contains(s)
        {
            return Err(self.error(&format!(
                "argument {name}: invalid choice: '{s}' (choose from {})",
                c.join(", ")
            )));
        }
        Ok(v)
    }

    /// `parse_known_args`: the namespace and the unrecognized arguments.
    fn parse_known(
        &self,
        argv: &[String],
        ns: &mut Namespace,
    ) -> std::result::Result<Vec<String>, Exit> {
        for a in &self.args {
            if a.kind != Kind::Help {
                ns.values
                    .entry(a.dest.clone())
                    .or_insert(a.default.clone().unwrap_or(Value::None));
            }
        }
        let positionals: Vec<&Arg> = self.args.iter().filter(|a| a.is_positional()).collect();
        let mut next_pos = 0;
        let mut extras = vec![];
        let looks_like_option =
            |t: &str| t.starts_with('-') && t.len() > 1 && t.parse::<f64>().is_err();
        let mut i = 0;
        let mut only_positional = false;
        while i < argv.len() {
            let token = &argv[i];
            i += 1;
            if !only_positional && token == "--" {
                only_positional = true;
                continue;
            }
            if only_positional || !looks_like_option(token) {
                if next_pos < positionals.len() {
                    let a = positionals[next_pos];
                    let v = self.convert(a, token)?;
                    ns.values.insert(a.dest.clone(), v);
                    next_pos += 1;
                } else if let Some((dest, subs)) = &self.subcommands {
                    let Some((_, sub)) = subs.iter().find(|(n, _)| n == token) else {
                        let names: Vec<&str> = subs.iter().map(|(n, _)| n.as_str()).collect();
                        return Err(self.error(&format!(
                            "argument {dest}: invalid choice: '{token}' (choose from {})",
                            names.join(", ")
                        )));
                    };
                    ns.command = Some(token.clone());
                    ns.values.insert(dest.clone(), Value::Str(token.clone()));
                    extras.extend(sub.parse_known(&argv[i..], ns)?);
                    i = argv.len();
                } else {
                    extras.push(token.clone());
                }
                continue;
            }
            // --opt=value
            let (name, inline) = match token.split_once('=') {
                Some((n, v)) if n.starts_with("--") => (n.to_string(), Some(v.to_string())),
                _ => (token.clone(), None),
            };
            // clustered short flags: -vvq, -ofile
            if !name.starts_with("--") && name.len() > 2 && inline.is_none() {
                let mut handled = true;
                let chars: Vec<char> = name[1..].chars().collect();
                let mut k = 0;
                while k < chars.len() {
                    let flag = format!("-{}", chars[k]);
                    match self.args.iter().find(|a| a.flags.contains(&flag)) {
                        Some(a) if a.kind == Kind::Store => {
                            let rest: String = chars[k + 1..].iter().collect();
                            let raw = if !rest.is_empty() {
                                rest
                            } else if i < argv.len() && !looks_like_option(&argv[i]) {
                                i += 1;
                                argv[i - 1].clone()
                            } else {
                                return Err(self.error(&format!(
                                    "argument {}: expected one argument",
                                    a.flags.join("/")
                                )));
                            };
                            let v = self.convert(a, &raw)?;
                            ns.values.insert(a.dest.clone(), v);
                            k = chars.len();
                        }
                        Some(a) => {
                            self.apply_flag(a, ns)?;
                            k += 1;
                        }
                        None => {
                            handled = false;
                            break;
                        }
                    }
                }
                if !handled {
                    extras.push(token.clone());
                }
                continue;
            }
            let Some(a) = self.find_option(&name)? else {
                extras.push(token.clone());
                continue;
            };
            if a.kind == Kind::Store {
                let raw = match inline {
                    Some(v) => v,
                    None if i < argv.len() && !looks_like_option(&argv[i]) => {
                        i += 1;
                        argv[i - 1].clone()
                    }
                    None => {
                        return Err(self.error(&format!(
                            "argument {}: expected one argument",
                            a.flags.join("/")
                        )));
                    }
                };
                let v = self.convert(a, &raw)?;
                ns.values.insert(a.dest.clone(), v);
            } else {
                if let Some(v) = inline {
                    return Err(self.error(&format!(
                        "argument {}: ignored explicit argument '{v}'",
                        a.flags.join("/")
                    )));
                }
                self.apply_flag(a, ns)?;
            }
        }
        let mut missing: Vec<String> = positionals[next_pos..]
            .iter()
            .map(|a| a.dest.clone())
            .collect();
        if let Some((dest, _)) = &self.subcommands
            && ns.command.is_none()
        {
            missing.push(dest.clone());
        }
        if !missing.is_empty() {
            return Err(self.error(&format!(
                "the following arguments are required: {}",
                missing.join(", ")
            )));
        }
        Ok(extras)
    }

    fn apply_flag(&self, a: &Arg, ns: &mut Namespace) -> std::result::Result<(), Exit> {
        match a.kind {
            Kind::Help => Err(Exit {
                code: 0,
                stdout: self.help(),
                stderr: String::new(),
            }),
            Kind::StoreTrue => {
                ns.values.insert(a.dest.clone(), Value::Bool(true));
                Ok(())
            }
            Kind::Count => {
                let n = ns.int(&a.dest).unwrap_or(0);
                ns.values.insert(a.dest.clone(), Value::Int(n + 1));
                Ok(())
            }
            Kind::Store => unreachable!("store options take a value"),
        }
    }

    /// `parse_args`: a namespace, or what to print and the exit code.
    pub fn parse(&self, argv: &[String]) -> std::result::Result<Namespace, Exit> {
        let mut ns = Namespace::default();
        let extras = self.parse_known(argv, &mut ns)?;
        if !extras.is_empty() {
            return Err(self.error(&format!("unrecognized arguments: {}", extras.join(" "))));
        }
        Ok(ns)
    }
}

/// `textwrap.wrap(text, width)` for help strings (whitespace collapsed).
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = vec![];
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() || lines.is_empty() {
        lines.push(line);
    }
    lines
}
