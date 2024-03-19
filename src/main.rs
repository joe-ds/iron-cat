//! IronCAT runs simulations of cellular automata right in the terminal.
//! # Features
//! * Animated using terminal codes.
//! * Accepts B/S rulestrings.
//! * Parallelism courtesy of the `rayon` library!

use std::{error, fmt, thread, time};
extern crate clap;
extern crate rand;
extern crate rayon;

use clap::{App, Arg};
use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;

/// This helps us gracefully exit the program while printing the cause.
/// This macro will take in a string and optionally an Error and print them
/// both.
macro_rules! die {
    ($s:expr $(, $e: ident)?) => {
        println!("Error: {}", $s);

        $( println!("{}", $e); )?

        std::process::exit(1);
    }
}

/// Represents one of two main errors with rulestrings.
#[derive(Debug)]
enum RuleError {
    BadString,
    InvalidInt(char),
}

impl error::Error for RuleError {}

impl fmt::Display for RuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            RuleError::BadString => write!(f, "Invalid Rulestring"),
            RuleError::InvalidInt(c) => write!(f, "Invalid value {} in Rulestring", c),
        }
    }
}

/// Represents our rulestrings.
///
/// # Use
/// Rulestrings are arrays of bools, and after finding out how many neighbours
/// a cell has, we can simply use that to index the rulestring to see if the
/// cell is born or survives.
///
/// # Example
///
/// ```
/// let mut rules = Rulestring::new();
/// rules.b[3] = true;
/// let neighbours = 3;
/// assert_eq!(rules.b[neighbours], true);
/// ```
#[derive(Debug)]
struct Rulestring {
    b: [bool; 9],
    s: [bool; 9],
}

/// A new `Rulestring` is an array of false. Applied to any seeded `Matrix`, it
/// will just result in every cell dying in a few iterations.
impl Rulestring {
    fn new() -> Rulestring {
        Rulestring {
            b: [false; 9],
            s: [false; 9],
        }
    }
}

/// Formats the rulestring in B/S notation.
impl fmt::Display for Rulestring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut b = String::from("B");
        let mut s = String::from("S");

        for i in 0..self.b.len() {
            if self.b[i] {
                b.push_str(&i.to_string());
            };
            if self.s[i] {
                s.push_str(&i.to_string());
            };
        }

        write!(f, "{}/{}", b, s)
    }
}

/// Rather than writing some sort of parser, we implement `FromStr`.
impl std::str::FromStr for Rulestring {
    type Err = RuleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut rules = Rulestring::new();

        let bs: Vec<&str> = s.trim().split('/').collect();

        if bs.len() != 2 {
            return Err(RuleError::BadString);
        }

        if bs[0].len() == 1 {
            if bs[0].starts_with('B') {
                rules.b[0] = true;
            } else {
                return Err(RuleError::BadString);
            }
        } else {
            for b in bs[0].chars().skip(1) {
                match b.to_digit(10) {
                    Some(i) => {
                        if i > 0 && i < 9 {
                            rules.b[i as usize] = true;
                        } else {
                            return Err(RuleError::InvalidInt(b));
                        }
                    }
                    None => return Err(RuleError::InvalidInt(b)),
                }
            }
        }

        if bs[1].len() == 1 {
            if bs[1].starts_with('S') {
                rules.s[0] = true;
            } else {
                return Err(RuleError::BadString);
            }
        } else {
            for s in bs[1].chars().skip(1) {
                match s.to_digit(10) {
                    Some(i) => {
                        if i > 0 && i < 9 {
                            rules.s[i as usize] = true;
                        } else {
                            return Err(RuleError::InvalidInt(s));
                        }
                    }
                    None => return Err(RuleError::InvalidInt(s)),
                }
            }
        }

        Ok(rules)
    }
}

/// The main struct used to represent the state of the automata.
struct Matrix {
    m: usize,
    n: usize,
    rules: Rulestring,
    rows: Vec<usize>,
}

impl Matrix {
    /// Technically, here, only one of `n` or `m` would need to be stored as
    /// the other can easily be computed. However for the space of a usize it
    /// is convenient to store them both.
    fn new(m: usize, n: usize, rules: Rulestring) -> Matrix {
        Matrix {
            m,
            n,
            rules,
            rows: vec![0; m * n],
        }
    }

    /// `seed` randomly sets a cell a certain number of times. If the number
    /// isn't provided, then the default is to do this for half the size of
    /// the matrix.
    fn seed(&mut self, cells: Option<usize>) {
        let mut rng = rand::thread_rng();
        let indices = Uniform::from(0..self.rows.len());
        let iterations: usize = match cells {
            Some(n) => n,
            None => self.rows.len() / 2,
        };

        for _ in 0..iterations {
            let i = indices.sample(&mut rng);
            self.rows[i] = 1;
        }
    }

    /// `pulse` mutates the present state by applying the given `Rulestring`.
    ///
    /// To make things easier, we have a 1D array and calculate our indices
    /// mathematically. We also use modular arithmetic to wrap our rows and
    /// columns.
    fn pulse(&mut self) {
        self.rows = self
            .rows
            .par_iter()
            .enumerate()
            .map(|(i, n)| {
                let c = self.n;
                let r = self.m;

                let col = i % c;
                let row = i / c;

                let col_next = (i + 1) % c;
                let col_prev = (i + (c - 1)) % c;

                let row_next = (row + 1) % r;
                let row_prev = (row + (r - 1)) % r;

                let sum = self.rows[(c * row) + col_prev]
                    + self.rows[(c * row) + col_next]
                    + self.rows[(c * row_prev) + col]
                    + self.rows[(c * row_prev) + col_prev]
                    + self.rows[(c * row_prev) + col_next]
                    + self.rows[(c * row_next) + col]
                    + self.rows[(c * row_next) + col_prev]
                    + self.rows[(c * row_next) + col_next];

                if self.rules.b[sum] {
                    1
                } else if self.rules.s[sum] {
                    *n
                } else {
                    0
                }
            })
            .collect();
    }
}

/// The output for the program is generated here. Since this is just a mapping
/// over `Matrix.rows`, it is simple enough to replace this with a different
/// one or even pass the array as output to a different program.
impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = String::new();

        for chunk in self.rows.chunks(self.n) {
            out.push_str(
                &chunk
                    .iter()
                    .map(|x| if *x == 0 { "░░" } else { "▓▓" })
                    .collect::<Vec<&str>>()
                    .join(""),
            );
            out.push('\n');
        }

        write!(f, "{}", out)
    }
}

/// `main` is where our `Matrix` is instantiated and where the output loop is.
///
/// Note that this loop will have to be terminated using ^c or an equivalent.
///
/// `main` also takes care of our arguments using the `clap` library.
fn main() {
    let matches = App::new("Iron Cellular Automata for Terminals")
        .version("1.0")
        .author("Joe Peterson")
        .about("Runs an animated cellular automata simulation in the terminal.")
        .arg(
            Arg::with_name("rows")
                .short("m")
                .long("rows")
                .value_name("ROWS")
                .help("Number of rows")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("columns")
                .short("n")
                .long("columns")
                .value_name("COLUMNS")
                .help("Number of columns")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("rulestring")
                .short("r")
                .long("rulestring")
                .value_name("RULESTRING")
                .help("Rulestring for the automata in B/S notation")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("seed_iter")
                .short("s")
                .long("seed")
                .value_name("SEED")
                .help("Set random cells SEED times")
                .takes_value(true),
        )
        .get_matches();

    let m = match matches.value_of("rows").unwrap_or("23").parse::<usize>() {
        Ok(x) => {
            if x > 0 {
                x
            } else {
                die!("Can't have zero rows.");
            }
        }
        Err(e) => {
            die!("Invalid value for 'rows'.", e);
        }
    };

    let n = match matches.value_of("columns").unwrap_or("38").parse::<usize>() {
        Ok(x) => {
            if x > 0 {
                x
            } else {
                die!("Can't have zero columns.");
            }
        }
        Err(e) => {
            die!("Invalid value for 'columns'.", e);
        }
    };

    let rulestring = match matches
        .value_of("rulestring")
        .unwrap()
        .parse::<Rulestring>()
    {
        Ok(r) => r,
        Err(e) => {
            die!("Invalid Rulestring.", e);
        }
    };

    let mut matrix = Matrix::new(m, n, rulestring);

    match matches.value_of("seed") {
        Some(s) => match s.parse::<usize>() {
            Ok(n) => matrix.seed(Some(n)),
            Err(e) => {
                die!("Invalid value for 'seed'", e);
            }
        },
        None => matrix.seed(None),
    };

    println!("\x1B[2J{}", &matrix);

    loop {
        matrix.pulse();
        println!("\x1B[H{}", &matrix);
        thread::sleep(time::Duration::new(1, 0));
    }
}
