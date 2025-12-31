use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: {} <number> <operator> <number>", args[0]);
        eprintln!("Example: {} 8 + 8", args[0]);
        std::process::exit(1);
    }

    let left: f64 = match args[1].parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("Error: '{}' is not a valid number", args[1]);
            std::process::exit(1);
        }
    };

    let operator = &args[2];

    let right: f64 = match args[3].parse() {
        Ok(n) => n,
        Err(_) => {
            eprintln!("Error: '{}' is not a valid number", args[3]);
            std::process::exit(1);
        }
    };

    let result = match operator.as_str() {
        "+" => left + right,
        "-" => left - right,
        "*" | "x" => left * right,
        "/" => {
            if right == 0.0 {
                eprintln!("Error: Division by zero");
                std::process::exit(1);
            }
            left / right
        }
        "%" => left % right,
        _ => {
            eprintln!("Error: Unknown operator '{}'. Use +, -, *, /, or %", operator);
            std::process::exit(1);
        }
    };

    // Format result: show as integer if it's a whole number
    if result.fract() == 0.0 {
        println!("{} {} {} = {}", left as i64, operator, right as i64, result as i64);
    } else {
        println!("{} {} {} = {}", left, operator, right, result);
    }
}
