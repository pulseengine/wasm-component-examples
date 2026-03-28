// Rust P3 async component example
// When built with wasi_version = "p3", the greeting trait methods are async
use hello_p3_bindings::exports::hello::p3::greeting::Guest;

struct Component;

impl Guest for Component {
    async fn greet(name: String) -> String {
        format!("Hello, {}! (async P3 component)", name)
    }
}

hello_p3_bindings::export!(Component with_types_in hello_p3_bindings);
