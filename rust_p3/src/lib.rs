// Rust P3 async component example
// The async fn works on WASM targets where wit-bindgen generates async traits.
// On host targets, the trait is sync so we use cfg to handle both.
use hello_p3_bindings::exports::hello::p3::greeting::Guest;

struct Component;

impl Guest for Component {
    #[cfg(target_arch = "wasm32")]
    async fn greet(name: String) -> String {
        format!("Hello, {}! (async P3 component)", name)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn greet(name: String) -> String {
        format!("Hello, {}! (P3 component - host mode)", name)
    }
}

hello_p3_bindings::export!(Component with_types_in hello_p3_bindings);
