// This build script is similar to the one in mpsgraph-rs
// It will generate bindings for Metal Performance Shaders Graph framework

fn main() {
    // Only run bindgen on macOS
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }
}
