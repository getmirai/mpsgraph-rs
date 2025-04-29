// This build script is similar to the one in mpsgraph-rs
// It will generate bindings for Metal Performance Shaders Graph framework

fn main() {
    // Run on both macOS and iOS
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
    }
}
