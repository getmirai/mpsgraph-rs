use mpsgraph::executable::{CompilationDescriptor, Executable};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Look for the package in the current directory or workspace
    let package_path = env::current_dir()?;

    // First, try a package in the current directory
    let mut possible_package = package_path.join("output.mpsgraphpackage");

    // If not found, try looking in other possible locations
    if !possible_package.exists() {
        // Try in the examples directory
        possible_package = package_path.join("examples").join("output.mpsgraphpackage");

        // Check if file exists in crates/mpsgraph-rs/examples
        if !possible_package.exists() && package_path.ends_with("mpsgraph-rs") {
            possible_package = package_path
                .join("crates")
                .join("mpsgraph-rs")
                .join("examples")
                .join("output.mpsgraphpackage");
        }
    }

    // Allow overriding with an environment variable
    let package_path = if let Ok(env_path) = env::var("MPSGRAPH_PACKAGE_PATH") {
        PathBuf::from(env_path)
    } else {
        possible_package
    };

    // Check if the package exists
    if !package_path.exists() {
        println!("Package not found at: {:?}", package_path);
        println!("Please run the full_flow_example first to create a package,");
        println!(
            "or set MPSGRAPH_PACKAGE_PATH environment variable to point to an existing package."
        );
        return Ok(());
    }

    println!("Loading MPSGraph package from: {:?}", package_path);

    // Create a compilation descriptor (without setting optimization profile)
    let compilation_descriptor = CompilationDescriptor::new();

    // Load the executable from the serialized package
    // Note: from_serialized_package already uses the proper Retained::from_raw pattern internally
    let executable =
        match Executable::from_serialized_package(&package_path, Some(&compilation_descriptor)) {
            Some(exe) => {
                println!("Successfully loaded MPSGraph executable!");
                exe
            }
            None => {
                println!("Failed to load MPSGraph executable!");
                return Ok(());
            }
        };

    // Try to get feed tensors and target tensors to verify the executable is valid
    if let Some(feed_tensors) = executable.feed_tensors() {
        println!("Feed tensors count: {}", feed_tensors.len());
    } else {
        println!("No feed tensors found");
    }

    if let Some(target_tensors) = executable.target_tensors() {
        println!("Target tensors count: {}", target_tensors.len());
    } else {
        println!("No target tensors found");
    }

    println!("Test completed successfully!");
    Ok(())
}
