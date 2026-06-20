use std::{
    env, fs,
    path::{Path, PathBuf},
};

// =============================
// OpenMP handling
// =============================
struct OpenMPConfig {
    c_flags: &'static [&'static str],
    link_libs: &'static [&'static str],
    dynamic_search: Vec<String>,
    dynamic_includes: Vec<String>,
    dynamic_link_args: Vec<String>,
}

fn detect_openmp(target: &str, is_msvc: bool, is_clang: bool) -> OpenMPConfig {
    if is_msvc {
        return OpenMPConfig {
            c_flags: &["/openmp"],
            link_libs: &[],
            dynamic_search: vec![],
            dynamic_includes: vec![],
            dynamic_link_args: vec![],
        };
    }
    if target.contains("apple-darwin") {
        let mut search = vec![];
        let mut includes = vec![];
        let mut link_args = vec![];
        if let Ok(prefix_out) = std::process::Command::new("brew").arg("--prefix").output() {
            if let Ok(prefix) = String::from_utf8(prefix_out.stdout) {
                let p = prefix.trim();
                if !p.is_empty() {
                    search.push(format!("{p}/lib"));
                    includes.push(format!("{p}/include"));
                    link_args.push(format!("-Wl,-rpath,{p}/lib"));
                }
            }
        }
        return OpenMPConfig {
            c_flags: &["-Xpreprocessor", "-fopenmp"],
            link_libs: &["omp"],
            dynamic_search: search,
            dynamic_includes: includes,
            dynamic_link_args: link_args,
        };
    }
    // Linux / MinGW
    let link_libs: &'static [&'static str] = if is_clang { &["omp"] } else { &["gomp"] };
    OpenMPConfig {
        c_flags: &["-fopenmp"],
        link_libs,
        dynamic_search: vec![],
        dynamic_includes: vec![],
        dynamic_link_args: vec![],
    }
}

fn emit_openmp(omp: &OpenMPConfig, build: &mut cc::Build) {
    for f in omp.c_flags {
        build.flag(f);
    }
    for s in &omp.dynamic_search {
        println!("cargo:rustc-link-search=native={s}");
    }
    for lib in omp.link_libs {
        println!("cargo:rustc-link-lib={lib}");
    }
    for a in &omp.dynamic_link_args {
        println!("cargo:rustc-link-arg={a}");
    }
    for inc in &omp.dynamic_includes {
        build.include(inc);
    }
    if let Ok(extra) = env::var("OPENMP_LIB_DIR") {
        println!("cargo:rustc-link-search=native={extra}");
    }
    if let Ok(extra_inc) = env::var("OPENMP_INCLUDE_DIR") {
        build.include(extra_inc);
    }
}

// =============================
// RoutingKit sources & patching
// =============================
const RK_ALLOW: &[&str] = &[
    "customizable_contraction_hierarchy.cpp",
    "contraction_hierarchy.cpp",
    "bit_vector.cpp",
    "bit_select.cpp",
    "id_mapper.cpp",
    "permutation.cpp",
    "nested_dissection.cpp",
    "verify.cpp",
    "graph_util.cpp",
    "timer.cpp",
];

/// Collect list of source files (patched versions if needed) to compile.
fn collect_routingkit_files(src_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(src_dir).unwrap() {
        let path = entry.unwrap().path();
        if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
            if RK_ALLOW.contains(&name) {
                if let Some(patcher) = PATCHERS.iter().find(|(n, _)| *n == name).map(|(_, f)| *f) {
                    let patched = emit_patched_with(&path, patcher)
                        .unwrap_or_else(|e| panic!("failed to patch {name}: {e}"));
                    out.push(patched);
                } else {
                    out.push(path);
                }
            }
        }
    }
    out
}

fn main() {
    // 1. Locate RoutingKit
    let rk_dir = env::var("ROUTINGKIT_DIR")
        .ok()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./RoutingKit"));
    let include_dir = rk_dir.join("include");
    let src_dir = rk_dir.join("src");
    if !include_dir.exists() {
        panic!("RoutingKit include dir missing: {:?}", include_dir);
    }
    if !src_dir.exists() {
        panic!("RoutingKit src dir missing: {:?}", src_dir);
    }

    // 2. Prepare CXX build
    let mut build: cc::Build = cxx_build::bridge("src/lib.rs");
    build.include(&include_dir).include("src").include(&src_dir);

    // 3. Add RoutingKit sources (patched if needed)
    for f in collect_routingkit_files(&src_dir) {
        build.file(f);
    }
    build.file("src/routingkit_cch_wrapper.cc");
    build.file("src/cch_bundle.cc");

    if cfg!(target_env = "msvc") {
        build.define("ROUTING_KIT_NO_GCC_EXTENSIONS", None);
    }

    if let Ok(target_triple) = env::var("TARGET") {
        if target_triple.contains("apple-darwin") {
            build.define("ROUTING_KIT_NO_ALIGNED_ALLOC", None);
        }
    }

    // 4. Flags & optimizations
    // Language / optimization
    build
        .flag_if_supported("-std=c++17")
        .flag_if_supported("/std:c++17");
    build.flag_if_supported("-O3").flag_if_supported("/O2");
    // Position independent code
    build.flag_if_supported("-fPIC");
    // Warnings
    build.flag_if_supported("-Wall").flag_if_supported("/W4");
    // Fast math
    build
        .flag_if_supported("-ffast-math")
        .flag_if_supported("/fp:fast");
    // Architecture tune
    if !cfg!(target_env = "msvc") {
        build.flag_if_supported("-march=native");
    }
    // Exceptions (MSVC)
    build.flag_if_supported("/EHsc");
    // Silence some warnings
    for w in [
        "-Wno-unused-parameter",
        "-Wno-psabi",
        "-Wno-unused-variable",
        "-Wno-unused-function",
    ] {
        build.flag_if_supported(w);
    }

    // 5. OpenMP (mandatory parallel)
    let target = env::var("TARGET").unwrap_or_default();
    let compiler = build.get_compiler();
    if std::env::var("CARGO_FEATURE_OPENMP").is_ok() {
        emit_openmp(
            &detect_openmp(&target, compiler.is_like_msvc(), compiler.is_like_clang()),
            &mut build,
        );
    }

    // 6. pthread (POSIX)
    build.flag_if_supported("-pthread");

    // Define NDEBUG for release-like (opt) builds; keep assertions in debug.
    if env::var("PROFILE").map(|p| p == "release").unwrap_or(false) {
        build.define("NDEBUG", None);
    } else {
        println!("cargo:warning=Compiling C++ in debug mode.");
    }

    // 7. Compile
    build.compile("routingkit_cch");

    // 8. Re-run triggers
    for item in ["ROUTINGKIT_DIR"] {
        println!("cargo:rerun-if-env-changed={item}");
    }
    for file in [
        "src/lib.rs",
        "src/routingkit_cch_wrapper.h",
        "src/routingkit_cch_wrapper.cc",
        "src/cch_bundle.h",
        "src/cch_bundle.cc",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }
    println!("cargo:rerun-if-changed={}", include_dir.display());
    println!("cargo:rerun-if-changed={}", src_dir.display());
}

/// Map of file name -> patch transform function.
/// Each transformer receives original file content and must return new content (or panic/todo!).
static PATCHERS: &[(&str, fn(&str) -> String)] = &[
    (
        "customizable_contraction_hierarchy.cpp",
        patch_customizable_contraction_hierarchy,
    ),
    // Add more (filename, function) pairs here as needed.
];

/// Generic emit helper using provided transformer.
fn emit_patched_with(original: &Path, transform: fn(&str) -> String) -> std::io::Result<PathBuf> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set by cargo"));
    let patched_dir = out_dir.join("patched");
    if !patched_dir.exists() {
        fs::create_dir_all(&patched_dir)?;
    }
    let target = patched_dir.join(original.file_name().unwrap());
    let src = fs::read_to_string(original)?;
    let transformed: String = transform(&src);
    fs::write(&target, transformed)?;
    println!("cargo:warning=Patched {:?}", original.file_name().unwrap());
    Ok(target)
}

fn patch_customizable_contraction_hierarchy(original: &str) -> String {
    let mut lines = original.lines().map(|x| x.to_string()).collect::<Vec<_>>();
    for row in [922, 930] {
        lines[row] = lines[row].replace("unsigned", "long");
    }
    lines.join("\n")
}
