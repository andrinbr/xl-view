# Windows

The Windows release contains a portable ZIP and an NSIS Setup executable. Both
packages contain `xl-view.exe` and the license notices under `licenses/`. The
installer uses a per-user installation and registers the file associations.

## Build

The build requires PowerShell 7.6 or newer, the repository's pinned Rust toolchain,
and cargo-packager. Install cargo-packager if necessary:

```console
cargo install cargo-packager --version 0.11.8 --locked
```

From the repository root, run:

```console
pwsh -File packaging/windows/build-windows.ps1
```

The script compiles the release binary, checks its static CRT linkage, creates
both packages, and confirms that the finished artifacts are non-empty. The
results are written to:

```text
target/windows/xl-view-vX.Y.Z-windows-x86_64-portable.zip
target/windows/xl-view-vX.Y.Z-windows-x86_64-setup.exe
```
