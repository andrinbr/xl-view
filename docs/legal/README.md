# Checking licenses and generating notices

Run these commands from the repository root.

## Install the tools

```console
cargo install --locked cargo-deny
cargo install --locked cargo-about --features="cli"
```

## Check dependency licenses

```console
cargo deny --locked check licenses sources
```

This checks the locked dependency graph against `deny.toml`. A successful run
ends with:

```text
licenses ok, sources ok
```

If the check rejects a dependency, inspect that exact package version and its
license before changing the policy. Keep `deny.toml` and `about.toml` aligned,
and prefer a package-specific exception over allowing an unusual license for
every dependency.

## Generate the notices

```console
cargo about generate --workspace --locked --fail \
  --output-file THIRD-PARTY-LICENSES.html docs/legal/about.hbs
```

Review and commit the updated `THIRD-PARTY-LICENSES.html`. Run this command
after changing `Cargo.toml` or `Cargo.lock`. The CI checks that the committed file
is current.

Cargo tools only inspect Rust packages. Notices for bundled fonts, icons, and
other assets must be maintained separately.
