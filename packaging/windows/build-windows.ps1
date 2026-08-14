#Requires -Version 7.6

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

# Fail when an expected packaging input or output is missing or empty.
function Assert-NonEmptyFile([string] $Path) {
    if ((Get-Item -LiteralPath $Path).Length -eq 0) {
        throw "$Path is empty."
    }
}

# Resolve every path from the script location so the script works from any directory.
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$packagerConfig = Join-Path $PSScriptRoot "Packager.toml"
$outputDirectory = Join-Path $repositoryRoot "target\windows"
$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) "xl-view-windows-$([guid]::NewGuid())"
$targetTriple = "x86_64-pc-windows-msvc"

# Cargo reads .cargo/config.toml from the working directory, not from the manifest
# path, so every Cargo command must run from the repository root to pick up crt-static.
Push-Location $repositoryRoot

try {
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null

    # Read the root package version and ensure the standalone Packager config agrees.
    $metadata = cargo metadata --format-version 1 --no-deps --locked | ConvertFrom-Json
    $rootPackages = @($metadata.packages | Where-Object { $_.name -eq "xl-view" })
    if ($rootPackages.Count -ne 1) {
        throw "Cargo metadata did not contain exactly one xl-view package."
    }

    $version = $rootPackages[0].version
    if ((Get-Content -LiteralPath $packagerConfig) -notcontains "version = `"$version`"") {
        throw "$packagerConfig version does not match xl-view $version."
    }

    $portableName = "xl-view-v$version-windows-x86_64-portable.zip"
    $setupName = "xl-view-v$version-windows-x86_64-setup.exe"
    $portableArchive = Join-Path $temporaryRoot $portableName

    # Build the release executable in Cargo's configured target directory.
    cargo build --release --locked --target $targetTriple
    $releaseExecutable = Join-Path $metadata.target_directory "$targetTriple\release\xl-view.exe"
    Assert-NonEmptyFile $releaseExecutable

    # Confirm the executable does not import the redistributable Visual C++ runtime.
    $rustHost = rustc --print host-tuple
    $llvmReadObj = Join-Path (rustc --print sysroot) "lib\rustlib\$rustHost\bin\llvm-readobj.exe"
    Assert-NonEmptyFile $llvmReadObj
    $imports = & $llvmReadObj --coff-imports $releaseExecutable
    $forbiddenImport = $imports | Select-String -Pattern '(?i)\b(?:vcruntime|msvcp)[a-z0-9_]*\.dll\b' |
        Select-Object -First 1
    if ($forbiddenImport) {
        throw "$releaseExecutable imports $($forbiddenImport.Matches[0].Value); the Visual C++ runtime is not static."
    }

    # Stage the portable directory with the executable and its license files.
    $portableRoot = Join-Path $temporaryRoot "portable\XL-View"
    $licenseDirectory = Join-Path $portableRoot "licenses"
    New-Item -ItemType Directory -Path $licenseDirectory | Out-Null
    Copy-Item -LiteralPath $releaseExecutable -Destination (Join-Path $portableRoot "xl-view.exe")

    # Keep this list in sync with Packager.toml so both packages ship the same notices.
    $licenses = @(
        "LICENSE"
        "COPYRIGHT"
        "docs\legal\THIRD-PARTY-LICENSES.html"
        "assets\fonts\AdwaitaSans-LICENSE.txt"
    )
    foreach ($relativePath in $licenses) {
        $source = Join-Path $repositoryRoot $relativePath
        Assert-NonEmptyFile $source
        Copy-Item -LiteralPath $source -Destination (Join-Path $licenseDirectory (Split-Path $relativePath -Leaf))
    }

    # Create the portable ZIP with XL-View as its top-level directory.
    Compress-Archive -LiteralPath $portableRoot -DestinationPath $portableArchive -CompressionLevel Optimal

    # Stage the binary where cargo-packager expects it and build the NSIS installer.
    $packagerDirectory = Join-Path $temporaryRoot "packager"
    New-Item -ItemType Directory -Path $packagerDirectory | Out-Null
    Copy-Item -LiteralPath $releaseExecutable -Destination (Join-Path $packagerDirectory "xl-view.exe")
    cargo packager --config $packagerConfig --out-dir $packagerDirectory --target $targetTriple

    $generatedInstallers = @(Get-ChildItem -LiteralPath $packagerDirectory -Filter "*-setup.exe" -File)
    if ($generatedInstallers.Count -ne 1) {
        throw "cargo-packager did not produce exactly one NSIS installer."
    }
    $setupExecutable = $generatedInstallers[0].FullName

    # Confirm both finished artifacts are non-empty before publishing them.
    foreach ($artifact in @($portableArchive, $setupExecutable)) {
        Assert-NonEmptyFile $artifact
    }

    # Publish the current artifacts, replacing only files for the same version.
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    Copy-Item -LiteralPath $portableArchive -Destination (Join-Path $outputDirectory $portableName) -Force
    Copy-Item -LiteralPath $setupExecutable -Destination (Join-Path $outputDirectory $setupName) -Force

    Write-Output "Windows packages created:"
    Write-Output "  $(Join-Path $outputDirectory $portableName)"
    Write-Output "  $(Join-Path $outputDirectory $setupName)"
}
finally {
    Pop-Location
    if (Test-Path -LiteralPath $temporaryRoot) {
        try {
            Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
        }
        catch {
            Write-Warning "Failed to remove temporary packaging directory '$temporaryRoot': $($_.Exception.Message)"
        }
    }
}
