# scripts/prepush_check.ps1
# Read-only verification script for prepush safety checks.
# Returns non-zero exit code on failure.

$ErrorActionPreference = "Stop"
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  PRE-PUSH SAFETY VERIFICATION CHECK" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$failed = $false

# 1. Tracked forbidden files check
Write-Host "`n[1/5] Checking for forbidden tracked files in git..." -NoNewline
$trackedFiles = git -c core.quotepath=false ls-files
$forbiddenMatches = $trackedFiles | Where-Object {
    $_ -match '^\.env$' -or
    $_ -match '\.db$' -or
    $_ -match '\.pyc$' -or
    $_ -match 'data_vector_db'
}

if ($forbiddenMatches) {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "  Found forbidden files tracked in git:" -ForegroundColor Red
    $forbiddenMatches | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    $failed = $true
} else {
    Write-Host " PASSED" -ForegroundColor Green
}

# 2. Secret scan over tracked files
Write-Host "`n[2/5] Scanning tracked files for secret patterns..." -NoNewline
$secretPatterns = @(
    'gsk_[A-Za-z0-9_-]{15,}',
    'tvly-[A-Za-z0-9_-]{15,}',
    'hf_[A-Za-z0-9_-]{15,}',
    'lsv2_[A-Za-z0-9_-]{15,}',
    'sk-[A-Za-z0-9_-]{20,}'
)

$foundSecrets = @()
foreach ($file in $trackedFiles) {
    if (Test-Path $file -PathType Leaf) {
        $ext = [System.IO.Path]::GetExtension($file)
        if ($ext -notin @('.pdf', '.png', '.jpg', '.jpeg', '.ico', '.parquet')) {
            $content = Get-Content $file -Raw -ErrorAction SilentlyContinue
            if ($content) {
                foreach ($pattern in $secretPatterns) {
                    if ($content -match $pattern) {
                        $foundSecrets += "$file (matched: $($Matches[0].Substring(0, [Math]::Min(10, $Matches[0].Length)))...)"
                    }
                }
            }
        }
    }
}

if ($foundSecrets.Count -gt 0) {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "  Potential credentials detected in tracked files:" -ForegroundColor Red
    $foundSecrets | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    $failed = $true
} else {
    Write-Host " PASSED" -ForegroundColor Green
}

# 3. Check POSH PDF vs Maternity PDF hash
Write-Host "`n[3/5] Verifying POSH Act PDF is distinct from Maternity Benefit Act PDF..." -NoNewline
$poshPdf = "data_raw/Sexual Harassment of Women at Workplace Act.pdf"
$maternityPdf = "data_raw/THE MATERNITY BENEFIT ACT.pdf"

if (-not (Test-Path $poshPdf) -or -not (Test-Path $maternityPdf)) {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "  One or both PDF files missing in data_raw!" -ForegroundColor Red
    $failed = $true
} else {
    $hash1 = (Get-FileHash -Path $poshPdf -Algorithm SHA256).Hash
    $hash2 = (Get-FileHash -Path $maternityPdf -Algorithm SHA256).Hash

    if ($hash1 -eq $hash2) {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  POSH Act PDF is an identical duplicate of Maternity Benefit Act PDF ($hash1)" -ForegroundColor Red
        $failed = $true
    } else {
        Write-Host " PASSED" -ForegroundColor Green
        Write-Host "  POSH hash:      $hash1" -ForegroundColor DarkGray
        Write-Host "  Maternity hash: $hash2" -ForegroundColor DarkGray
    }
}

# 4. Check requirements-docker.txt for banned packages
Write-Host "`n[4/5] Checking requirements-docker.txt for banned packages..." -NoNewline
$dockerReqs = Get-Content "requirements-docker.txt" -Raw
$bannedPackages = @("torch", "transformers", "ragas", "pandas", "datasets")
$foundBanned = @()

foreach ($pkg in $bannedPackages) {
    if ($dockerReqs -match "(?im)^$pkg([=<>!]|\s|$)") {
        $foundBanned += $pkg
    }
}

if ($foundBanned.Count -gt 0) {
    Write-Host " FAILED" -ForegroundColor Red
    Write-Host "  Found banned heavy dependencies in requirements-docker.txt:" -ForegroundColor Red
    $foundBanned | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    $failed = $true
} else {
    Write-Host " PASSED" -ForegroundColor Green
}

# 5. Run test suite
Write-Host "`n[5/5] Running offline unit and integration test suite..." -ForegroundColor Cyan
$pythonExe = if (Test-Path ".\venv\Scripts\python.exe") { ".\venv\Scripts\python.exe" } else { "python" }

& $pythonExe -m unittest discover -s test -p "test_*.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Tests FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
    $failed = $true
} else {
    Write-Host "  All tests PASSED successfully." -ForegroundColor Green
}

Write-Host "`n=========================================" -ForegroundColor Cyan
if ($failed) {
    Write-Host "  PRE-PUSH CHECK FAILED. Do not push until issues are resolved." -ForegroundColor Red
    Write-Host "=========================================" -ForegroundColor Cyan
    exit 1
} else {
    Write-Host "  ALL PRE-PUSH CHECKS PASSED. Ready for prepush verification." -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Cyan
    exit 0
}
